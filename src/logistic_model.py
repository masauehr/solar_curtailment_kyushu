"""九州エリアで「日照時間＋曜日（土日祝）」から出力制御の有無をどこまで予測できるかを
簡単なロジスティック回帰で測る（次にやるなら①の実施）。

特徴量:
- sunshine_h: 福岡の日照時間（晴天の持続性の代理。README「発見・注意」で全天日射量より判別力が高いことを確認済み）
- is_weekend_or_holiday: 土日または祝日（低需要の代理）
評価: 時系列として前から7割を学習・残り3割をテストに分け、ROC-AUC・適合率・再現率を報告する
（ランダム分割だと晴天が多い季節に学習・テストが両方入り楽観的になりやすいため、時系列分割を使う）。
"""
from pathlib import Path

import jpholiday
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_RATIO = 0.7


def load_capacity_daily() -> pd.Series:
    """資源エネルギー庁FITポータルの四半期データ(fetch_pv_capacity.py)を日次に線形補間する。"""
    cap = pd.read_csv(PROCESSED_DIR / "kyushu_pv_capacity_quarterly.csv", parse_dates=["date"])
    cap = cap.set_index("date")["capacity_kw"].sort_index()
    daily_index = pd.date_range(cap.index.min(), cap.index.max(), freq="D")
    return cap.reindex(daily_index).interpolate("linear")


def build_dataset() -> pd.DataFrame:
    curtail = pd.read_csv(PROCESSED_DIR / "kyushu_curtail_days.csv", parse_dates=["date"])
    sun = pd.read_csv(PROCESSED_DIR / "sunshine_fukuoka.csv", parse_dates=["date"])

    df = sun.copy()
    df["is_curtailed"] = df["date"].isin(curtail["date"]).astype(int)
    df["weekday"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["weekday"].isin([5, 6])
    df["is_holiday"] = df["date"].apply(lambda d: jpholiday.is_holiday(d.date()))
    df["is_weekend_or_holiday"] = (df["is_weekend"] | df["is_holiday"]).astype(int)
    df = df.dropna(subset=["sunshine_h"]).sort_values("date").reset_index(drop=True)

    # 「次にやるなら」#7: 単一点(福岡)より九州全体の面で見た方が出力制御予測の精度が
    # 上がるかを検証する。(a)気象庁AMeDAS7県平均、(b)農研機構AMGSDSの九州メッシュ面平均、の2通り。
    kyushu_mean = pd.read_csv(PROCESSED_DIR / "sunshine_kyushu_mainland_mean.csv", parse_dates=["date"])
    df = df.merge(kyushu_mean[["date", "sunshine_h_kyushu_mean"]], on="date", how="left")
    amgsds_mesh = pd.read_csv(PROCESSED_DIR / "sunshine_amgsds_kyushu_mesh_mean.csv", parse_dates=["date"])
    df = df.merge(amgsds_mesh[["date", "sunshine_h_amgsds_mesh_mean"]], on="date", how="left")

    # 導入設備容量の増加を素朴に近似するトレンド項（開始日からの経過年数）。
    # 制御頻度が年々増えているのは主に天気ではなく太陽光の導入量増加が効いていると見て、
    # 天気・曜日の効果と分離できるかを確認するために追加。
    df["years_since_start"] = (df["date"] - df["date"].min()).dt.days / 365.25

    # 「次にやるなら」#4: 経過年数の素朴な近似ではなく、資源エネルギー庁FITポータルの
    # 実際の導入容量(九州本土7県、百万kW)に置き換えたらトレンド項として優れるかを検証する。
    capacity_daily = load_capacity_daily()
    df["capacity_million_kw"] = df["date"].map(capacity_daily) / 1e6
    return df


def evaluate(df: pd.DataFrame, features: list, label: str) -> dict:
    n_train = int(len(df) * TRAIN_RATIO)
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train["is_curtailed"])

    proba = model.predict_proba(test[features])[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(test["is_curtailed"], proba)
    precision = precision_score(test["is_curtailed"], pred, zero_division=0)
    recall = recall_score(test["is_curtailed"], pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test["is_curtailed"], pred).ravel()

    print(f"--- {label} ---")
    print(f"  特徴量: {features}")
    print(f"  係数: {dict(zip(features, model.coef_[0].round(3)))} / 切片: {model.intercept_[0]:.3f}")
    print(f"  学習期間: {train['date'].min().date()}〜{train['date'].max().date()}（{len(train)}日、制御日{train['is_curtailed'].sum()}日）")
    print(f"  評価期間: {test['date'].min().date()}〜{test['date'].max().date()}（{len(test)}日、制御日{test['is_curtailed'].sum()}日）")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  しきい値0.5: 適合率{precision:.3f} 再現率{recall:.3f}  (TN={tn} FP={fp} FN={fn} TP={tp})")
    print()
    return {"label": label, "features": features, "auc": auc, "precision": precision, "recall": recall}


def main():
    df = build_dataset()
    df["month"] = df["date"].dt.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    month_cols = list(month_dummies.columns)

    results = []
    results.append(evaluate(df, ["is_weekend_or_holiday"], "曜日（土日祝）のみ"))
    results.append(evaluate(df, ["sunshine_h"], "日照時間のみ"))
    results.append(evaluate(df, ["sunshine_h", "is_weekend_or_holiday"], "日照時間＋曜日（土日祝）"))
    results.append(evaluate(df, ["sunshine_h", "is_weekend_or_holiday", "years_since_start"],
                             "日照時間＋曜日＋トレンド（経過年数の近似）"))
    results.append(evaluate(df, ["sunshine_h", "is_weekend_or_holiday", "capacity_million_kw"],
                             "日照時間＋曜日＋トレンド（実際の導入容量）"))
    results.append(evaluate(df, month_cols + ["years_since_start"], "月（季節）のみ＋トレンド（経過年数）"))
    results.append(evaluate(df, month_cols + ["capacity_million_kw"], "月（季節）のみ＋トレンド（導入容量）"))
    results.append(evaluate(df, month_cols + ["sunshine_h", "is_weekend_or_holiday", "years_since_start"],
                             "月＋日照＋曜日＋トレンド（経過年数、フル）"))
    results.append(evaluate(df, month_cols + ["sunshine_h", "is_weekend_or_holiday", "capacity_million_kw"],
                             "月＋日照＋曜日＋トレンド（実際の導入容量、フル）"))

    # 「次にやるなら」#7: 単一点(福岡)より九州全体の面で見た方が精度が上がるか
    df_mesh = df.dropna(subset=["sunshine_h_kyushu_mean", "sunshine_h_amgsds_mesh_mean"]).reset_index(drop=True)
    results.append(evaluate(df_mesh, ["sunshine_h", "is_weekend_or_holiday", "years_since_start"],
                             "日照（福岡単一点）＋曜日＋トレンド"))
    results.append(evaluate(df_mesh, ["sunshine_h_kyushu_mean", "is_weekend_or_holiday", "years_since_start"],
                             "日照（AMeDAS九州7県平均）＋曜日＋トレンド"))
    results.append(evaluate(df_mesh, ["sunshine_h_amgsds_mesh_mean", "is_weekend_or_holiday", "years_since_start"],
                             "日照（AMGSDS九州メッシュ面平均）＋曜日＋トレンド"))
    results.append(evaluate(df_mesh, month_cols + ["sunshine_h", "is_weekend_or_holiday", "years_since_start"],
                             "フル（福岡単一点）"))
    results.append(evaluate(df_mesh, month_cols + ["sunshine_h_kyushu_mean", "is_weekend_or_holiday", "years_since_start"],
                             "フル（AMeDAS九州7県平均）"))
    results.append(evaluate(df_mesh, month_cols + ["sunshine_h_amgsds_mesh_mean", "is_weekend_or_holiday", "years_since_start"],
                             "フル（AMGSDS九州メッシュ面平均）"))

    print("=== まとめ ===")
    for r in results:
        print(f"  {r['label']:28s} AUC={r['auc']:.3f}  適合率={r['precision']:.3f}  再現率={r['recall']:.3f}")


if __name__ == "__main__":
    main()
