"""九州の過去の出力制御実績に対し、「AMGSDS模擬予報（快晴バイアス補正済み）」と「天気カテゴリの簡略化」
のどちらが出力制御の予測に有効かをバックテストする。

ユーザーの提案: AMGSDSの実際の予報vintageアーカイブは福岡には存在しないため、代わりに
**実測の日照時間（確定後の真値）に、ml_forecastの実証済みバイアスを加えて「AMGSDSが
その日こう予報していただろう」という値を合成**する。バイアスは天気カテゴリ別（快晴/晴れ/
薄曇り〜曇り/曇り・雨、日照率で分類）に異なり、ml_forecastのWEATHER_ACCURACY_VALIDATION.md
のlead=1（前日予測）の値を使う。

簡略化側も同様に、実測から求めた「実現したカテゴリ」を、forecast_narrative.pyと同じ
3区分（晴れ系→75%ile/曇り系→50%ile/雨雪系→10%ile）にマッピングして求める
（過去のJMA予報テキストのアーカイブが無いため、「カテゴリ予報は概ね当たっていた」という
前提のベストケース近似。両手法とも「実際の天気カテゴリが分かっている」という同じ前提に立つ
ことで、公平な比較にしている）。

注意: これは実際の予報アーカイブを使った検証ではなく、既知のバイアス特性を使った**シミュレーション**
であることを明記する。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from logistic_model import build_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FUKUOKA_LAT = 33.6064
TRAIN_RATIO = 0.7

# ml_forecast WEATHER_ACCURACY_VALIDATION.md のlead=1（前日予測）SSDバイアス（予測-実測、h）
BIAS_LEAD1 = {
    "快晴": -5.30,
    "晴れ": -0.97,
    "薄曇り〜曇り": +2.74,
    "曇り・雨": +2.43,
}
# 簡略化側のマッピング（forecast_narrative.pyのweather_code_to_sunshine_proxyと同じ3区分）
REALIZED_TO_SIMPLIFIED_BUCKET = {
    "快晴": "q75", "晴れ": "q75",
    "薄曇り〜曇り": "q50",
    "曇り・雨": "q10",
}


def daylength_hours(date: pd.Timestamp, lat_deg: float = FUKUOKA_LAT) -> float:
    """標準的な天文計算による可照時間（大気差等は簡略化）。"""
    doy = date.dayofyear
    decl = np.radians(23.44) * np.sin(np.radians(360 / 365 * (doy - 81)))
    lat = np.radians(lat_deg)
    cos_h = -np.tan(lat) * np.tan(decl)
    cos_h = np.clip(cos_h, -1, 1)
    return 24 / np.pi * np.arccos(cos_h)


def classify_weather(sunshine_h: float, daylen: float) -> str:
    ratio = np.clip(sunshine_h / daylen * 100, 0, 100) if daylen > 0 else 0
    if ratio >= 80:
        return "快晴"
    elif ratio >= 40:
        return "晴れ"
    elif ratio >= 20:
        return "薄曇り〜曇り"
    else:
        return "曇り・雨"


def build_backtest_dataset() -> pd.DataFrame:
    df = build_dataset()  # date, sunshine_h(実測), is_curtailed, is_weekend_or_holiday, years_since_start
    df["daylen"] = df["date"].apply(daylength_hours)
    df["weather_class"] = df.apply(lambda r: classify_weather(r["sunshine_h"], r["daylen"]), axis=1)

    df["amgsds_proxy_h"] = (df["sunshine_h"] + df["weather_class"].map(BIAS_LEAD1)).clip(lower=0)

    q = {
        "q75": df["sunshine_h"].quantile(0.75),
        "q50": df["sunshine_h"].quantile(0.50),
        "q10": df["sunshine_h"].quantile(0.10),
    }
    df["simplified_h"] = df["weather_class"].map(REALIZED_TO_SIMPLIFIED_BUCKET).map(q)
    return df


def evaluate(df: pd.DataFrame, sunshine_col: str, label: str) -> dict:
    features = [sunshine_col, "is_weekend_or_holiday", "years_since_start"]
    n_train = int(len(df) * TRAIN_RATIO)
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    model = LogisticRegression()
    model.fit(train[features], train["is_curtailed"])
    proba = model.predict_proba(test[features])[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(test["is_curtailed"], proba)
    precision = precision_score(test["is_curtailed"], pred, zero_division=0)
    recall = recall_score(test["is_curtailed"], pred, zero_division=0)
    print(f"--- {label} ---")
    print(f"  特徴量: {sunshine_col}")
    print(f"  ROC-AUC: {auc:.3f}  適合率: {precision:.3f}  再現率: {recall:.3f}")
    return {"label": label, "auc": auc, "precision": precision, "recall": recall}


def main():
    df = build_backtest_dataset()
    print("=== 天気カテゴリの分布（実測日照時間から分類） ===")
    print(df["weather_class"].value_counts())
    print()
    print("=== カテゴリ別バイアス適用後の日照時間サンプル（先頭5件） ===")
    print(df[["date", "sunshine_h", "weather_class", "amgsds_proxy_h", "simplified_h"]].head())
    print()

    results = []
    results.append(evaluate(df, "sunshine_h", "実測そのまま（オラクル、参考上限）"))
    results.append(evaluate(df, "amgsds_proxy_h", "AMGSDS模擬予報（快晴バイアス補正シミュレーション）"))
    results.append(evaluate(df, "simplified_h", "簡略化（天気カテゴリ→分位点）"))

    print("\n=== まとめ ===")
    for r in results:
        print(f"  {r['label']:36s} AUC={r['auc']:.3f}  適合率={r['precision']:.3f}  再現率={r['recall']:.3f}")


if __name__ == "__main__":
    main()
