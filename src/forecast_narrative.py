"""ID-01のGenAI角: 翌日の日照時間予報から、明日の出力制御の可能性を
**明確な判定＋確率**で表示し、補足としてClaudeの平文説明も生成する（次にやるなら②の実施）。

モデルは本リポジトリで最も精度が高かった構成（追加検証8、ROC-AUC 0.924）を使う:
月（季節ダミー）＋日照時間（九州メッシュ面平均）＋曜日（土日祝）＋トレンド（経過年数）。

パイプライン:
1. 日照時間の予報値（九州メッシュ面平均）を取得する。優先順位:
   a. 農研機構メッシュ農業気象データ（AMGSDS）の数値予報（SSD、九州本土メッシュの陸地セル平均）
      ※ CLAUDE.mdで例外的に使用を許可した第三者API。**注意（README.md「追加検証3」参照）**:
         AMGSDSが簡略化より「精度が高い」という根拠は無い。むしろ`ml_forecast`プロジェクトの
         実測検証では、快晴日（出力制御が最も起きやすい条件）にSSD予報が-5h前後系統的に
         過少評価する既知のバイアスがある。
   b. AMGSDSが使えない場合（API障害・トークン失効等）のフォールバックとして、気象庁 bosai の
      短期予報JSON（福岡県 400000）の天気カテゴリ（晴れ/曇り/雨系）を、実測の「九州7県AMeDAS
      平均」日照時間分布の分位点に置き換える簡略化を使う。
2. 明日が土日祝かどうかをjpholidayで判定
3. logistic_model.pyと同じ特徴量（月＋日照＋曜日＋トレンド）でロジスティック回帰を学習し、
   明日の出力制御確率を予測
4. 確率を3段階（低い/中程度/高い）の明確な判定ラベルに変換して一番最初に表示する
   （生の確率だけでは分かりにくいとのユーザー指摘を受けて追加）
5. 予報値・出典・曜日・モデル確率（数値のみ）をClaude（テキストのみ）に渡し、
   一般向けの補足説明を生成する（数値の再計算はさせず解釈のみ、ID-35のexplain_report.pyと同じ方針）
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import jpholiday
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

from logistic_model import build_dataset

sys.path.insert(0, "/Users/masahiro/projects/common")

ROOT_DIR = Path(__file__).resolve().parents[1]  # solar_curtailment_kyushu/（独立リポジトリのルート）
load_dotenv(ROOT_DIR / ".env")

import anthropic

MODEL = "claude-sonnet-5"
FORECAST_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/400000.json"
AREA_NAME = "福岡地方"
KYUSHU_BBOX = [31.0, 34.0, 129.5, 131.9]  # 九州本土をおおむね覆う範囲
SEA_FILL_THRESHOLD = 24.0  # 1日の日照時間は最大でも24h。これを超える値は海上等の欠損値

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# 確率→3段階ラベルの境界（モデルの閾値0.5運用を踏まえた目安。断定ではない）
PROBA_BANDS = [(0.30, "低い"), (0.60, "中程度")]  # これ未満ならそのラベル、超えたら「高い」


def proba_to_label(proba: float) -> str:
    for threshold, label in PROBA_BANDS:
        if proba < threshold:
            return label
    return "高い"


SYSTEM_PROMPT = """あなたは電力系統のデータアナリストです。翌日の日照時間予報と、太陽光の出力制御を
予測する統計モデルの結果（数値のみ）が与えられます。これを読んで、一般の太陽光発電事業者にも
わかる補足説明を書いてください（判定自体はすでに別途明示されているので、ここでは背景・注意点のみ）。

厳守事項:
- 与えられた数値を書き換えない・新しい数値を計算しない（解釈・要約のみ）。
- 日照時間の出典が「簡略化（天気カテゴリから代用）」の場合は、厳密な日射予測ではないことを明記する。
  出典が「農研機構AMGSDS（数値予報）」の場合も、**快晴に近い条件では日照時間を少なめに見積もる
  既知の傾向がある**ため、特に晴れ予報のときは予測確率を低めに見積もっている可能性がある旨を
  一言添える。
- モデルの判別力（ROC-AUC 0.924、高めの判別力）を踏まえつつ、断定的な言い方を避ける。
- 出力制御の有無を保証するものではない旨を明記する（最終判断は電力会社の公式発表による）。
- 出力は日本語のプレーンテキスト。150字程度で簡潔に（判定ラベル・確率の言い直しは不要）。"""


def fetch_tomorrow_forecast() -> dict:
    data = requests.get(FORECAST_URL, timeout=20).json()
    ts_weather = data[0]["timeSeries"][0]
    ts_pop = data[0]["timeSeries"][1]

    area_w = next(a for a in ts_weather["areas"] if a["area"]["name"] == AREA_NAME)
    area_p = next(a for a in ts_pop["areas"] if a["area"]["name"] == AREA_NAME)

    # timeDefines[1] が「明日」（[0]は今日）
    tomorrow_date = pd.Timestamp(ts_weather["timeDefines"][1]).date()
    weather_code = area_w["weatherCodes"][1]
    weather_text = area_w["weathers"][1]

    # pop(降水確率)は3時間毎。明日分（timeDefinesの日付が一致するもの）の平均を使う
    pops_tomorrow = [
        int(p) for t, p in zip(ts_pop["timeDefines"], area_p["pops"])
        if pd.Timestamp(t).date() == tomorrow_date and p
    ]
    pop_avg = sum(pops_tomorrow) / len(pops_tomorrow) if pops_tomorrow else None

    return {
        "date": tomorrow_date,
        "weather_code": weather_code,
        "weather_text": weather_text,
        "pop_avg": pop_avg,
    }


def weather_code_to_sunshine_proxy(weather_code: str, sunshine_quantiles: dict) -> tuple:
    """天気コードの先頭数字（1=晴れ系/2=曇り系/3,4=雨雪系）で、実測「九州7県AMeDAS平均」日照時間の
    分位点に置き換える（AMGSDSメッシュ予報が使えないときのフォールバック）。
    返り値: (推定sunshine_h, カテゴリ名, 使った分位点)
    """
    first_digit = weather_code[0]
    if first_digit == "1":
        return sunshine_quantiles["q75"], "晴れ系", "75%ile"
    elif first_digit == "2":
        return sunshine_quantiles["q50"], "曇り系", "50%ile"
    else:
        return sunshine_quantiles["q10"], "雨/雪系", "10%ile"


def fetch_amgsds_mesh_sunshine(target_date: dt.date) -> float:
    """農研機構AMGSDSから指定日の日照時間(SSD)数値予報を、九州本土メッシュの陸地セル平均で取得する
    （追加検証8: 単一点よりメッシュ面平均の方が精度が高いことを確認済み）。
    失敗時（API障害・トークン失効等）は例外を投げる（呼び出し側でフォールバックする）。
    """
    import AMD_Tools4_ue3 as amd
    today = dt.date.today().strftime("%Y-%m-%d")
    target_str = target_date.strftime("%Y-%m-%d")
    itsu = sorted({today, target_str})
    data, tim, _, _ = amd.GetMetData("SSD", itsu, KYUSHU_BBOX)
    idx = [pd.Timestamp(t).date() for t in tim].index(target_date)
    grid = data[idx]
    land = grid[grid <= SEA_FILL_THRESHOLD]
    return float(np.mean(land))


def train_model(df: pd.DataFrame) -> tuple[LogisticRegression, list, list]:
    """追加検証8で最も精度が高かった構成（月＋日照＋曜日＋トレンド、AUC 0.924）を学習する。"""
    df = df.dropna(subset=["sunshine_h_amgsds_mesh_mean"]).copy()
    df["month"] = df["date"].dt.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    month_cols = list(month_dummies.columns)
    features = month_cols + ["sunshine_h_amgsds_mesh_mean", "is_weekend_or_holiday", "years_since_start"]

    model = LogisticRegression(max_iter=1000)
    model.fit(df[features], df["is_curtailed"])
    return model, features, month_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Claude APIを呼ばず判定結果だけ表示")
    args = parser.parse_args()

    df = build_dataset()
    sunshine_quantiles = {
        "q75": df["sunshine_h_kyushu_mean"].quantile(0.75),
        "q50": df["sunshine_h_kyushu_mean"].quantile(0.50),
        "q10": df["sunshine_h_kyushu_mean"].quantile(0.10),
    }

    forecast = fetch_tomorrow_forecast()

    try:
        sunshine_h = fetch_amgsds_mesh_sunshine(forecast["date"])
        sunshine_source = "農研機構AMGSDS（数値予報、九州本土メッシュ面平均）"
    except Exception as e:
        sunshine_h, category, quantile_used = weather_code_to_sunshine_proxy(
            forecast["weather_code"], sunshine_quantiles)
        sunshine_source = f"簡略化（AMGSDS取得失敗: {e} → 天気カテゴリ「{category}」を実測の九州7県平均{quantile_used}で代用）"

    tomorrow_ts = pd.Timestamp(forecast["date"])
    is_weekend_or_holiday = int(
        tomorrow_ts.dayofweek in (5, 6) or jpholiday.is_holiday(forecast["date"]))
    years_since_start = (tomorrow_ts - df["date"].min()).days / 365.25

    model, features, month_cols = train_model(df)
    x_row = {f"month_{tomorrow_ts.month}": 1} if f"month_{tomorrow_ts.month}" in month_cols else {}
    x_row.update({
        "sunshine_h_amgsds_mesh_mean": sunshine_h,
        "is_weekend_or_holiday": is_weekend_or_holiday,
        "years_since_start": years_since_start,
    })
    x_tomorrow = pd.DataFrame([{c: x_row.get(c, 0) for c in features}])
    proba = float(model.predict_proba(x_tomorrow)[0, 1])
    label = proba_to_label(proba)

    # --- ここが今回追加した「明確な判定」。数値の解釈を人や生成AIに委ねず、先頭に断定表示する ---
    print(f"\n{'=' * 44}")
    print(f"  明日（{forecast['date']}）の出力制御判定: 可能性「{label}」")
    print(f"  予測確率: {proba * 100:.0f}%（判定境界: 低い<30%・中程度<60%・高い≥60%）")
    print(f"{'=' * 44}\n")

    summary = {
        "対象日": str(forecast["date"]),
        "判定": label,
        "予測確率": round(proba, 3),
        "天気予報(気象庁)": forecast["weather_text"],
        "降水確率平均": forecast["pop_avg"],
        "日照時間の予報値(九州面平均)": f"{sunshine_h:.2f}h",
        "日照時間の出典": sunshine_source,
        "土日祝か": bool(is_weekend_or_holiday),
        "モデルの判別力(参考)": "ROC-AUC 0.924（高めの判別力。月+日照(九州メッシュ面平均)+曜日+トレンド）",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    client = anthropic.Anthropic()
    user_text = (
        f"明日（{summary['対象日']}）の福岡地方の天気予報: {summary['天気予報(気象庁)']}\n"
        f"降水確率平均: {summary['降水確率平均']}%\n"
        f"日照時間の予報値(九州本土の面平均): {summary['日照時間の予報値(九州面平均)']}\n"
        f"日照時間の出典: {summary['日照時間の出典']}\n"
        f"土日祝か: {summary['土日祝か']}\n"
        f"統計モデルによる出力制御の予測確率: {summary['予測確率']}（判定: {summary['判定']}）\n"
        f"モデルの判別力: {summary['モデルの判別力(参考)']}\n\n"
        "この情報の背景・注意点について補足説明してください（判定自体の言い直しは不要）。"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": user_text}],
    )
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        print("エラー: text ブロックがありません。stop_reason=", response.stop_reason)
        return 1

    print("\n=== 補足説明 ===")
    print(text)
    cost = response.usage.input_tokens * 2 / 1e6 + response.usage.output_tokens * 10 / 1e6
    print(f"\n概算コスト: ${cost:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"forecast_narrative_{forecast['date']}.txt"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n\n" + text)
    print(f"保存先: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
