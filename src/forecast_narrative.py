"""ID-01のGenAI角: 翌日の日照時間予報から、明日の出力制御の可能性を
**明確な判定＋確率**で表示し、補足としてClaudeの平文説明も生成する（次にやるなら②の実施）。

モデルは月（季節ダミー）＋日照時間（九州7県AMeDAS平均）＋曜日（土日祝）＋トレンド（経過年数）、
ROC-AUC 0.925（追加検証8で確認した構成のうち最良のもの）。

【2026-10-06改訂】予報ソースをAMGSDS（農研機構、OAuth認証が必要・稀に障害）から、
気象庁の天気予報（無料・登録不要）だけで完結する構成に変更した。
追加検証8で「AMeDAS九州7県平均」と「AMGSDS九州メッシュ面平均」がほぼ同精度（AUC 0.925 vs 0.924）
だったにもかかわらず、それまでの予測パイプラインはAMGSDSを第一候補にしたままだった
（コスト面で有利な方法が実装に反映されていないとのユーザー指摘を受けて修正）。

パイプライン:
1. 気象庁 bosai の短期予報JSON（福岡・佐賀・長崎・熊本・大分・宮崎・鹿児島の県庁所在地を含む
   予報区、計7地域）から明日の天気カテゴリ（晴れ/曇り/雨雪系）を取得し、7地域の平均カテゴリに
   変換する（多数が晴れなら「晴れ系」等）。これを実測の「九州7県AMeDAS平均」日照時間分布の
   分位点に置き換える（晴れ系=75%ile/曇り系=50%ile/雨雪系=10%ile）。
2. 明日が土日祝かどうかをjpholidayで判定
3. logistic_model.pyと同じ特徴量（月＋日照＋曜日＋トレンド）でロジスティック回帰を学習し、
   明日の出力制御確率を予測
4. 確率を3段階（低い/中程度/高い）の明確な判定ラベルに変換して一番最初に表示する
5. 予報値・出典・曜日・モデル確率（数値のみ）をClaude（テキストのみ）に渡し、
   一般向けの補足説明を生成する（数値の再計算はさせず解釈のみ、ID-35のexplain_report.pyと同じ方針）
"""
import argparse
import json
import sys
from pathlib import Path

import jpholiday
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
FORECAST_URL_TEMPLATE = "https://www.jma.go.jp/bosai/forecast/data/forecast/{code}.json"

# 九州本土7県の県庁所在地を含む気象庁予報区（コード, 予報区名）。
# 「AMeDAS九州7県平均」の構成県と対応させている（src/fetch_sunshine.py参照）。
KYUSHU_FORECAST_AREAS = [
    ("400000", "福岡地方"),      # 福岡市
    ("410000", "南部"),          # 佐賀市
    ("420000", "南部"),          # 長崎市
    ("430000", "熊本地方"),      # 熊本市
    ("440000", "中部"),          # 大分市
    ("450000", "南部平野部"),    # 宮崎市
    ("460100", "薩摩地方"),      # 鹿児島市
]

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
- 日照時間は「気象庁の天気予報(晴れ/曇り/雨)を九州7県で集約し、実測分布の分位点に置き換えた
  簡略値」であり、厳密な日射予測ではないことを明記する。
- モデルの判別力（ROC-AUC 0.925、高めの判別力）を踏まえつつ、断定的な言い方を避ける。
- 出力制御の有無を保証するものではない旨を明記する（最終判断は電力会社の公式発表による）。
- 出力は日本語のプレーンテキスト。150字程度で簡潔に（判定ラベル・確率の言い直しは不要）。"""


def fetch_tomorrow_forecast_one(code: str, area_name: str) -> dict:
    """1つの予報区について、明日の天気コード・降水確率平均を取得する。"""
    data = requests.get(FORECAST_URL_TEMPLATE.format(code=code), timeout=20).json()
    ts_weather = data[0]["timeSeries"][0]
    ts_pop = data[0]["timeSeries"][1]

    area_w = next(a for a in ts_weather["areas"] if a["area"]["name"] == area_name)
    area_p = next(a for a in ts_pop["areas"] if a["area"]["name"] == area_name)

    # timeDefines[1] が「明日」（[0]は今日）
    tomorrow_date = pd.Timestamp(ts_weather["timeDefines"][1]).date()
    weather_code = area_w["weatherCodes"][1]
    weather_text = area_w["weathers"][1]

    pops_tomorrow = [
        int(p) for t, p in zip(ts_pop["timeDefines"], area_p["pops"])
        if pd.Timestamp(t).date() == tomorrow_date and p
    ]
    pop_avg = sum(pops_tomorrow) / len(pops_tomorrow) if pops_tomorrow else None

    return {"date": tomorrow_date, "weather_code": weather_code, "weather_text": weather_text, "pop_avg": pop_avg}


def fetch_tomorrow_forecast_kyushu() -> dict:
    """九州本土7県の予報区すべてから明日の予報を取得し、天気カテゴリを平均（多数決）する。
    晴れ=1/曇り=2/雨雪=3のランクを平均し、四捨五入して代表カテゴリを決める
    （「おおむね晴れだが一部曇り」のような混在を滑らかに反映するため）。
    """
    results = [fetch_tomorrow_forecast_one(code, area_name) for code, area_name in KYUSHU_FORECAST_AREAS]

    def rank(weather_code: str) -> int:
        d = weather_code[0]
        return 1 if d == "1" else (2 if d == "2" else 3)

    ranks = [rank(r["weather_code"]) for r in results]
    avg_rank = round(sum(ranks) / len(ranks))
    category = {1: "晴れ系", 2: "曇り系", 3: "雨/雪系"}[avg_rank]

    pops = [r["pop_avg"] for r in results if r["pop_avg"] is not None]
    pop_avg = sum(pops) / len(pops) if pops else None

    texts = ", ".join(f"{name}:{r['weather_text']}" for (_, name), r in zip(KYUSHU_FORECAST_AREAS, results))

    return {
        "date": results[0]["date"],
        "category": category,
        "pop_avg": pop_avg,
        "per_area_text": texts,
        "n_sunny": sum(1 for r in ranks if r == 1),
        "n_cloudy": sum(1 for r in ranks if r == 2),
        "n_rainy": sum(1 for r in ranks if r == 3),
    }


def category_to_sunshine_proxy(category: str, sunshine_quantiles: dict) -> tuple:
    """天気カテゴリ（九州7県の平均）を、実測「九州7県AMeDAS平均」日照時間分布の分位点に置き換える。
    返り値: (推定sunshine_h, 使った分位点)
    """
    if category == "晴れ系":
        return sunshine_quantiles["q75"], "75%ile"
    elif category == "曇り系":
        return sunshine_quantiles["q50"], "50%ile"
    else:
        return sunshine_quantiles["q10"], "10%ile"


def train_model(df: pd.DataFrame) -> tuple[LogisticRegression, list, list]:
    """追加検証8の構成（月＋日照[九州7県AMeDAS平均]＋曜日＋トレンド、ROC-AUC 0.925）を学習する。"""
    df = df.dropna(subset=["sunshine_h_kyushu_mean"]).copy()
    df["month"] = df["date"].dt.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    month_cols = list(month_dummies.columns)
    features = month_cols + ["sunshine_h_kyushu_mean", "is_weekend_or_holiday", "years_since_start"]

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

    forecast = fetch_tomorrow_forecast_kyushu()
    sunshine_h, quantile_used = category_to_sunshine_proxy(forecast["category"], sunshine_quantiles)
    sunshine_source = (f"気象庁の天気予報（九州7県集約、晴れ{forecast['n_sunny']}/曇り{forecast['n_cloudy']}/"
                        f"雨雪{forecast['n_rainy']}地域 → 代表カテゴリ「{forecast['category']}」）を"
                        f"実測の九州7県AMeDAS平均{quantile_used}で代用")

    tomorrow_ts = pd.Timestamp(forecast["date"])
    is_weekend_or_holiday = int(
        tomorrow_ts.dayofweek in (5, 6) or jpholiday.is_holiday(forecast["date"]))
    years_since_start = (tomorrow_ts - df["date"].min()).days / 365.25

    model, features, month_cols = train_model(df)
    x_row = {f"month_{tomorrow_ts.month}": 1} if f"month_{tomorrow_ts.month}" in month_cols else {}
    x_row.update({
        "sunshine_h_kyushu_mean": sunshine_h,
        "is_weekend_or_holiday": is_weekend_or_holiday,
        "years_since_start": years_since_start,
    })
    x_tomorrow = pd.DataFrame([{c: x_row.get(c, 0) for c in features}])
    proba = float(model.predict_proba(x_tomorrow)[0, 1])
    label = proba_to_label(proba)

    # --- 生の確率だけでなく、明確な判定ラベルを先頭に断定表示する ---
    print(f"\n{'=' * 44}")
    print(f"  明日（{forecast['date']}）の出力制御判定: 可能性「{label}」")
    print(f"  予測確率: {proba * 100:.0f}%（判定境界: 低い<30%・中程度<60%・高い≥60%）")
    print(f"{'=' * 44}\n")

    summary = {
        "対象日": str(forecast["date"]),
        "判定": label,
        "予測確率": round(proba, 3),
        "天気予報(九州7県の内訳)": forecast["per_area_text"],
        "天気予報(代表カテゴリ)": forecast["category"],
        "降水確率平均(九州7県)": round(forecast["pop_avg"], 1) if forecast["pop_avg"] is not None else None,
        "日照時間の予報値(九州7県AMeDAS平均で代用)": f"{sunshine_h:.2f}h",
        "日照時間の出典": sunshine_source,
        "土日祝か": bool(is_weekend_or_holiday),
        "モデルの判別力(参考)": "ROC-AUC 0.925（高めの判別力。月+日照[九州7県AMeDAS平均]+曜日+トレンド）",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    client = anthropic.Anthropic()
    user_text = (
        f"明日（{summary['対象日']}）の九州7県の天気予報: {summary['天気予報(九州7県の内訳)']}\n"
        f"代表カテゴリ: {summary['天気予報(代表カテゴリ)']}\n"
        f"降水確率平均(九州7県): {summary['降水確率平均(九州7県)']}%\n"
        f"日照時間の予報値: {summary['日照時間の予報値(九州7県AMeDAS平均で代用)']}\n"
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
