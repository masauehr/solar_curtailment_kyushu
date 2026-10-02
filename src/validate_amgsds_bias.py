"""【訂正あり・README.md「追加検証3」参照】ユーザーの指摘（AMGSDSの日照時間は快晴時に過少評価
する傾向があるため、気象庁の天気予報より精度が良いかは分からない）を検証しようとしたスクリプト。

**このスクリプトの検証方法は誤りだった**: AMGSDSのAPIは「過去日付」を指定すると予報ではなく
実況（解析値）を返すだけで、過去のある日に「1日先予報としてどう予測していたか」のアーカイブは
取得できない（API自体に予報のvintage＝発表日ごとの履歴を遡る機能が無い）。そのため以下で得られる
「相関0.999・ほぼバイアス無し」という結果は、解析値どうし（ほぼ同じ観測データ由来）を比較した
当然の帰結であり、**予報精度について何も語っていない**。

実際の予報精度（lead=0,1日の予報 vs 確定実況）は `ml_forecast` プロジェクトの
`WEATHER_ACCURACY_VALIDATION.md` で既に検証されており、**快晴日にSSD予報が-5.04h(lead=0)/
-5.30h(lead=1)系統的に過少評価する**ことが分かっている（ユーザーの指摘が正しかった）。
このスクリプトは「検証方法を誤った実例」として削除せず残す。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/masahiro/projects/common")
import AMD_Tools4_ue3 as amd  # noqa: E402

FUKUOKA_POINT = [33.6064, 33.6064, 130.4181, 130.4181]

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def fetch_amgsds_history(start: str, end: str) -> pd.DataFrame:
    data, tim, _, _ = amd.GetMetData("SSD", [start, end], FUKUOKA_POINT)
    return pd.DataFrame({
        "date": [pd.Timestamp(t).normalize() for t in tim],
        "ssd_amgsds": data[:, 0, 0],
    })


def main():
    etrn = pd.read_csv(PROCESSED_DIR / "sunshine_fukuoka.csv", parse_dates=["date"])
    start = etrn["date"].min().strftime("%Y-%m-%d")
    end = etrn["date"].max().strftime("%Y-%m-%d")

    amgsds = fetch_amgsds_history(start, end)
    df = etrn.merge(amgsds, on="date", how="inner")
    df["diff"] = df["ssd_amgsds"] - df["sunshine_h"]  # 負なら過小評価

    print(f"比較対象: {len(df)}日（{df['date'].min().date()} 〜 {df['date'].max().date()}）\n")

    print("=== 全日の一致度 ===")
    corr = df["sunshine_h"].corr(df["ssd_amgsds"])
    mae = df["diff"].abs().mean()
    bias = df["diff"].mean()
    print(f"相関係数: {corr:.3f}  MAE: {mae:.2f}h  平均バイアス(AMGSDS-実測): {bias:+.2f}h\n")

    curtail = pd.read_csv(PROCESSED_DIR / "kyushu_curtail_days.csv", parse_dates=["date"])
    df["is_curtailed"] = df["date"].isin(curtail["date"])

    print("=== 実測の日照時間レベル別のバイアス（出力制御は晴天日=高日照時間日に集中） ===")
    bins = [-0.01, 2, 5, 8, 15]
    labels = ["0-2h(曇天/雨)", "2-5h(薄曇り)", "5-8h(晴れ間)", "8h+(快晴)"]
    df["band"] = pd.cut(df["sunshine_h"], bins=bins, labels=labels)
    summary = df.groupby("band").agg(
        n=("diff", "size"),
        実測平均=("sunshine_h", "mean"),
        AMGSDS平均=("ssd_amgsds", "mean"),
        バイアス平均=("diff", "mean"),
        制御日割合=("is_curtailed", "mean"),
    )
    print(summary.to_string())

    print("\n=== 出力制御日 vs 非制御日でのAMGSDS解析値の一致度 ===")
    for label, sub in [("出力制御日", df[df["is_curtailed"]]), ("非制御日", df[~df["is_curtailed"]])]:
        print(f"{label}: n={len(sub)}  実測平均={sub['sunshine_h'].mean():.2f}h  "
              f"AMGSDS平均={sub['ssd_amgsds'].mean():.2f}h  バイアス={sub['diff'].mean():+.2f}h")

    print("\n=== 快晴帯(実測8h以上)のうち、出力制御日とそれ以外でのバイアス比較 ===")
    clear = df[df["sunshine_h"] >= 8]
    for label, sub in [("快晴×出力制御日", clear[clear["is_curtailed"]]),
                        ("快晴×非制御日", clear[~clear["is_curtailed"]])]:
        if len(sub) == 0:
            print(f"{label}: データなし")
            continue
        print(f"{label}: n={len(sub)}  実測平均={sub['sunshine_h'].mean():.2f}h  "
              f"AMGSDS平均={sub['ssd_amgsds'].mean():.2f}h  バイアス={sub['diff'].mean():+.2f}h")


if __name__ == "__main__":
    main()
