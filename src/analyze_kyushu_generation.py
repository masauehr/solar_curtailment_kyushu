"""九州電力送配電の実際の発電量と、気象データ（実測の日照時間・全天日射量、および各種予報手法の
シミュレーション値）の相関を検証する。

**太陽光単独**（kyushu_solar_only_daily.csv、2018-10〜2026-09・旧形式+新形式を接続して取得）を主軸にする。
風力は日射と無関係なため、「エリア風力・太陽光発電量」の合計（kyushu_wind_solar_daily.csv、
2022-07〜のみ）を使うと太陽光単独の相関より弱く出てしまう（ユーザーからの指摘で判明、比較結果は
下記に出力する）。

注意: 発電量は「出力制御後」の値。制御が発生した日は「天気が良いのに発電量が増えない/落ちる」
という逆転が起きるため、相関を見る際は制御日と非制御日を分けて確認する。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from logistic_model import build_dataset
from backtest_kyushu_forecast_methods import build_backtest_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def main():
    solar = pd.read_csv(PROCESSED_DIR / "kyushu_solar_only_daily.csv", parse_dates=["date"])
    solar_mj = pd.read_csv(PROCESSED_DIR / "solar_fukuoka.csv", parse_dates=["date"])
    backtest = build_backtest_dataset()  # sunshine_h(実測) / amgsds_proxy_h / simplified_h / is_curtailed

    df = solar.merge(backtest, on="date", how="inner").merge(solar_mj, on="date", how="inner")
    print(f"比較対象（太陽光単独、全期間）: {len(df)}日（{df['date'].min().date()} 〜 {df['date'].max().date()}）\n")

    print("=== 実際の発電量[MWh]との相関（全日） ===")
    for col, name in [("sunshine_h", "実測日照時間"), ("solar_mj", "実測全天日射量"),
                       ("amgsds_proxy_h", "AMGSDS模擬予報(快晴バイアス補正)"), ("simplified_h", "簡略化")]:
        r = df["solar_mwh"].corr(df[col])
        print(f"  {name:28s} r={r:.3f}")

    print("\n=== 非制御日のみでの相関（出力制御による頭打ちを除く） ===")
    non_curtailed = df[df["is_curtailed"] == 0]
    print(f"  (n={len(non_curtailed)})")
    for col, name in [("sunshine_h", "実測日照時間"), ("solar_mj", "実測全天日射量"),
                       ("amgsds_proxy_h", "AMGSDS模擬予報(快晴バイアス補正)"), ("simplified_h", "簡略化")]:
        r = non_curtailed["solar_mwh"].corr(non_curtailed[col])
        print(f"  {name:28s} r={r:.3f}")

    print("\n=== 制御日 vs 非制御日の発電量レベル（全期間、天気が同程度でも制御で頭打ちになるか） ===")
    print("  【注意】全期間(2018-2026)だと太陽光の導入量が年々増えているため、単純比較は")
    print("  「天気」ではなく「導入量の違う年を混ぜて比較」になり、頭打ち効果が隠れる/逆転しうる。")
    clear_days = df[df["weather_class"].isin(["快晴", "晴れ"])]
    for label, sub in [("快晴/晴れ×制御日", clear_days[clear_days["is_curtailed"] == 1]),
                        ("快晴/晴れ×非制御日", clear_days[clear_days["is_curtailed"] == 0])]:
        print(f"  {label}: n={len(sub)}  発電量平均={sub['solar_mwh'].mean():.1f}MWh  "
              f"実測日照時間平均={sub['sunshine_h'].mean():.2f}h")

    print("\n=== 同じ比較（導入量がほぼ一定の直近期間に限定、2023-01以降） ===")
    recent_clear = clear_days[clear_days["date"] >= "2023-01-01"]
    for label, sub in [("快晴/晴れ×制御日(直近)", recent_clear[recent_clear["is_curtailed"] == 1]),
                        ("快晴/晴れ×非制御日(直近)", recent_clear[recent_clear["is_curtailed"] == 0])]:
        print(f"  {label}: n={len(sub)}  発電量平均={sub['solar_mwh'].mean():.1f}MWh  "
              f"実測日照時間平均={sub['sunshine_h'].mean():.2f}h")

    print("\n=== 参考: 風力+太陽光の合計（旧データ、2022-07〜）を使うと相関が弱まる ===")
    wind_solar = pd.read_csv(PROCESSED_DIR / "kyushu_wind_solar_daily.csv", parse_dates=["date"])
    wind_solar["wind_solar_mwh"] = wind_solar["wind_solar_kwh"] / 1000
    df_ws = df.merge(wind_solar[["date", "wind_solar_mwh"]], on="date", how="inner")
    for col, name in [("sunshine_h", "実測日照時間"), ("solar_mj", "実測全天日射量")]:
        r_solar = df_ws["solar_mwh"].corr(df_ws[col])
        r_ws = df_ws["wind_solar_mwh"].corr(df_ws[col])
        print(f"  {name}: 太陽光単独 r={r_solar:.3f}  風力+太陽光合計 r={r_ws:.3f}  (n={len(df_ws)})")

    out_path = PROCESSED_DIR / "kyushu_generation_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    main()
