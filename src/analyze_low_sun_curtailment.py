"""追加検証9: 「全天日射量が小さい領域でも出力制御がある日」の要因を調べる。

ユーザーからの気づき（2026-10-02）: fig08（全天日射量 vs 発電量の散布図）で、日射量が
小さい側にも制御日（赤）がまだ目立つ。考えられる要因として
  (a) 風力発電の増大（晴れていなくても風力が多ければ供給過多になりうる）
  (b) 電力需要の低下（休日等で需要自体が下がれば、少ない発電でも供給過多になりうる）
の2つを検証する。

データ:
- 風力: kyushu_wind_solar_daily.csv（風力+太陽光の合計）- kyushu_solar_only_daily.csv（太陽光単独）
- 需要: kyushu_demand_daily.csv（fetch_kyushu_demand.pyで新規取得）
いずれも2022年7月〜のデータのため、出力制御・全天日射量データとの重なり期間のみで検証する。
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std


def build_dataset() -> pd.DataFrame:
    wind_solar = pd.read_csv(PROCESSED_DIR / "kyushu_wind_solar_daily.csv", parse_dates=["date"])
    wind_solar["wind_solar_mwh"] = wind_solar["wind_solar_kwh"] / 1000
    solar_only = pd.read_csv(PROCESSED_DIR / "kyushu_solar_only_daily.csv", parse_dates=["date"])
    demand = pd.read_csv(PROCESSED_DIR / "kyushu_demand_daily.csv", parse_dates=["date"])
    demand["demand_mwh"] = demand["demand_kwh"] / 1000
    merged = pd.read_csv(PROCESSED_DIR / "kyushu_generation_merged.csv", parse_dates=["date"])[
        ["date", "solar_mj", "is_curtailed"]]

    df = (wind_solar.merge(solar_only, on="date", how="inner")
                     .merge(merged, on="date", how="inner")
                     .merge(demand, on="date", how="inner"))
    df["wind_only_mwh"] = df["wind_solar_mwh"] - df["solar_mwh"]
    df["month"] = df["date"].dt.month
    df["decile"] = pd.qcut(df["solar_mj"], 10, labels=False)
    return df


def main():
    df = build_dataset()
    print(f"重なり期間: {df['date'].min().date()} 〜 {df['date'].max().date()}（{len(df)}日）\n")

    # 全天日射量10分位ごとの制御率（「晴れるほど制御が増える」という基本構造の確認）
    summary = df.groupby("decile").agg(
        n=("is_curtailed", "size"),
        n_curtailed=("is_curtailed", "sum"),
    )
    summary["curtail_pct"] = (summary["n_curtailed"] / summary["n"] * 100).round(1)
    print("=== 全天日射量10分位ごとの制御率 ===")
    print(summary)
    print()

    # 下位20%帯（最も晴れていない帯）に限定して、風力・需要それぞれの効果を見る
    low = df[df["decile"] <= 1]
    low_c = low[low["is_curtailed"] == 1]
    low_nc = low[low["is_curtailed"] == 0]
    print(f"=== 全天日射量下位20%帯（{len(low)}日、制御日{len(low_c)}日） ===")

    for col, label in [("wind_only_mwh", "風力発電量"), ("demand_mwh", "需要")]:
        d = cohens_d(low_c[col], low_nc[col])
        corr = low[col].corr(low["is_curtailed"])
        print(f"{label}: 制御日平均={low_c[col].mean():,.0f} 非制御日平均={low_nc[col].mean():,.0f} "
              f"Cohen's d={d:.2f} 相関r={corr:.3f}")

    print("\n=== 低日射量帯での制御日8日の個別明細 ===")
    detail = low_c.sort_values("date")[["date", "solar_mj", "demand_mwh", "wind_only_mwh", "month"]]
    print(detail.to_string(index=False))

    nc_demand_q = low_nc["demand_mwh"].quantile([0.1, 0.25, 0.5])
    nc_wind_mean = low_nc["wind_only_mwh"].mean()
    print(f"\n(参考) 同帯・非制御日の需要の分位点: {nc_demand_q.to_dict()}")
    print(f"(参考) 同帯・非制御日の風力発電量平均: {nc_wind_mean:,.0f} MWh")


if __name__ == "__main__":
    main()
