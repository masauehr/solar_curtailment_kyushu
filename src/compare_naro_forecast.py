"""気象庁の天気予報カテゴリ（晴れ/曇り）を実測データの分位点で代用した簡略化（forecast_narrative.py）
と、農研機構メッシュ農業気象データ（AMGSDS）の実際の日照時間（SSD）数値予報を比較する。

AMGSDSは気象庁の数値予報モデル出力を基にした公的機関の派生データであり、CLAUDE.mdで明示的に
例外として使用を許可している（第三者APIだが、気象庁GPVを直接扱わずに済む手段として）。

九州本土を覆う範囲（メッシュ）で予報を取得し、海上セル（欠損値）を除いた陸上メッシュの分布も見る
（農研機構のデータはメッシュ情報なので、単一地点だけでなく面で捉えられるという利点を確認する）。
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/masahiro/projects/common")
import AMD_Tools4_ue3 as amd  # noqa: E402

from logistic_model import build_dataset  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
import jpholiday  # noqa: E402

FUKUOKA_POINT = [33.6064, 33.6064, 130.4181, 130.4181]
KYUSHU_BBOX = [31.0, 34.0, 129.5, 131.9]  # 九州本土をおおむね覆う範囲
SEA_FILL_THRESHOLD = 24.0  # 1日の日照時間は最大でも24h。これを超える値は海上等の欠損値とみなす

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def fetch_tomorrow_ssd_point(point) -> float:
    today = dt.date.today().strftime("%Y-%m-%d")
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    data, tim, _, _ = amd.GetMetData("SSD", [today, tomorrow], point)
    return float(data[1, 0, 0])


def fetch_tomorrow_ssd_area(bbox) -> np.ndarray:
    today = dt.date.today().strftime("%Y-%m-%d")
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    data, tim, lat, lon = amd.GetMetData("SSD", [today, tomorrow], bbox)
    tomorrow_grid = data[1]
    land = tomorrow_grid[tomorrow_grid <= SEA_FILL_THRESHOLD]
    return land


def predict_curtailment_proba(sunshine_h: float, target_date: dt.date, df: pd.DataFrame) -> float:
    features = ["sunshine_h", "is_weekend_or_holiday", "years_since_start"]
    model = LogisticRegression()
    model.fit(df[features], df["is_curtailed"])

    ts = pd.Timestamp(target_date)
    is_weekend_or_holiday = int(ts.dayofweek in (5, 6) or jpholiday.is_holiday(target_date))
    years_since_start = (ts - df["date"].min()).days / 365.25
    x = pd.DataFrame([{
        "sunshine_h": sunshine_h,
        "is_weekend_or_holiday": is_weekend_or_holiday,
        "years_since_start": years_since_start,
    }])[features]
    return float(model.predict_proba(x)[0, 1])


def main():
    tomorrow = dt.date.today() + dt.timedelta(days=1)

    fukuoka_ssd = fetch_tomorrow_ssd_point(FUKUOKA_POINT)
    land_cells = fetch_tomorrow_ssd_area(KYUSHU_BBOX)

    # forecast_narrative.py と同じ「晴れ系→75%ile」の簡略化値（天気予報が「晴れ」の前提）
    df = build_dataset()
    proxy_sunshine = df["sunshine_h"].quantile(0.75)

    print(f"=== 明日（{tomorrow}）の日照時間予報の比較 ===\n")
    print(f"簡略化（JMA天気カテゴリ「晴れ」→実測分布の75%ile）: {proxy_sunshine:.2f}h")
    print(f"農研機構AMGSDS（福岡単一点の数値予報）: {fukuoka_ssd:.2f}h")
    print(f"農研機構AMGSDS（九州本土メッシュ、陸上セル{len(land_cells)}点の分布）:")
    print(f"  min={land_cells.min():.2f}h  median={np.median(land_cells):.2f}h  "
          f"mean={land_cells.mean():.2f}h  max={land_cells.max():.2f}h")

    print("\n=== 出力制御の予測確率への影響 ===")
    for label, val in [
        ("簡略化(75%ile)", proxy_sunshine),
        ("AMGSDS福岡単一点", fukuoka_ssd),
        ("AMGSDS九州メッシュ平均", float(land_cells.mean())),
    ]:
        proba = predict_curtailment_proba(val, tomorrow, df)
        print(f"  {label:24s} 日照時間={val:5.2f}h → 出力制御確率={proba:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "date": str(tomorrow),
        "proxy_sunshine_h": round(float(proxy_sunshine), 2),
        "amgsds_fukuoka_point_h": round(fukuoka_ssd, 2),
        "amgsds_kyushu_mesh_land_cells": int(len(land_cells)),
        "amgsds_kyushu_mesh_min_h": round(float(land_cells.min()), 2),
        "amgsds_kyushu_mesh_median_h": round(float(np.median(land_cells)), 2),
        "amgsds_kyushu_mesh_mean_h": round(float(land_cells.mean()), 2),
        "amgsds_kyushu_mesh_max_h": round(float(land_cells.max()), 2),
    }
    out_path = RESULTS_DIR / f"naro_vs_proxy_{tomorrow}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    main()
