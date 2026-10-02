"""農研機構メッシュ農業気象データ（AMGSDS）から、九州本土全域メッシュの日照時間(SSD)の
日別面平均を取得する。単一点(福岡)より面で見た方が出力制御予測の精度が上がるかを検証するため
（README「次にやるなら」#7、ユーザー指示で実施。CLAUDE.mdの例外条項を2026-10-02に
「過去データの回帰・相関分析にも使用可」へ拡張した上で実施）。

AMGSDSは過去日付を問い合わせると実況（解析値）を返す（予報ではない。追加検証3で確認済み）ため、
ここでの「過去の面平均」は実況値の面平均であり、予報精度の検証ではないことに注意。

1回のAPI呼び出しで複数日・九州全域メッシュ(361×193セル)を取得できる（実測: 1年分で約13秒）ため、
年単位でチャンクして取得し、陸地セル(値が24h以下。海上は欠損値で24h超のfill値が入る)の
日別平均だけを残してすぐ破棄する(全期間を一度にメモリに置くと大きくなるため)。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/masahiro/projects/common")
import AMD_Tools4_ue3 as amd  # noqa: E402

KYUSHU_BBOX = [31.0, 34.0, 129.5, 131.9]  # 九州本土をおおむね覆う範囲
SEA_FILL_THRESHOLD = 24.0  # 1日の日照時間は最大でも24hなので、これを超える値は海上等の欠損値

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "sunshine_amgsds_kyushu_mesh_mean.csv"

PERIOD_START = "2018-10-01"
PERIOD_END = "2026-03-31"


def fetch_year_chunk(start: str, end: str) -> pd.DataFrame:
    data, tim, lat, lon = amd.GetMetData("SSD", [start, end], KYUSHU_BBOX)
    rows = []
    for i, t in enumerate(tim):
        day = data[i]
        land = day[day <= SEA_FILL_THRESHOLD]
        rows.append({
            "date": pd.Timestamp(t).normalize(),
            "sunshine_h_amgsds_mesh_mean": float(np.mean(land)) if land.size else np.nan,
            "n_land_cells": int(land.size),
        })
    return pd.DataFrame(rows)


def main():
    years = pd.period_range(PERIOD_START, PERIOD_END, freq="Y")
    frames = []
    for p in years:
        start = max(pd.Timestamp(PERIOD_START), p.start_time).strftime("%Y-%m-%d")
        end = min(pd.Timestamp(PERIOD_END), p.end_time).strftime("%Y-%m-%d")
        print(f"{start} 〜 {end} を取得中...", flush=True)
        t0 = time.time()
        frames.append(fetch_year_chunk(start, end))
        print(f"  完了（{time.time() - t0:.1f}秒）")

    df = pd.concat(frames, ignore_index=True).sort_values("date")
    df.to_csv(OUT_PATH, index=False)
    print(f"\n{len(df)}日分 → {OUT_PATH}")
    print(f"陸地セル数の範囲: {df['n_land_cells'].min()} 〜 {df['n_land_cells'].max()}"
          f"（九州全域メッシュ361×193=69,673セル中）")


if __name__ == "__main__":
    main()
