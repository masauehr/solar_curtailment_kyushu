"""九州電力送配電の「需給調整業務の実施状況の公表」日次CSVから、エリアの風力・太陽光発電量を取得する。

出典: https://www.kyuden.co.jp/td_power_usages/download_jukyu.html
      （日次CSV、30分値、`エリア風力・太陽光発電量`列。実測で2022年7月頃〜現在まで利用可能を確認）
注意: この列は**風力＋太陽光の合計**であり、太陽光だけの値ではない（九州エリアは風力の比率が
小さいため太陽光が大半を占めると推測されるが、厳密には混在している点に注意）。
"""
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.kyuden.co.jp/td_power_usages/csv/kouhyo/imbalance/21110_TSO9_0_{date}.csv"
REQUEST_INTERVAL_SEC = 0.3

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "kyushu_wind_solar_daily.csv"


def fetch_day(date: dt.date, max_retry: int = 3) -> float | None:
    """指定日の30分値を合計し、1日のkWh合計（風力+太陽光）を返す。取得できない日はNone。
    一時的なタイムアウト等は数回リトライする。"""
    url = BASE_URL.format(date=date.strftime("%Y%m%d"))
    for attempt in range(max_retry):
        try:
            r = requests.get(url, timeout=20)
            break
        except requests.exceptions.RequestException:
            if attempt + 1 >= max_retry:
                return None
            time.sleep(3)
    if r.status_code != 200:
        return None
    r.encoding = "shift_jis"
    lines = r.text.splitlines()
    total = 0.0
    n = 0
    for line in lines[3:]:  # 先頭3行はヘッダ（更新情報2行＋列名1行）
        cols = line.split(",")
        if len(cols) < 7:
            continue
        try:
            total += float(cols[6])
            n += 1
        except ValueError:
            continue
    if n == 0:
        return None
    return total


def main():
    start = dt.date(2022, 7, 1)
    end = dt.date.today() - dt.timedelta(days=1)

    rows = []
    done_dates = set()
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, parse_dates=["date"])
        rows = existing.to_dict("records")
        for r in rows:
            r["date"] = r["date"].date()
        done_dates = {r["date"] for r in rows}
        print(f"既存データ{len(done_dates)}日分を引き継ぎ、続きから取得します")

    d = start
    n_ok, n_missing = 0, 0
    while d <= end:
        if d not in done_dates:
            val = fetch_day(d)
            if val is not None:
                rows.append({"date": d, "wind_solar_kwh": val})
                n_ok += 1
            else:
                n_missing += 1
            time.sleep(REQUEST_INTERVAL_SEC)
            if d.day == 1:
                pd.DataFrame(rows).to_csv(OUT_PATH, index=False)  # 途中経過を都度保存（中断耐性）
                print(f"  {d} まで取得中... (今回成功{n_ok}日 欠測{n_missing}日)", flush=True)
        d += dt.timedelta(days=1)

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(OUT_PATH, index=False)
    print(f"\n完了: {len(df)}日分（{df['date'].min()} 〜 {df['date'].max()}）, 今回の新規取得{n_ok}日 欠測{n_missing}日")
    print(f"保存先: {OUT_PATH}")


if __name__ == "__main__":
    main()
