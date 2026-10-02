"""九州電力送配電の「需給調整業務の実施状況の公表」日次CSVから、エリア総需要量を取得する
（fetch_kyushu_generation.pyと同じCSV、列4=エリア総需要量〔kWh〕を使う）。

追加検証9: 「全天日射量が小さい領域でも出力制御がある日」が、風力発電の増大だけでなく
電力需要の低下（休日等）でも説明できるかを検証するために追加（ユーザー指摘、2026-10-02）。

出典: https://www.kyuden.co.jp/td_power_usages/download_jukyu.html
      （日次CSV、30分値。列: 日付,時間コマ,時間帯＿自,時間帯＿至,エリア総需要量,エリア総発電量,エリア風力・太陽光発電量）
"""
import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.kyuden.co.jp/td_power_usages/csv/kouhyo/imbalance/21110_TSO9_0_{date}.csv"
REQUEST_INTERVAL_SEC = 0.3

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "kyushu_demand_daily.csv"


def fetch_day(date: dt.date, max_retry: int = 3) -> float | None:
    """指定日の30分値を合計し、1日のkWh合計（エリア総需要量）を返す。取得できない日はNone。"""
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
        if len(cols) < 5:
            continue
        try:
            total += float(cols[4])  # エリア総需要量
            n += 1
        except ValueError:
            continue
    return total if n else None


def main():
    start = dt.date(2022, 7, 1)  # fetch_kyushu_generation.pyと同じ開始日（このCSV形式の実測確認範囲）
    end = dt.date.today() - dt.timedelta(days=1)

    rows = []
    d = start
    while d <= end:
        kwh = fetch_day(d)
        if kwh is not None:
            rows.append({"date": d, "demand_kwh": kwh})
        time.sleep(REQUEST_INTERVAL_SEC)
        d += dt.timedelta(days=1)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)}日分 → {OUT_PATH}")


if __name__ == "__main__":
    main()
