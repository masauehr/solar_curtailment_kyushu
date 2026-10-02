"""気象庁「過去の気象データ検索」(etrn) daily_s1.php から全天日射量(MJ/m^2)を取得する。

日照時間（sunshine_h、fetch_sunshine.py）は「直射光が当たった時間数」で曇りの薄明るさを
拾えないのに対し、全天日射量は発電量に直結する実測量（曇天でも弱い発電はする＝日射量は0にならない）
なので、太陽光発電との相関はこちらの方が本来強いはず、という指摘を受けて追加。
cs07_veg_price/fetch_jma_daily.py の a3 ビュー取得パターンを流用。
出典: 気象庁ホームページ https://www.data.jma.go.jp/stats/etrn/
"""
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ETRN_URL = "https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php"
REQUEST_INTERVAL_SEC = 1.0
IDX_SOLAR = 10  # daily_s1.php a3ビューの全天日射量列（cs07_veg_priceで実測確認済み）

STATIONS = {
    "fukuoka": {"prec_no": 82, "block_no": "47807", "name": "福岡（九州エリアの代表点）"},
    "naha": {"prec_no": 91, "block_no": "47936", "name": "那覇（沖縄エリアの代表点）"},
}

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "processed"


def fetch_month(prec_no: int, block_no: str, year: int, month: int) -> pd.DataFrame:
    r = requests.get(ETRN_URL, timeout=30, params={
        "prec_no": prec_no, "block_no": block_no,
        "year": year, "month": month, "day": "", "view": "a3"})
    r.raise_for_status()
    r.encoding = "shift_jis"
    table = BeautifulSoup(r.text, "html.parser").find("table", id="tablefix1")
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells or not cells[0].get_text(strip=True).isdigit():
            continue
        day = int(cells[0].get_text(strip=True))
        vals = [c.get_text(strip=True) for c in cells]
        raw = vals[IDX_SOLAR] if IDX_SOLAR < len(vals) else ""
        s = raw.strip().replace(")", "").replace("]", "")
        solar_mj = float("nan") if s in ("", "--", "×", "///", "#") else float(s)
        rows.append({"date": dt.date(year, month, day), "solar_mj": solar_mj})
    return pd.DataFrame(rows)


def fetch_station(key: str, year_start: int, month_start: int, year_end: int, month_end: int) -> pd.DataFrame:
    st = STATIONS[key]
    frames, today = [], dt.date.today()
    y, m = year_start, month_start
    while (y, m) <= (year_end, month_end):
        if dt.date(y, m, 1) > today:
            break
        print(f"  {st['name']} {y}-{m:02d} ...", flush=True)
        frames.append(fetch_month(st["prec_no"], st["block_no"], y, m))
        time.sleep(REQUEST_INTERVAL_SEC)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def main():
    df = fetch_station("fukuoka", 2018, 10, 2026, 3)
    df.to_csv(OUT_DIR / "solar_fukuoka.csv", index=False)
    print(f"福岡: {len(df)}日分 → {OUT_DIR / 'solar_fukuoka.csv'}")

    df2 = fetch_station("naha", 2022, 4, 2026, 3)
    df2.to_csv(OUT_DIR / "solar_naha.csv", index=False)
    print(f"那覇: {len(df2)}日分 → {OUT_DIR / 'solar_naha.csv'}")


if __name__ == "__main__":
    sys.exit(main())
