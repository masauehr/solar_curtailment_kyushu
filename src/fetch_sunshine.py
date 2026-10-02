"""気象庁「過去の気象データ検索」(etrn) daily_s1.php から日照時間だけを取得する。

cs07_veg_price/fetch_jma_daily.py の p1 ビュー取得パターンを流用（日射量a3は今回不要のため省略）。
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
IDX_SUNSHINE = 16  # daily_s1.php p1ビューの日照時間列（cs07_veg_priceで実測確認済み）

STATIONS = {
    "fukuoka": {"prec_no": 82, "block_no": "47807", "name": "福岡（九州エリアの代表点）"},
    "naha": {"prec_no": 91, "block_no": "47936", "name": "那覇（沖縄エリアの代表点）"},
    # 九州本土7県の県庁所在地（気象官署）。単一点(福岡)より面で見た方が出力制御予測の
    # 精度が上がるかを検証するため追加（READMEの「次にやるなら」#7、ユーザー指示で実施）。
    "saga": {"prec_no": 85, "block_no": "47813", "name": "佐賀"},
    "nagasaki": {"prec_no": 84, "block_no": "47817", "name": "長崎"},
    "kumamoto": {"prec_no": 86, "block_no": "47819", "name": "熊本"},
    "oita": {"prec_no": 83, "block_no": "47815", "name": "大分"},
    "miyazaki": {"prec_no": 87, "block_no": "47830", "name": "宮崎"},
    "kagoshima": {"prec_no": 88, "block_no": "47827", "name": "鹿児島"},
}

KYUSHU_MAINLAND_KEYS = ["fukuoka", "saga", "nagasaki", "kumamoto", "oita", "miyazaki", "kagoshima"]

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "processed"


def fetch_month(prec_no: int, block_no: str, year: int, month: int) -> pd.DataFrame:
    r = requests.get(ETRN_URL, timeout=30, params={
        "prec_no": prec_no, "block_no": block_no,
        "year": year, "month": month, "day": "", "view": "p1"})
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
        raw = vals[IDX_SUNSHINE] if IDX_SUNSHINE < len(vals) else ""
        s = raw.strip().replace(")", "").replace("]", "")
        sunshine_h = 0.0 if s == "--" else (float(s) if s not in ("", "×", "///", "#") else float("nan"))
        rows.append({"date": dt.date(year, month, day), "sunshine_h": sunshine_h})
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


def fetch_kyushu_mainland_mean():
    """九州本土7県（福岡以外の6県は新規取得、福岡は既存キャッシュを再利用）の面平均を作る。
    単一点(福岡)より面で見た方が出力制御予測の精度が上がるかを検証するため（次にやるなら#7）。
    """
    fukuoka_path = OUT_DIR / "sunshine_fukuoka.csv"
    frames = {"fukuoka": pd.read_csv(fukuoka_path, parse_dates=["date"])} if fukuoka_path.exists() \
        else {"fukuoka": fetch_station("fukuoka", 2018, 10, 2026, 3)}

    for key in KYUSHU_MAINLAND_KEYS:
        if key == "fukuoka":
            continue
        out_path = OUT_DIR / f"sunshine_{key}.csv"
        if out_path.exists():
            d = pd.read_csv(out_path, parse_dates=["date"])
        else:
            d = fetch_station(key, 2018, 10, 2026, 3)
            d.to_csv(out_path, index=False)
        print(f"{STATIONS[key]['name']}: {len(d)}日分 → {out_path}")
        frames[key] = d

    merged = frames["fukuoka"][["date"]].copy()
    for key, d in frames.items():
        merged = merged.merge(d.rename(columns={"sunshine_h": f"sunshine_{key}"}), on="date", how="outer")
    value_cols = [f"sunshine_{k}" for k in frames]
    merged["sunshine_h_kyushu_mean"] = merged[value_cols].mean(axis=1, skipna=True)
    merged = merged.sort_values("date")
    merged[["date", "sunshine_h_kyushu_mean"] + value_cols].to_csv(
        OUT_DIR / "sunshine_kyushu_mainland_mean.csv", index=False)
    print(f"九州本土7県平均: {len(merged)}日分 → {OUT_DIR / 'sunshine_kyushu_mainland_mean.csv'}")


def main():
    df = fetch_station("fukuoka", 2018, 10, 2026, 3)
    df.to_csv(OUT_DIR / "sunshine_fukuoka.csv", index=False)
    print(f"福岡: {len(df)}日分 → {OUT_DIR / 'sunshine_fukuoka.csv'}")

    df2 = fetch_station("naha", 2022, 4, 2026, 3)
    df2.to_csv(OUT_DIR / "sunshine_naha.csv", index=False)
    print(f"那覇: {len(df2)}日分 → {OUT_DIR / 'sunshine_naha.csv'}")

    fetch_kyushu_mainland_mean()


if __name__ == "__main__":
    sys.exit(main())
