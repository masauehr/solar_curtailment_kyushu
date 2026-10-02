"""九州電力送配電の太陽光発電実績「単独」の日次合計を取得する（風力を含まない）。

**注意: 「太陽光実績」はパネルが作った電力の総量ではなく、系統（電力網）に実際に流れ込んだ
太陽光の出力を指す**（経産省「系統情報の公表の考え方」に基づく需給実績公表、系統運用のための数値。
出力制御でこの値自体が下がる）。「発電量」と呼ぶと発電能力そのものと誤解されるため注意。

これまでの fetch_kyushu_generation.py は「エリア風力・太陽光発電量」の合計しか
取得できなかった（出典: 需給調整業務の実施状況の公表、2022年7月〜）。
風力は日射と無関係なので、太陽光だけとの相関を見るにはこの合計では不十分
（ユーザーからの指摘）。

太陽光を風力と分離した実績は、以下の2つの異なる形式のデータを繋ぎ合わせて作る:
1. 旧形式「エリア需給実績データ」（2016年度〜2023年度、四半期CSV、時別値 MWh）
   出典: https://www.kyuden.co.jp/td_area_jukyu/jukyu.html
   列: ...,太陽光実績[MWh],太陽光抑制量[MWh],風力発電実績[MWh],風力抑制量[MWh],...
2. 新形式「燃料種別発電実績」（2024年初頭〜現在、日次CSV、30分値 万kW平均）
   出典: https://www.kyuden.co.jp/td_area_jukyu/csv/eria_kyokyujisseki_<YYYYMMDD>.csv
   （ページ埋め込みのjukyu.jsから実際のファイル名パターンを特定）
   列: ...,風力,太陽光,...  単位は「万kW平均」(30分の平均出力)なので
   MWh = 値 × 10(万kW→MW) × 0.5(30分=0.5h) = 値 × 5
"""
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "kyushu_solar_only_daily.csv"

OLD_QUARTERS = (
    ["H30_3Q", "H30_4Q"]
    + [f"{y}_{q}Q" for y in range(2019, 2024) for q in range(1, 5)]
)
OLD_URL = "https://www.kyuden.co.jp/td_area_jukyu/csv_area_jyukyu_jisseki/area_jyukyu_jisseki_{q}.csv"
NEW_URL = "https://www.kyuden.co.jp/td_area_jukyu/csv/eria_kyokyujisseki_{date}.csv"
NEW_START = dt.date(2024, 1, 1)


def fetch_old_quarter(q: str) -> pd.DataFrame:
    r = requests.get(OLD_URL.format(q=q), timeout=30)
    r.raise_for_status()
    r.encoding = "shift_jis"
    lines = r.text.splitlines()
    rows = []
    for line in lines[2:]:
        cols = line.split(",")
        if len(cols) < 8:
            continue
        try:
            dt_str = cols[0].strip('"')
            date = pd.Timestamp(dt_str).date()
            solar_mwh = float(cols[7])  # 太陽光実績[MWh]（列順は実測で確認済み）
        except (ValueError, IndexError):
            continue
        rows.append({"date": date, "solar_mwh": solar_mwh})
    df = pd.DataFrame(rows)
    return df.groupby("date", as_index=False)["solar_mwh"].sum()


def fetch_new_day(date: dt.date, max_retry: int = 3) -> float | None:
    url = NEW_URL.format(date=date.strftime("%Y%m%d"))
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
    total_mankw_half = 0.0
    n = 0
    for line in lines[2:]:
        cols = [c.strip('"') for c in line.split(",")]
        if len(cols) < 10:
            continue
        try:
            total_mankw_half += float(cols[11])  # 太陽光列（実測で確認済み: DATE,TIME,原子力,地熱,水力,LNG,石炭,石油,その他,バイオマス,風力,太陽光,...）
            n += 1
        except ValueError:
            continue
    if n == 0:
        return None
    return total_mankw_half * 5  # 万kW平均(30分) -> MWh


def main():
    print("=== 旧形式（H30_3Q 〜 2023_4Q）取得中 ===")
    old_frames = []
    for q in OLD_QUARTERS:
        print(f"  {q} ...", flush=True)
        old_frames.append(fetch_old_quarter(q))
        time.sleep(0.5)
    old_df = pd.concat(old_frames, ignore_index=True).sort_values("date")

    print("\n=== 新形式（2024-01-01 〜 前日）取得中 ===")
    end = dt.date.today() - dt.timedelta(days=1)
    new_rows = []
    d = NEW_START
    n_ok, n_missing = 0, 0
    while d <= end:
        val = fetch_new_day(d)
        if val is not None:
            new_rows.append({"date": d, "solar_mwh": val})
            n_ok += 1
        else:
            n_missing += 1
        time.sleep(0.3)
        if d.day == 1:
            print(f"  {d} まで取得中... (成功{n_ok}日 欠測{n_missing}日)", flush=True)
        d += dt.timedelta(days=1)
    new_df = pd.DataFrame(new_rows)

    # 新形式の方が実測に近い(直接の太陽光列)ため、重複日は新形式を優先する
    combined = pd.concat([old_df[old_df["date"] < NEW_START], new_df], ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset="date", keep="last")
    combined.to_csv(OUT_PATH, index=False)
    print(f"\n完了: {len(combined)}日分（{combined['date'].min()} 〜 {combined['date'].max()}）")
    print(f"保存先: {OUT_PATH}")


if __name__ == "__main__":
    main()
