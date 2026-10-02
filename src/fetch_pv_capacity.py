"""資源エネルギー庁 FIT/FIPポータル(https://www.fit-portal.go.jp/publicinfosummary)が
四半期ごとに公表する「A表 都道府県別認定・導入量」から、九州本土7県
(福岡・佐賀・長崎・熊本・大分・宮崎・鹿児島、沖縄電力管内は除く)の太陽光発電
導入容量(kW、運転開始済み=FIT/FIP認定のうち実際に稼働している設備)を合計し、
四半期時系列として保存する。

【用途】ロジスティック回帰の「経過年数(years_since_start)」トレンド項を、
実際の導入量(設備容量)に置き換えて予測精度が上がるか検証するため
（README.md「次にやるなら」#4）。

【データ】公開データ(政府統計、認証・ダウンロード不要)。ファイルは四半期末時点
(3/6/9/12月末)ごとに発行され、新しい時点は.xlsx、古い時点(2018年頃以前)は
レガシー形式の.xls で配信される(拡張子はURLに現れないため、内容で判別する)。
シート「②-１ 都道府県別導入容量(新規認定分)」「②-２ (移行認定分)」の
「10kW未満」列＋「10kW以上」列(サブカテゴリの内訳は除く)を合計する。
"""
import html as ihtml
import re
import unicodedata
from pathlib import Path

import openpyxl
import pandas as pd
import requests
import xlrd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "fit_capacity"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = BASE_DIR / "data" / "processed" / "kyushu_pv_capacity_quarterly.csv"

PAGE_URL = "https://www.fit-portal.go.jp/publicinfosummary"
KYUSHU_MAINLAND_PREFS = ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県"]
CAPACITY_SHEETS = ["表A②－１", "表A②－２"]  # 新規認定分 + 移行認定分 = 全体


def discover_a_table_links() -> list[tuple[pd.Timestamp, str]]:
    """公開ページをスクレイピングし、[(四半期末日付, ダウンロードURL), ...] を新しい順に返す。"""
    raw_html = requests.get(PAGE_URL, timeout=30).text
    link_pattern = re.compile(
        r'<a href="(https://www\.fit-portal\.go\.jp/servlet/servlet\.FileDownload\?file=[A-Za-z0-9]+)"'
        r'[^>]*>(.*?)</a>',
        re.S,
    )
    date_pattern = re.compile(r"(\d{4})年(\d{1,2})月末時点")

    results = []
    for url, inner in link_pattern.findall(raw_html):
        text = unicodedata.normalize("NFKC", ihtml.unescape(re.sub(r"<[^>]+>", "", inner)))
        if "都道府県別認定・導入量" not in text:
            continue
        m = date_pattern.search(text)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        date = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
        results.append((date, url))
    return sorted(results, key=lambda x: x[0])


def download_cached(date: pd.Timestamp, url: str) -> Path:
    # 拡張子で判定するライブラリ(openpyxl)があるため、実体(zip=xlsx or OLE=xls)で拡張子を決める
    existing = list(RAW_DIR.glob(f"A_pref_{date.strftime('%Y%m')}.*"))
    if existing:
        return existing[0]
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    ext = "xlsx" if r.content[:2] == b"PK" else "xls"
    path = RAW_DIR / f"A_pref_{date.strftime('%Y%m')}.{ext}"
    path.write_bytes(r.content)
    return path


def _find_toplevel_columns(header_row) -> tuple[int, int]:
    """「10kW未満」「10kW以上」(内訳ではないトップレベルの列)のインデックス(0始まり)を返す。"""
    idx_under, idx_over = None, None
    for i, v in enumerate(header_row):
        if v == "10kW未満":
            idx_under = i
        elif v == "10kW以上":
            idx_over = i
    if idx_under is None or idx_over is None:
        raise ValueError(f"「10kW未満」「10kW以上」列が見つからない: {header_row}")
    return idx_under, idx_over


def sum_kyushu_capacity(path: Path) -> float:
    """1つのA表ファイルから、九州本土7県の太陽光導入容量(kW)の合計を返す。"""
    total = 0.0
    header = path.read_bytes()[:8]
    is_xlsx = header[:2] == b"PK"

    for sheet_name in CAPACITY_SHEETS:
        if is_xlsx:
            wb = openpyxl.load_workbook(path, data_only=True)
            ws = wb[sheet_name]
            rows = [[c.value for c in row] for row in ws.iter_rows()]
        else:
            book = xlrd.open_workbook(path)
            sh = book.sheet_by_name(sheet_name)
            rows = [sh.row_values(r) for r in range(sh.nrows)]

        header_row = rows[3]  # 0始まりで4行目=「10kW未満」「10kW以上」の行
        idx_under, idx_over = _find_toplevel_columns(header_row)

        for row in rows:
            pref = row[0]
            if pref in KYUSHU_MAINLAND_PREFS:
                total += (row[idx_under] or 0) + (row[idx_over] or 0)

    return total


def main():
    links = discover_a_table_links()
    print(f"A表のダウンロードリンク: {len(links)}件見つかった"
          f"（{links[0][0].date()} 〜 {links[-1][0].date()}）")

    rows = []
    for date, url in links:
        path = download_cached(date, url)
        capacity_kw = sum_kyushu_capacity(path)
        rows.append({"date": date, "capacity_kw": capacity_kw})
        print(f"{date.date()}: 九州本土7県 太陽光導入容量 = {capacity_kw:,.0f} kW "
              f"({capacity_kw / 1e6:.2f} 百万kW)")

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(OUT_PATH, index=False)
    print(f"\n保存先: {OUT_PATH}")


if __name__ == "__main__":
    main()
