"""沖縄電力「過去の出力制御指示内容」PDF（沖縄本島、年度別）から、出力制御が実施された日付を抽出する。

【重要・修正履歴】当初は「前日指示」列だけを読んでいたが、これは「前日は晴れ予報で指示したが、
当日は曇って実際には制御不要だった」というケースを誤って「制御日」に含めてしまうバグだった
（九州側で発電量の散布図から発見、同じ報告書形式の沖縄にも同じ修正を適用）。

九州電力送配電のExcelと同じ形式の報告書（PDFのテーブル抽出、pdfplumber）:
1ページに4件程度の「通し番号」ブロックがあり、
- 行0: 通し番号
- 行1: 発信日＋ラベル（"3月25日(水) 17時頃\n（前日指示）" のように日付とラベルが同じセルに入っている）
- 行2: 再エネ出力制御期間（"M月D日(曜)\n時刻" の文字列。「出力制御なし」なら制御なし）
列は「前日指示」列（発信日の翌日を対象）と「速報」列（発信日と同じ日を対象、前日指示の見直し）が
交互に並ぶ。表は発信日の昇順で並んでいるため、列を左から順に処理し対象日ごとに上書きすることで
「最新の決定」が残る（九州のExcelパーサーと同じロジック）。
"""
import datetime as dt
import re
from pathlib import Path

import pdfplumber
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_PATH = BASE_DIR / "data" / "processed" / "okinawa_curtail_days.csv"

DATE_STR_RE = re.compile(r"(\d{1,2})月(\d{1,2})日")


def parse_leading_date(text, fy_start_year: int):
    if not text:
        return None
    m = DATE_STR_RE.search(text)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    year = fy_start_year if month >= 4 else fy_start_year + 1
    try:
        return pd.Timestamp(year, month, day).date()
    except ValueError:
        return None


def extract_pdf(path: Path, fy_start_year: int) -> dict:
    """{日付: 制御の有無(bool)} の辞書を返す（速報があれば速報を優先）。"""
    status: dict = {}
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                seq_row_idx = next((i for i, row in enumerate(table)
                                     if row and row[0] == "通し番号"), None)
                if seq_row_idx is None:
                    continue
                issue_label_row = table[seq_row_idx + 1]
                period_row = table[seq_row_idx + 2]
                for c, cell in enumerate(issue_label_row):
                    if not cell or "発" in cell[:3]:  # 先頭列（ラベル自身）はスキップ
                        continue
                    is_advance = "前日指示" in cell
                    is_sameday = "速報" in cell
                    if not (is_advance or is_sameday):
                        continue
                    issue_date = parse_leading_date(cell, fy_start_year)
                    if issue_date is None or c >= len(period_row):
                        continue
                    period_val = period_row[c]
                    period_date = parse_leading_date(period_val, fy_start_year)
                    if period_date is not None:
                        status[period_date] = True
                    elif period_val and "出力制御なし" in period_val:
                        target_date = issue_date + dt.timedelta(days=1 if is_advance else 0)
                        status[target_date] = False
    return status


def main():
    rows = []
    for path in sorted(RAW_DIR.glob("okinawa_previous_control_*.pdf")):
        fy = int(path.stem.rsplit("_", 1)[-1])
        status = extract_pdf(path, fy)
        curtailed = sorted(d for d, v in status.items() if v)
        cancelled = sum(1 for v in status.values() if not v)
        for d in curtailed:
            rows.append({"fiscal_year": fy, "date": d})
        print(f"{fy}年度: 制御日 {len(curtailed)}日（速報等で後から取り消されたもの{cancelled}件を除外）")

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(OUT_PATH, index=False)
    print(f"\n合計 制御日数: {len(df)}日（{df['date'].min()} 〜 {df['date'].max()}）")
    print(f"保存先: {OUT_PATH}")


if __name__ == "__main__":
    main()
