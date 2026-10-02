"""九州電力送配電「過去の出力制御実績＿九州本土」Excelから、出力制御が実施された日付を抽出する。

【重要・修正履歴】当初は「前日指示」（前日の予報に基づく事前指示）列だけを読んでいたが、
これは「前日は晴れ予報で指示したが、当日は曇って実際には制御不要だった」というケースを誤って
「制御日」に含めてしまうバグだった（ユーザーが発電量の散布図から発見）。出力制御の実績報告書には
前日指示の後に「速報」（当日見直し）という、実際の当日の状況に基づく最終確認が必ずセットで
記載されており、速報の方が新しい・正しい情報。本スクリプトは速報があれば速報を優先する。

シート構成（年度ごとに1シート、レイアウトは年度によって3パターンある。実測で確認）:
- 「通し番号」の行（2018年度=9行目、2019〜2022年度=10行目、2023〜2025年度=8行目）を探し、
  その1行下=発信日、2行下=ラベル（前日指示/速報/当日見直し）、3行下=再エネ出力制御期間、という
  構成は全年度で共通。
- 列は「前日指示」列（発信日の翌日を対象）と「速報」列（発信日と同じ日を対象、前日指示の
  見直し）が交互に並ぶ。シートは発信日の昇順（＝時系列順）に並んでいるため、各列を左から順に
  処理し、対象日ごとに最後に書き込まれた値で上書きすることで、自動的に「最新の決定」が残る
  （速報は前日指示より後の列に出てくるため、上書きで速報が優先される）。
- 日付の値は、新しい年度はdatetime型、古い年度は "10/13（土）" のような文字列（曜日付き）。
  文字列の場合は年度の開始年（4月始まり）から実際の暦年を推定してパースする。
- 「出力制御なし」の場合はその列が対象とする日付を特定し、制御なしとして記録する
  （前日指示列なら発信日の翌日、速報列なら発信日と同じ日）。
"""
import re
import datetime as dt
from pathlib import Path

import openpyxl
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "kyushu_curtail_history.xlsx"
OUT_PATH = BASE_DIR / "data" / "processed" / "kyushu_curtail_days.csv"

DATE_STR_RE = re.compile(r"^(\d{1,2})/(\d{1,2})")


def fiscal_start_year(sheet_name: str) -> int:
    return int(re.match(r"(\d{4})", sheet_name).group(1))


def find_seq_row(ws, max_row: int = 15) -> int:
    for r in range(1, max_row + 1):
        for c in range(1, 6):
            v = ws.cell(row=r, column=c).value
            if isinstance(v, str) and "通し番号" in v:
                return r
    raise ValueError("「通し番号」の行が見つからない")


def parse_date_cell(val, fy_start_year: int):
    """datetime型 or "10/13（土）..." 形式の文字列を日付に変換する（制御期間・発信日の両方に使う）。"""
    if hasattr(val, "date"):
        return val.date()
    if isinstance(val, str):
        m = DATE_STR_RE.match(val.strip())
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            year = fy_start_year if month >= 4 else fy_start_year + 1
            try:
                return pd.Timestamp(year, month, day).date()
            except ValueError:
                return None
    return None


def extract_sheet(ws, fy_start_year: int) -> dict:
    """1つの年度シートから、{日付: 制御の有無(bool)} の辞書を返す。
    列を左から順に処理し、同じ対象日を後から書き込まれた値で上書きすることで、
    「前日指示 → 速報」の順で最新の決定が残るようにする。
    """
    seq_row = find_seq_row(ws)
    issue_row = seq_row + 1
    label_row = seq_row + 2
    period_row = seq_row + 3
    seq_cols = [c for c in range(1, ws.max_column + 1) if isinstance(ws.cell(row=seq_row, column=c).value, int)]
    if not seq_cols:
        return {}
    min_col, max_col = min(seq_cols), max(seq_cols) + 4  # 最後のseqの速報列まで走査範囲に含める

    status: dict = {}
    for c in range(min_col, max_col + 1):
        label = ws.cell(row=label_row, column=c).value
        if not isinstance(label, str):
            continue
        is_advance = "前日指示" in label
        is_sameday = ("速報" in label) or ("当日見直し" in label)
        if not (is_advance or is_sameday):
            continue

        issue_date = parse_date_cell(ws.cell(row=issue_row, column=c).value, fy_start_year)
        period_val = ws.cell(row=period_row, column=c).value
        period_date = parse_date_cell(period_val, fy_start_year)

        if period_date is not None:
            status[period_date] = True
        elif isinstance(period_val, str) and "出力制御なし" in period_val:
            if issue_date is None:
                continue
            target_date = issue_date + dt.timedelta(days=1 if is_advance else 0)
            status[target_date] = False
        # それ以外（空欄等）は対象外

    return status


def main():
    wb = openpyxl.load_workbook(RAW_PATH, data_only=True)
    rows = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        status = extract_sheet(ws, fiscal_start_year(sheet_name))
        curtailed = sorted(d for d, v in status.items() if v)
        cancelled = sum(1 for v in status.values() if not v)
        for d in curtailed:
            rows.append({"fiscal_year_sheet": sheet_name, "date": d})
        print(f"{sheet_name}: 制御日 {len(curtailed)}日（速報等で後から取り消されたもの{cancelled}件を除外）")

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(OUT_PATH, index=False)
    print(f"\n合計 制御日数: {len(df)}日（{df['date'].min()} 〜 {df['date'].max()}）")
    print(f"保存先: {OUT_PATH}")


if __name__ == "__main__":
    main()
