# CLAUDE.md — solar_curtailment_kyushu

## このプロジェクトについて
太陽光の出力制御（カーテイルメント）が天気・曜日・季節でどこまで説明できるかを、
九州電力送配電・沖縄電力の実際の出力制御実績データで検証するプロジェクト。
[weather_hackathon_ideas](https://github.com/masauehr/weather_hackathon_ideas)のID-01スパイクから
2026-10-02に独立。

## データ利用ルール
- 気象データは気象庁HPから取得する（過去データ = [obsdl](https://www.data.jma.go.jp/risk/obsdl/index.php)・
  [etrn](https://www.data.jma.go.jp/stats/etrn/)、実況・予報 = 気象庁 bosai JSON）。
- **例外: 農研機構メッシュ農業気象データ（AMGSDS）は使用可。**
  気象庁の数値予報モデル出力を基にした公的機関（農研機構）の派生データであり、
  気象庁GPV（GRIB）を直接扱わずに日照時間・全天日射量の予報値・実況（解析値）をメッシュで
  得る手段として使用する（予報用途・過去データの回帰分析用途の両方で使用可）。
  OAuth認証が必要な専用API（`/Users/masahiro/projects/common/AMD_Tools4_ue3.py`、
  `nouken`/`ml_forecast`プロジェクトと共用）。
- 資源エネルギー庁 [FIT/FIPポータル](https://www.fit-portal.go.jp/publicinfosummary)（太陽光導入容量、
  都道府県別・四半期）も公開データとして使用する。
- 個人データ・非公開データは使わない。使ってよいのは誰でも同じ手順でアクセスできる公開データのみ
  （気象庁、電力会社の公開実績、政府統計等）。

## 作業ルール
- ドキュメントは日本語。
- 検証コードは `src/` に置く。データは `data/`（git除外、ただし`notebook.ipynb`が依存する
  一部CSVは例外的にコミット済み）。
- Python は `met_env` 相当の環境を想定。`requirements.txt` 参照。

## GitHub更新ルール
- ファイルを変更・追加した場合は、必ず `git add` → `git commit` → `git push` までを行うこと。
- push前にユーザーの確認を求める（破壊的操作のため）。
