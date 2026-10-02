"""ID-01スパイクの全検証結果を、スライド化を見据えた図にまとめる（REPORT.md用）。
results/figs/ にPNGを出力する。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from logistic_model import build_dataset
from backtest_kyushu_forecast_methods import build_backtest_dataset, evaluate as bt_evaluate
import logistic_model as lm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "results" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "Hiragino Sans"


def fig00_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, text, color="#dbeafe", fontsize=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                            linewidth=1.4, edgecolor="#334155", facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                      linewidth=1.6, color="#334155"))

    box(0.3, 4.6, 2.6, 1.3, "九州・沖縄の\n出力制御実績\n(Excel/PDF)", color="#fef3c7")
    box(0.3, 2.6, 2.6, 1.3, "気象データ\n(日照時間/全天日射量\n実測・AMGSDS予報)", color="#fef3c7")
    arrow(2.9, 5.25, 3.8, 4.4)
    arrow(2.9, 3.25, 3.8, 4.0)
    box(3.8, 3.6, 2.6, 1.3, "相関・判別力検証\n(Cohen's d,\n九州vs沖縄比較)", color="#dbeafe")
    arrow(6.4, 4.25, 7.3, 4.25)
    box(7.3, 3.6, 2.6, 1.3, "ロジスティック回帰\n(日照+曜日+トレンド)\nAUC 0.669", color="#dbeafe")
    arrow(9.9, 4.25, 10.8, 4.25)
    box(10.8, 3.6, 2.0, 1.3, "Claudeによる\n平文説明\n生成", color="#e9d5ff")

    arrow(8.6, 3.6, 8.6, 2.3)
    box(5.8, 0.9, 5.6, 1.3, "AMGSDS vs 簡略化の比較\n→「AMGSDSが正確」は誤りと訂正\n（快晴日の過少評価バイアス発覚）", color="#fee2e2", fontsize=9.5)

    fig.suptitle("ID-01 検証パイプライン全体図", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "00_pipeline.png", dpi=140)
    plt.close(fig)


def fig01_scatter_sunshine_curtail():
    kyu_c = pd.read_csv(PROCESSED_DIR / "kyushu_curtail_days.csv", parse_dates=["date"])
    oki_c = pd.read_csv(PROCESSED_DIR / "okinawa_curtail_days.csv", parse_dates=["date"])
    fuk = pd.read_csv(PROCESSED_DIR / "sunshine_fukuoka.csv", parse_dates=["date"])
    naha = pd.read_csv(PROCESSED_DIR / "sunshine_naha.csv", parse_dates=["date"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (name, sun, curtail) in zip(axes, [("九州(福岡)", fuk, kyu_c), ("沖縄(那覇)", naha, oki_c)]):
        sun = sun.copy()
        sun["is_curtailed"] = sun["date"].isin(curtail["date"])
        for label, sub, color in [("非制御日", sun[~sun["is_curtailed"]], "#94a3b8"),
                                   ("制御日", sun[sun["is_curtailed"]], "#dc2626")]:
            jitter = rng.uniform(-0.15, 0.15, len(sub))
            x = (sub["is_curtailed"].astype(int) + jitter)
            ax.scatter(x, sub["sunshine_h"], s=8, alpha=0.35, color=color, label=label)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["非制御日", "制御日"])
        ax.set_title(name)
        ax.set_ylabel("日照時間 (h)")
    fig.suptitle("① 出力制御日は日照時間が長い側に偏る（散布図）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_scatter_sunshine.png", dpi=140)
    plt.close(fig)


def fig02_yearly_trend():
    kyu = pd.read_csv(PROCESSED_DIR / "kyushu_curtail_days.csv", parse_dates=["date"])
    # 気象庁側ではなく「fiscal_year_sheet」列があればそれを使う
    if "fiscal_year_sheet" in kyu.columns:
        counts = kyu["fiscal_year_sheet"].str.extract(r"(\d{4})")[0].astype(int).value_counts().sort_index()
    else:
        counts = kyu["date"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#2563eb")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center", fontsize=10)
    ax.set_xlabel("年度")
    ax.set_ylabel("出力制御の実施日数")
    ax.set_title("② 九州の出力制御日数は年々増加（2018〜2025年度）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_yearly_trend.png", dpi=140)
    plt.close(fig)


def fig03_weekday_ratio():
    kyu_c = pd.read_csv(PROCESSED_DIR / "kyushu_curtail_days.csv", parse_dates=["date"])
    oki_c = pd.read_csv(PROCESSED_DIR / "okinawa_curtail_days.csv", parse_dates=["date"])

    days_ja = ["月", "火", "水", "木", "金", "土", "日"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(7)
    for i, (name, df, color) in enumerate([("九州", kyu_c, "#2563eb"), ("沖縄", oki_c, "#dc2626")]):
        counts = df["date"].dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
        ratio = counts / counts.sum() * 100
        ax.bar(x + (i - 0.5) * width, ratio.values, width, label=name, color=color, alpha=0.85)
    ax.axhline(100 / 7, color="gray", ls=":", lw=1.5, label="カレンダー上の均等割合(14.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(days_ja)
    ax.set_ylabel("制御日に占める割合 (%)")
    ax.set_title("③ 出力制御は土日に多い（特に日曜）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_weekday_ratio.png", dpi=140)
    plt.close(fig)


def fig04_discriminative_power():
    labels = ["九州\n全日", "九州\n日曜のみ", "沖縄\n全日", "沖縄\n日曜のみ"]
    sunshine_d = [0.73, 0.89, 0.16, -0.01]
    solar_d = [0.63, 0.76, 0.05, -0.22]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, sunshine_d, width, label="日照時間", color="#2563eb")
    ax.bar(x + width / 2, solar_d, width, label="全天日射量", color="#f59e0b")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cohen's d（制御日 vs 非制御日の差の大きさ）")
    ax.set_title("④ 判別力は日照時間の方が全天日射量より高い（意外な結果）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_discriminative_power.png", dpi=140)
    plt.close(fig)


def fig05_auc_comparison():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    df = build_dataset()
    df["month"] = df["date"].dt.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    month_cols = list(month_dummies.columns)

    n_train = int(len(df) * 0.7)
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    results = []
    for feats, label in [
        (["is_weekend_or_holiday"], "曜日のみ"),
        (["sunshine_h"], "日照時間のみ"),
        (["sunshine_h", "is_weekend_or_holiday"], "日照時間\n+曜日"),
        (["sunshine_h", "is_weekend_or_holiday", "years_since_start"], "+トレンド項"),
        (month_cols + ["sunshine_h", "is_weekend_or_holiday", "years_since_start"], "+月（季節）\n★最大の効果"),
    ]:
        model = LogisticRegression(max_iter=1000)
        model.fit(train[feats], train["is_curtailed"])
        proba = model.predict_proba(test[feats])[:, 1]
        auc = roc_auc_score(test["is_curtailed"], proba)
        results.append((label, auc))

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r[0] for r in results]
    aucs = [r[1] for r in results]
    colors = ["#94a3b8", "#94a3b8", "#2563eb", "#16a34a", "#dc2626"]
    bars = ax.bar(labels, aucs, color=colors)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(0.5, color="gray", ls=":", lw=1.5, label="ランダム(AUC=0.5)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 0.98)
    ax.set_title("⑤ ロジスティック回帰: 月（季節）を入れると精度が跳ね上がる")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_auc_comparison.png", dpi=140)
    plt.close(fig)


def fig06_amgsds_bias():
    categories = ["快晴", "晴れ", "薄曇り〜曇り", "曇り・雨"]
    bias_lead0 = [-5.04, -0.78, 2.61, 2.03]
    bias_lead1 = [-5.30, -0.97, 2.74, 2.43]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, bias_lead0, width, label="当日予測(lead=0)", color="#2563eb")
    ax.bar(x + width / 2, bias_lead1, width, label="前日予測(lead=1)", color="#dc2626")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("予報バイアス (予報−実測, h)")
    ax.set_title("⑥ AMGSDSは快晴日に日照時間を大きく過少評価\n（ml_forecastの実測検証、沖縄）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_amgsds_bias.png", dpi=140)
    plt.close(fig)


def fig07_backtest_comparison():
    df = build_backtest_dataset()
    results = []
    for col, label in [("sunshine_h", "実測そのまま\n(参考上限)"),
                        ("amgsds_proxy_h", "AMGSDS模擬予報\n(快晴バイアス補正)"),
                        ("simplified_h", "簡略化\n(カテゴリ→分位点)")]:
        r = bt_evaluate(df, col, label.replace("\n", " "))
        results.append((label, r["auc"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r[0] for r in results]
    aucs = [r[1] for r in results]
    colors = ["#94a3b8", "#dc2626", "#16a34a"]
    bars = ax.bar(labels, aucs, color=colors)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.55, 0.85)
    ax.set_title("⑦ バックテスト: 簡略化がAMGSDS模擬予報による補正より僅かに優位\n（快晴バイアス補正がリスク信号を弱めるため）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_backtest_comparison.png", dpi=140)
    plt.close(fig)


def fig08_generation_scatter():
    gen = pd.read_csv(PROCESSED_DIR / "kyushu_generation_merged.csv", parse_dates=["date"])
    # 太陽光の導入量が年々増えているため、頭打ち効果を公平に見るには導入量がほぼ一定の
    # 直近期間に限定する（全期間だと「天気」ではなく「導入量の違う年の混在」になってしまう）
    gen = gen[gen["date"] >= "2023-01-01"]
    non_c = gen[gen["is_curtailed"] == 0]
    curt = gen[gen["is_curtailed"] == 1]

    # 非制御日だけで1次式を回帰（「晴れれば発電量も素直に伸びる」という線形の基準線）。
    # 制御日がこの基準線からどれだけ下にずれるかで、頭打ち(非線形)を可視化する。
    slope, intercept = np.polyfit(non_c["solar_mj"], non_c["solar_mwh"], 1)
    r2 = np.corrcoef(non_c["solar_mj"], non_c["solar_mwh"])[0, 1] ** 2
    x_line = np.array([gen["solar_mj"].min(), gen["solar_mj"].max()])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(non_c["solar_mj"], non_c["solar_mwh"], s=10, alpha=0.4, color="#94a3b8", label="非制御日")
    ax.scatter(curt["solar_mj"], curt["solar_mwh"], s=10, alpha=0.4, color="#dc2626", label="制御日")
    ax.plot(x_line, slope * x_line + intercept, color="#1d4ed8", lw=2, ls="--",
            label=f"非制御日の回帰直線 (R²={r2:.2f})\n天候だけなら発電量はこの直線に乗るはず")

    # 制御日の各点が、この基準線からどれだけ下にずれているか（頭打ち分）を薄く示す
    curt_pred = slope * curt["solar_mj"] + intercept
    below = curt_pred - curt["solar_mwh"]
    ax.vlines(curt["solar_mj"], curt["solar_mwh"], curt_pred, color="#dc2626", alpha=0.08, lw=1)

    ax.set_xlabel("実測 全天日射量 (MJ/m²)")
    ax.set_ylabel("実際の発電量（太陽光単独, MWh/日）")
    ax.set_title("⑧ 発電量との関係: 非制御日は線形、制御日は直線から下に外れる（頭打ち＝非線形）\n"
                 f"2023年以降・導入量がほぼ一定の期間に限定／制御日の基準線からの平均ずれ {below.mean():,.0f}MWh",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_generation_scatter.png", dpi=140)
    plt.close(fig)


def fig09_generation_corr_comparison():
    gen = pd.read_csv(PROCESSED_DIR / "kyushu_generation_merged.csv", parse_dates=["date"])
    non_curtailed = gen[gen["is_curtailed"] == 0]
    methods = [("sunshine_h", "日照時間"), ("solar_mj", "全天日射量"),
               ("amgsds_proxy_h", "AMGSDS模擬予報"), ("simplified_h", "簡略化")]
    all_r = [gen["solar_mwh"].corr(gen[c]) for c, _ in methods]
    nc_r = [non_curtailed["solar_mwh"].corr(non_curtailed[c]) for c, _ in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, all_r, width, label="全日", color="#94a3b8")
    ax.bar(x + width / 2, nc_r, width, label="非制御日のみ", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylabel("実際の発電量との相関係数 r")
    ax.set_ylim(0.6, 0.95)
    ax.set_title("⑨ 発電量（太陽光単独）との相関: 全天日射量が最も強い（日照時間より上）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_generation_corr.png", dpi=140)
    plt.close(fig)


def main():
    fig00_pipeline_diagram()
    fig01_scatter_sunshine_curtail()
    fig02_yearly_trend()
    fig03_weekday_ratio()
    fig04_discriminative_power()
    fig05_auc_comparison()
    fig06_amgsds_bias()
    fig07_backtest_comparison()
    fig08_generation_scatter()
    fig09_generation_corr_comparison()
    fig10_seasonal_weekday_effect()
    fig11_okinawa_seasonal()
    fig12_logistic_concept()
    fig13_roc_curve()
    fig14_point_vs_area_comparison()
    print(f"図を保存: {FIG_DIR}")


def fig10_seasonal_weekday_effect():
    df = pd.read_csv(PROCESSED_DIR / "kyushu_curtail_days.csv", parse_dates=["date"])
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6])

    counts = df["month"].value_counts().reindex(range(1, 13), fill_value=0)
    weekend_ratio = df.groupby("month")["is_weekend"].mean().reindex(range(1, 13)) * 100

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(counts.index, counts.values, color="#94a3b8", alpha=0.8, label="制御日数（左軸）")
    ax1.set_xlabel("月")
    ax1.set_ylabel("出力制御の実施日数（2018〜2025年度合計）")
    ax1.set_xticks(range(1, 13))

    ax2 = ax1.twinx()
    # サンプル数が小さい月(n<10)は参考値として区別する（7・8月はn=2〜3のため）
    for m in range(1, 13):
        n = counts[m]
        if n == 0:
            continue
        marker = "o" if n >= 10 else "x"
        alpha = 1.0 if n >= 10 else 0.4
        ax2.plot(m, weekend_ratio[m], marker=marker, color="#dc2626", alpha=alpha, ms=9)
    ax2.plot([], [], marker="o", color="#dc2626", label="土日の割合（右軸, 10件以上）")
    ax2.plot([], [], marker="x", color="#dc2626", alpha=0.4, label="土日の割合（右軸, 10件未満・参考値）")
    ax2.axhline(2 / 7 * 100, color="gray", ls=":", lw=1.5, label="カレンダー上の土日割合(28.6%)")
    ax2.set_ylabel("制御日に占める土日の割合 (%)")
    ax2.set_ylim(0, 105)

    ax1.set_title("⑩ 「土日に多い」は季節で大きく異なる\n（春は制御日自体が多く土日比率は低い、夏〜初秋は制御がまれで土日に偏る）", pad=14)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_seasonal_weekday.png", dpi=140)
    plt.close(fig)


def fig11_okinawa_seasonal():
    df = pd.read_csv(PROCESSED_DIR / "okinawa_curtail_days.csv", parse_dates=["date"])
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6])

    counts = df["month"].value_counts().reindex(range(1, 13), fill_value=0)
    weekend_ratio = df.groupby("month")["is_weekend"].mean().reindex(range(1, 13)) * 100
    n_by_month = df["month"].value_counts().reindex(range(1, 13), fill_value=0)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(counts.index, counts.values, color="#94a3b8", alpha=0.8, label="制御日数（左軸）")
    ax1.set_xlabel("月")
    ax1.set_ylabel("出力制御の実施日数（2023〜2025年度合計）")
    ax1.set_xticks(range(1, 13))

    ax2 = ax1.twinx()
    # サンプル数が小さい月(n<10)は点を薄く・マーカーを変えて示す
    for m in range(1, 13):
        n = n_by_month[m]
        if n == 0:
            continue
        alpha = 1.0 if n >= 10 else 0.4
        marker = "o" if n >= 10 else "x"
        ax2.plot(m, weekend_ratio[m], marker=marker, color="#dc2626", alpha=alpha, ms=9)
    ax2.plot([], [], marker="o", color="#dc2626", label="土日の割合（右軸, 10件以上）")
    ax2.plot([], [], marker="x", color="#dc2626", alpha=0.4, label="土日の割合（右軸, 10件未満・参考値）")
    ax2.axhline(2 / 7 * 100, color="gray", ls=":", lw=1.5, label="カレンダー上の土日割合(28.6%)")
    ax2.set_ylabel("制御日に占める土日の割合 (%)")
    ax2.set_ylim(0, 105)

    ax1.set_title("⑪ 沖縄: 6〜10月は出力制御が0日\n（九州と異なり「季節自体」が制御の有無を分ける）", pad=14)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_okinawa_seasonal.png", dpi=140)
    plt.close(fig)


def fig12_logistic_concept():
    """ロジスティック回帰の仕組みを図解する概念図。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, text, color="#dbeafe", fontsize=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                            linewidth=1.4, edgecolor="#334155", facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                      linewidth=1.6, color="#334155"))

    ax.text(0.2, 6.1, "⑫ ロジスティック回帰の仕組み", fontsize=14, weight="bold")

    box(0.3, 3.9, 2.3, 1.4, "特徴量\n(日照時間・曜日・\n月・経過年数)", color="#fef3c7", fontsize=9.5)
    arrow(2.7, 4.6, 3.6, 4.6)
    box(3.6, 3.9, 2.3, 1.4, "各特徴量に重みを\nかけて合計する\n(回帰式)", color="#dbeafe", fontsize=9.5)
    arrow(6.0, 4.6, 6.9, 4.6)
    box(6.9, 3.9, 2.8, 1.4, "シグモイド関数で\n0〜1の確率に変換\n(出力制御の確率)", color="#e9d5ff", fontsize=9.5)

    arrow(8.3, 3.9, 8.3, 2.7)
    box(5.5, 1.5, 4.2, 1.1, "しきい値(例:0.5)で判定\n確率が0.5以上 → 「制御あり」と予測\n確率<0.5 → 「制御なし」と予測", color="#fee2e2", fontsize=9)

    ax.text(0.2, 1.9, "この「しきい値」の選び方で\n適合率・再現率が変わる\n(下図⑬参照)", fontsize=9.5, color="#334155")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_logistic_concept.png", dpi=140)
    plt.close(fig)


def fig13_roc_curve():
    """フルモデル(月+日照+曜日+トレンド)の実際のROC曲線を描き、AUCの意味を図解する。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score

    df = build_dataset()
    df["month"] = df["date"].dt.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    feats = list(month_dummies.columns) + ["sunshine_h", "is_weekend_or_holiday", "years_since_start"]

    n_train = int(len(df) * 0.7)
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    model = LogisticRegression(max_iter=1000)
    model.fit(train[feats], train["is_curtailed"])
    proba = model.predict_proba(test[feats])[:, 1]
    fpr, tpr, _ = roc_curve(test["is_curtailed"], proba)
    auc = roc_auc_score(test["is_curtailed"], proba)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(fpr, tpr, color="#dc2626", lw=2.5, label=f"フルモデル (AUC={auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#dc2626")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1.5, label="ランダム (AUC=0.5)")
    ax.set_xlabel("偽陽性率\n(制御が無い日を「制御あり」と誤予測した割合)")
    ax.set_ylabel("真陽性率\n(制御がある日を正しく「制御あり」と予測できた割合)")
    ax.set_title("⑬ ROC曲線: AUCは「ランダムな2日を見せたとき\n制御日をより高い確率と正しく順位付けできる割合」")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_roc_curve.png", dpi=140)
    plt.close(fig)


def fig14_point_vs_area_comparison():
    """単一点(福岡)よりAMeDAS九州7県平均・AMGSDSメッシュ面平均の方が精度が上がることを示す。"""
    df = build_dataset()
    df["month"] = df["date"].dt.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    month_cols = list(month_dummies.columns)
    df = df.dropna(subset=["sunshine_h_kyushu_mean", "sunshine_h_amgsds_mesh_mean"]).reset_index(drop=True)

    methods = [
        ("sunshine_h", "福岡単一点"),
        ("sunshine_h_kyushu_mean", "AMeDAS\n九州7県平均"),
        ("sunshine_h_amgsds_mesh_mean", "AMGSDS\n九州メッシュ面平均"),
    ]
    aucs = []
    for col, label in methods:
        r = lm.evaluate(df, month_cols + [col, "is_weekend_or_holiday", "years_since_start"], label.replace("\n", " "))
        aucs.append(r["auc"])

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [m[1] for m in methods]
    colors = ["#94a3b8", "#2563eb", "#16a34a"]
    bars = ax.bar(labels, aucs, color=colors)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
    ax.set_ylabel("ROC-AUC（フルモデル: 月+日照+曜日+トレンド）")
    ax.set_ylim(0.85, 0.95)
    ax.set_title("⑭ 単一点より九州全体の面で見た方が精度が上がる\n（本リポジトリ最高のAUC。AMeDAS平均とAMGSDSメッシュ平均はほぼ同精度）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_point_vs_area.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
