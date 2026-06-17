from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", palette="muted")
BLUE   = "#147EC5"
ORANGE = "#1D913E"
GRAY   = "#828A8AB3"
RED    = "#E74C3C"

WF_RESULTS_PATH  = Path(__file__).parents[1] / "data/processed/walk_forward_results.csv"
XGB_RESULTS_PATH = Path(__file__).parents[1] / "data/processed/xgb_corrected_results.csv"
CHARTS_PATH      = Path(__file__).parents[1] / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

EXPERIMENTS = [
    ("exp1_512h",  512),
    ("exp2_672h",  672),
    ("exp3_8736h", 8736),
]

###########################################
# Load results
###########################################
wf = pd.read_csv(WF_RESULTS_PATH, parse_dates=["date"])
wf["datetime"] = wf["date"] + pd.to_timedelta(wf["hour"], unit="h")

# xgb = pd.read_csv(XGB_RESULTS_PATH, parse_dates=["date"])  # uncomment after 04 is ready

###########################################
# Metrics
###########################################
print("\n=== Metrics ===")
for i, (exp_name, _) in enumerate(EXPERIMENTS):
    df         = wf[wf["experiment"] == exp_name]
    mae        = (df["actual"] - df["predicted"]).abs().mean()
    mape       = ((df["actual"] - df["predicted"]).abs() / df["actual"]).mean() * 100
    mae_naive  = (df["actual"] - df["naive"]).abs().mean()
    mape_naive = ((df["actual"] - df["naive"]).abs() / df["actual"]).mean() * 100
    smae       = mae / mae_naive
    if i == 0:
        print(f"  {'naive_lag24h':12s}: MAE={mae_naive:,.0f} MW | MAPE={mape_naive:.2f}%")
        print(f"  {'-'*60}")
    print(f"  {exp_name:12s}: MAE={mae:,.0f} MW | MAPE={mape:.2f}% | Scaled MAE={smae:.3f}")

###########################################
# Chart 1 — Full test period
###########################################
n_exp = len(EXPERIMENTS)

fig, axes = plt.subplots(n_exp, 1, figsize=(18, 4 * n_exp), sharex=True)
if n_exp == 1:
    axes = [axes]
fig.suptitle("TimesFM — Actual vs Predicted (full test period)", fontsize=18, y=1.01)

for ax, (exp_name, _) in zip(axes, EXPERIMENTS):
    df = wf[wf["experiment"] == exp_name].sort_values("datetime")
    ax.plot(df["datetime"], df["actual"],    color=GRAY, linewidth=0.5, alpha=0.8, label="Actual")
    ax.plot(df["datetime"], df["predicted"], color=BLUE, linewidth=0.5, alpha=0.9, label="TimesFM")
    ax.set_title(exp_name, fontsize=16)
    ax.set_ylabel("MW", fontsize=15)
    ax.tick_params(labelsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "forecast_full_period.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Chart 2 — Zoom: 2-week window
###########################################
zoom_start = pd.Timestamp("2018-03-01")
zoom_end   = pd.Timestamp("2018-03-14 23:00")

fig, axes = plt.subplots(n_exp + 1, 1, figsize=(16, 4 * (n_exp + 1)), sharex=True)
fig.suptitle("TimesFM — Zoom: 2-week window (Mar 2018)", fontsize=18, y=1.01)

# Naive subplot (top)
df_naive = (
    wf[wf["experiment"] == EXPERIMENTS[0][0]]
    .sort_values("datetime")
    .query("@zoom_start <= datetime <= @zoom_end")
)
axes[0].plot(df_naive["datetime"], df_naive["actual"], color=GRAY,   linewidth=1.5, alpha=0.8, label="Actual")
axes[0].plot(df_naive["datetime"], df_naive["naive"],  color=ORANGE, linewidth=1.5, alpha=0.9, label="Naive (lag-24h)")
axes[0].set_title("naive_lag24h", fontsize=16)
axes[0].set_ylabel("MW", fontsize=15)
axes[0].tick_params(labelsize=14)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
axes[0].legend(loc="upper right", fontsize=12, framealpha=0.9)
sns.despine(ax=axes[0])

for ax, (exp_name, _) in zip(axes[1:], EXPERIMENTS):
    df = (
        wf[wf["experiment"] == exp_name]
        .sort_values("datetime")
        .query("@zoom_start <= datetime <= @zoom_end")
    )
    ax.plot(df["datetime"], df["actual"],    color=GRAY, linewidth=1.5, alpha=0.8, label="Actual")
    ax.plot(df["datetime"], df["predicted"], color=BLUE, linewidth=1.5, alpha=0.9, label="TimesFM")
    ax.set_title(exp_name, fontsize=16)
    ax.set_ylabel("MW", fontsize=15)
    ax.tick_params(labelsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "forecast_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
