from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="white", palette="muted")
BLUE   = "#147EC5"
ORANGE = "#F1993A"
GRAY   = "#828A8AB3"

DATA_PATH   = Path(__file__).parents[1] / "data/raw/gasoline_weekly.csv"
CHARTS_PATH = Path(__file__).parents[1] / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

###########################################
# Load & basic info
###########################################
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Range  : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Weeks  : {len(df)}")
print(f"Units  : thousand barrels per day (kbpd)")
print(df["kbpd"].describe().round(1))

###########################################
# Missing weeks
###########################################
expected = pd.date_range(df["date"].min(), df["date"].max(), freq="W-FRI")
missing  = expected.difference(df["date"])
print(f"\nExpected weekly points : {len(expected)}")
print(f"Missing weeks          : {len(missing)}")
if len(missing):
    print(missing)

###########################################
# Full series overview — with COVID shock highlighted
###########################################
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(df["date"], df["kbpd"], color=BLUE, linewidth=0.9)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
           alpha=0.15, color=ORANGE, label="COVID shock (Mar–Jun 2020)")
ax.set_title("U.S. Finished Motor Gasoline — Product Supplied (2010–2025)", fontsize=14)
ax.set_ylabel("Thousand Barrels/Day", fontsize=12)
ax.tick_params(labelsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "eda_full_series.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Annual seasonality: median demand by ISO week + overall mean line
###########################################
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
seasonal = df[df["date"].dt.year != 2020].groupby("week_of_year")["kbpd"].median()
overall_mean = seasonal.mean()

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(seasonal.index, seasonal.values, color=BLUE, linewidth=1.8, label="Median demand")
ax.axhline(overall_mean, color=ORANGE, linewidth=1.4, linestyle="--",
           label=f"Annual mean ({overall_mean/1000:.1f}k kbpd)")
# ax.fill_between(seasonal.index, seasonal.values, overall_mean,
#                 where=(seasonal.values >= overall_mean),
#                 alpha=0.12, color=BLUE, label="Above mean (peak season)")
# ax.fill_between(seasonal.index, seasonal.values, overall_mean,
#                 where=(seasonal.values < overall_mean),
#                 alpha=0.12, color=GRAY, label="Below mean (off-season)")
ax.set_title("Annual Seasonality — Median Demand by Week of Year (excl. 2020)", fontsize=14)
ax.set_xlabel("ISO Week", fontsize=12)
ax.set_ylabel("Thousand Barrels/Day", fontsize=12)
ax.tick_params(labelsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "eda_seasonality.png", dpi=150, bbox_inches="tight")
plt.show()