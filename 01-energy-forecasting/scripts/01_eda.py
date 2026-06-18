from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="white", palette="muted")
BLUE   = "#147EC5"
ORANGE = "#F1993A"
GRAY   = "#828A8AB3"

DATA_PATH   = Path(__file__).parents[1] / "data/raw/pjm_hourly_est.csv"
CHARTS_PATH = Path(__file__).parents[1] / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

###########################################
# Load & basic info
###########################################
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)
df.info()

###########################################
# Focus on PJME (eastern network, Jan 2002+)
###########################################
series = df[["Datetime", "PJME"]].dropna().copy()
series = series[series["Datetime"] >= "2002-01-01"].reset_index(drop=True)
series = series.rename(columns={"PJME": "mw"})
print(f"Range: {series['Datetime'].min()} → {series['Datetime'].max()}")
print(f"Points: {len(series):,}")
print(series["mw"].describe())

###########################################
# Missing hours & duplicates
###########################################
total_hours = (series["Datetime"].max() - series["Datetime"].min()).total_seconds() / 3600 + 1
missing_hours = int(total_hours) - len(series)
duplicates = series.duplicated("Datetime").sum()
print(f"Expected hourly points : {int(total_hours):,}")
print(f"Missing hours          : {missing_hours:,}")
print(f"Duplicate timestamps   : {duplicates}")

###########################################
# Full series overview
###########################################
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(series["Datetime"], series["mw"], color=BLUE, linewidth=0.4, alpha=0.8)
ax.set_title("PJM East Load — Full Series (2002–2018)", fontsize=16)
ax.set_ylabel("MW", fontsize=15)
ax.tick_params(labelsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "eda_full_series.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Zoom: 4 weeks to see daily pattern
###########################################
window = series[(series["Datetime"] >= "2018-01-01") & (series["Datetime"] < "2018-01-29")]
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(window["Datetime"], window["mw"], color=BLUE, linewidth=1.2)
ax.set_title("PJM East Load — 4-Week Window (Jan 2018)", fontsize=16)
ax.set_ylabel("MW", fontsize=15)
ax.tick_params(labelsize=14)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "eda_4week_zoom.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Distribution
###########################################
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(series["mw"], bins=80, color=BLUE, edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(series["mw"].mean(), color=GRAY, linestyle="--", linewidth=1.5,
           label=f"Mean {series['mw'].mean()/1000:.1f}k MW")
ax.set_title("PJM East Load Distribution", fontsize=16)
ax.set_xlabel("MW", fontsize=15)
ax.set_ylabel("Count", fontsize=15)
ax.tick_params(labelsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend(fontsize=12, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "eda_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Outlier detection via seasonal z-score (month × hour)
###########################################
series["month"] = series["Datetime"].dt.month
series["hour"]  = series["Datetime"].dt.hour
grp = series.groupby(["month", "hour"])["mw"]
series["zscore"] = grp.transform(lambda x: stats.zscore(x, ddof=1))
outliers = series[series["zscore"].abs() > 3]
print(f"Outliers (|z| > 3): {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)")
print(outliers[["Datetime", "mw", "zscore"]].head(10))

###########################################
# Highlight outliers on full series
###########################################
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(series["Datetime"], series["mw"], color=BLUE, linewidth=0.4, alpha=0.7)
ax.scatter(outliers["Datetime"], outliers["mw"], color=ORANGE, s=8, zorder=5,
           label=f"Outliers ({len(outliers)})")
ax.set_title("PJM East Load — Outliers Highlighted", fontsize=16)
ax.set_ylabel("MW", fontsize=15)
ax.tick_params(labelsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend(fontsize=12, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "eda_outliers.png", dpi=150, bbox_inches="tight")
plt.show()
