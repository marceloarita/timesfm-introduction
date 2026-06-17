###########################################
# Import libraries
###########################################
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="muted")
BLUE = "#147EC5"
GRAY = "#95A5A6"

DATA_PATH = Path(__file__).parents[1] / "data/raw/pjm_hourly_est.csv"

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
ax.set(title="PJM East Load — Full Series (2002–2018)", ylabel="MW", xlabel="")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
sns.despine()
plt.tight_layout()
plt.show()

###########################################
# Zoom: 4 weeks to see daily pattern
###########################################
window = series[(series["Datetime"] >= "2018-01-01") & (series["Datetime"] < "2018-01-29")]
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(window["Datetime"], window["mw"], color=BLUE, linewidth=1.2)
ax.set(title="PJM East Load — 4-Week Window (Jan 2018)", ylabel="MW", xlabel="")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
sns.despine()
plt.tight_layout()
plt.show()

###########################################
# Distribution
###########################################
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(series["mw"], bins=80, color=BLUE, edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(series["mw"].mean(), color=GRAY, linestyle="--", linewidth=1.2, label=f"Mean {series['mw'].mean()/1000:.1f}k MW")
ax.set(title="PJM East Load Distribution", xlabel="MW", ylabel="Count")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend()
sns.despine()
plt.tight_layout()
plt.show()

###########################################
# Outlier detection via seasonal z-score (month × hour)
###########################################
series["month"] = series["Datetime"].dt.month
series["hour"] = series["Datetime"].dt.hour
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
ax.scatter(outliers["Datetime"], outliers["mw"], color="#F1993A", s=8, zorder=5, label=f"Outliers ({len(outliers)})")
ax.set(title="PJM East Load — Outliers Highlighted", ylabel="MW", xlabel="")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend()
sns.despine()
plt.tight_layout()
plt.show()
