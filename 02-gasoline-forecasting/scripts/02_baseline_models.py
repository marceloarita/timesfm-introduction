"""
02_baseline_models.py
─────────────────────
TimesFM baseline for U.S. gasoline demand (2024 test set).

Runs a context-length sweep (52–416 weeks) to find the optimal context window.
The selected context (156 weeks, elbow) becomes the TimesFM baseline.

Price data is also downloaded here and cached for use in script 03.

Outputs:
  charts/elbow_context_length.png
  data/processed/walk_forward_results.csv   (TimesFM sweep — cached)
  data/processed/baseline_results_2024.csv  (TimesFM predictions for script 03)
  data/raw/gasoline_price_weekly.csv        (cached for script 03)
"""

from pathlib import Path
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from dotenv import load_dotenv

sns.set_theme(style="white", palette="muted")
BLUE = "#147EC5"
GRAY = "#95A5A6"

BASE_DIR     = Path(__file__).parents[1]
DATA_PATH    = BASE_DIR / "data/raw/gasoline_weekly.csv"
PRICE_PATH   = BASE_DIR / "data/raw/gasoline_price_weekly.csv"
WF_PATH      = BASE_DIR / "data/processed/walk_forward_results.csv"
RESULTS_PATH = BASE_DIR / "data/processed/baseline_results_2024.csv"
CHARTS_PATH  = BASE_DIR / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

CONTEXT_LEN = 156
HORIZON     = 1
TEST_YEAR   = 2024

# ── 1. Load demand ─────────────────────────────────────────────────────────────
demand = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
values = demand["kbpd"].to_numpy(dtype=float)
dates  = demand["date"]
print(f"Demand: {len(demand)} weeks  ({demand['date'].min().date()} → {demand['date'].max().date()})")

# ── 2. Download / cache price data (used by script 03) ─────────────────────────
load_dotenv()
API_KEY = os.environ.get("EIA_API_KEY", "")

if not PRICE_PATH.exists():
    print("Downloading price data from EIA...")
    url = (
        "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
        f"?api_key={API_KEY}"
        "&frequency=weekly"
        "&data[0]=value"
        "&facets[product][]=EPMR"
        "&facets[duoarea][]=NUS"
        "&start=2010-01-01"
        "&end=2025-12-31"
        "&sort[0][column]=period"
        "&sort[0][direction]=asc"
        "&length=5000"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    records = resp.json()["response"]["data"]
    if not records:
        raise ValueError("EIA price API returned 0 records.")
    price_df = pd.DataFrame(records)[["period", "value"]]
    price_df.columns = ["date", "price_usd"]
    price_df["date"]      = pd.to_datetime(price_df["date"])
    price_df["price_usd"] = pd.to_numeric(price_df["price_usd"], errors="coerce")
    price_df = price_df.sort_values("date").reset_index(drop=True)
    price_df.to_csv(PRICE_PATH, index=False)
    print(f"  Saved → {PRICE_PATH} ({len(price_df)} weeks)")
else:
    print(f"Price data loaded from cache ({PRICE_PATH.name})")

# ── 3. TimesFM context sweep (load cache or run) ───────────────────────────────
if WF_PATH.exists():
    wf_results = pd.read_csv(WF_PATH, parse_dates=["date"])
    print(f"\nTimesFM sweep loaded from cache ({len(wf_results)} records, "
          f"{wf_results['experiment'].nunique()} experiments)")
else:
    print("\nRunning TimesFM context sweep (no cache found)...")
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.timesfm_2p5.timesfm_2p5_base import ForecastConfig

    tfm = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    EXPERIMENTS    = [(f"exp_{w}w", w) for w in range(52, 417, 52)]
    test_positions = demand[demand["date"].dt.year == TEST_YEAR].index.tolist()

    records = []
    for exp_name, ctx_len in EXPERIMENTS:
        tfm.compile(ForecastConfig(max_context=ctx_len, max_horizon=128, per_core_batch_size=1))
        print(f"  [{exp_name}] {len(test_positions)} steps")
        for pos in test_positions:
            if pos < ctx_len + 52:
                continue
            ctx   = values[pos - ctx_len : pos]
            pf, _ = tfm.forecast(horizon=HORIZON, inputs=[ctx])
            records.append({
                "experiment": exp_name, "context_len": ctx_len,
                "date":       dates[pos],
                "actual":     values[pos],
                "predicted":  float(pf[0][0]),
                "naive":      values[pos - 52],
            })

    wf_results = pd.DataFrame(records)
    WF_PATH.parent.mkdir(exist_ok=True)
    wf_results.to_csv(WF_PATH, index=False)
    print(f"  Saved → {WF_PATH}")

# ── 4. Elbow plot — Scaled MAE vs context length ───────────────────────────────
summary_rows = []
for ctx_len, grp in wf_results.groupby("context_len"):
    m         = (grp["predicted"] - grp["actual"]).abs().mean()
    naive_mae = (grp["naive"]     - grp["actual"]).abs().mean()
    summary_rows.append({
        "ctx_len":    ctx_len,
        "exp_name":   grp["experiment"].iloc[0],
        "scaled_mae": m / naive_mae,
    })
summary_df = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(summary_df["ctx_len"], summary_df["scaled_mae"],
        color=BLUE, linewidth=2, marker="o", markersize=7)
for _, row in summary_df.iterrows():
    ax.annotate(f"{row['scaled_mae']:.3f}",
                xy=(row["ctx_len"], row["scaled_mae"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=11, color=BLUE)
ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.2, label="Naive lag-52 (= 1.0)")
ax.set_title("TimesFM — Scaled MAE vs Context Length (2024 test)", fontsize=14)
ax.set_xlabel("Context Length (weeks)", fontsize=12)
ax.set_ylabel("Scaled MAE", fontsize=12)
ax.set_xticks(summary_df["ctx_len"])
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "elbow_context_length.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 5. Save TimesFM exp_156w predictions for script 03 ────────────────────────
tfm_156 = (
    wf_results[wf_results["experiment"] == f"exp_{CONTEXT_LEN}w"]
    [["date", "actual", "naive", "predicted"]]
    .rename(columns={"predicted": "timesfm"})
    .reset_index(drop=True)
)

naive_mae = (tfm_156["naive"] - tfm_156["actual"]).abs().mean()
tfm_mae   = (tfm_156["timesfm"] - tfm_156["actual"]).abs().mean()
print(f"\nTimesFM exp_{CONTEXT_LEN}w (2024):")
print(f"  MAE        = {tfm_mae:.1f} kbpd")
print(f"  Scaled MAE = {tfm_mae / naive_mae:.3f}")

tfm_156.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved → {RESULTS_PATH}  ({len(tfm_156)} weeks)")
