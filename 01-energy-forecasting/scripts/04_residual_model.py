from pathlib import Path
import pandas as pd

WF_RESULTS_PATH  = Path(__file__).parents[1] / "data/processed/walk_forward_results.csv"
XGB_RESULTS_PATH = Path(__file__).parents[1] / "data/processed/xgb_corrected_results.csv"

EXPERIMENTS = [
    ("exp1_512h",  512),
    ("exp2_672h",  672),
    ("exp3_8736h", 8736),
]

###########################################
# Load TimesFM walk-forward results
###########################################
wf = pd.read_csv(WF_RESULTS_PATH, parse_dates=["date"])
wf["datetime"] = wf["date"] + pd.to_timedelta(wf["hour"], unit="h")
wf["residual"] = wf["actual"] - wf["predicted"]

# TODO: build features and train XGBoost per experiment
# TODO: predict residual and compute predicted_corrected = predicted + residual_pred
# TODO: save xgb_corrected_results.csv
