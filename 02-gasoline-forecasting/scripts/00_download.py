from pathlib import Path
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("EIA_API_KEY", "")

START    = "2010-01-01"
END      = "2025-12-31"
RAW_PATH = Path(__file__).parents[1] / "data/raw/gasoline_weekly.csv"

###########################################
# Guard: missing or unfilled key
###########################################
if not API_KEY or API_KEY == "YOUR_EIA_API_KEY":
    raise ValueError(
        "EIA_API_KEY not set.\n"
        "Add your key to .env: EIA_API_KEY=abc123\n"
        "Register free at: https://www.eia.gov/opendata/register.php"
    )

###########################################
# Fetch from EIA API v2
# product=EPM0  → Finished Motor Gasoline (all grades, all formulations)
# duoarea=NUS   → U.S. total
# URL built as string — requests encodes brackets otherwise, breaking EIA v2
###########################################
url = (
    "https://api.eia.gov/v2/petroleum/cons/wpsup/data/"
    f"?api_key={API_KEY}"
    f"&frequency=weekly"
    f"&data[0]=value"
    f"&facets[product][]=EPM0F"
    f"&facets[duoarea][]=NUS"
    f"&start={START}"
    f"&end={END}"
    f"&sort[0][column]=period"
    f"&sort[0][direction]=asc"
    f"&length=5000"
)
resp = requests.get(url, timeout=30)
resp.raise_for_status()
payload = resp.json()

records = payload["response"]["data"]
print(f"Points returned : {len(records)}")
print(f"Sample record   : {records[0]}")

###########################################
# Parse
###########################################
df = pd.DataFrame(records)[["period", "value", "units", "series-description"]]
df = df.rename(columns={"period": "date", "value": "kbpd"})
df["date"] = pd.to_datetime(df["date"])
df["kbpd"] = pd.to_numeric(df["kbpd"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

print(f"\nSeries : {df['series-description'].iloc[0]}")
print(f"Units  : {df['units'].iloc[0]}")
print(f"Range  : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Weeks  : {len(df)}")
print(f"Missing: {df['kbpd'].isna().sum()}")
print(df[["date", "kbpd"]].head())

###########################################
# Save (only date + kbpd)
###########################################
df[["date", "kbpd"]].to_csv(RAW_PATH, index=False)
print(f"\nSaved → {RAW_PATH}")