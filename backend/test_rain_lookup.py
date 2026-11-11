import json
import pandas as pd
import numpy as np

with open("data/cache/kruger_rainfall_daily.json", "r") as f:
    r_data = json.load(f)

daily = r_data["daily"]
df_rain = pd.DataFrame({
    "date": pd.to_datetime(daily["time"]),
    "precip": daily["precipitation_sum"]
}).sort_values("date").reset_index(drop=True)

df_rain["rain_7d"] = df_rain["precip"].rolling(window=7, min_periods=1).sum()
df_rain["rain_30d"] = df_rain["precip"].rolling(window=30, min_periods=1).sum()

# Create fast lookup dict
rain_lookup = {}
for _, row in df_rain.iterrows():
    d_str = row["date"].strftime("%Y-%m-%d")
    rain_lookup[d_str] = (round(float(row["rain_7d"]), 2), round(float(row["rain_30d"]), 2))

print(f"Total dates in rainfall lookup: {len(rain_lookup)}")
print("Sample lookup:", list(rain_lookup.items())[:3])

