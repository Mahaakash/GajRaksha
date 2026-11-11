import pandas as pd

df = pd.read_parquet("data/clean_tracks.parquet")
print("Columns:", list(df.columns))
print("Total rows:", len(df))
print(f"Latitude range:  [{df['location_lat'].min():.4f}, {df['location_lat'].max():.4f}]")
print(f"Longitude range: [{df['location_long'].min():.4f}, {df['location_long'].max():.4f}]")

