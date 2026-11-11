import os
import json
import numpy as np
import pandas as pd

tracks_file = "data/clean_tracks.parquet"
df_tracks = pd.read_parquet(tracks_file)

lat_min, lat_max = df_tracks["location_lat"].min(), df_tracks["location_lat"].max()
lon_min, lon_max = df_tracks["location_long"].min(), df_tracks["location_long"].max()

print(f"Tracking BBox: Lat [{lat_min:.4f}, {lat_max:.4f}], Lon [{lon_min:.4f}, {lon_max:.4f}]")

