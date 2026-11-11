from build_hotspots_dataset import df

print(f"Total rows: {len(df)}")
unique_pairs = df.drop_duplicates(subset=["name", "district"])
print(f"Unique (name, district) pairs: {len(unique_pairs)}")

