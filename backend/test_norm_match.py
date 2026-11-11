from test_norm import local_places, normalize
from build_hotspots_dataset import df

matched = 0
unmatched = []
for _, row in df.iterrows():
    norm_v = normalize(row['name'])
    if norm_v in local_places:
        matched += 1
    else:
        unmatched.append((row['name'], row['block'], row['district']))

print(f"Matched with normalization: {matched} / {len(df)} ({matched/len(df)*100:.1f}%)")

