"""
GajRaksha - WII Report Hotspot Extractor & OpenStreetMap Geocoder
=================================================================
Extracts conflict hotspots in South Bengal (Jhargram, Bankura, Purulia,
Paschim Medinipur) from the Wildlife Institute of India technical report:
"Human-elephant conflict in the State of West Bengal, India (2000-2025)"
(Mathur et al., TR-0039-2026, 96 pages).

Geocodes locations with OpenStreetMap Nominatim Search API:
- Caches all responses in data/cache/osm_nominatim_hotspots_cache.json
- Records latitude, longitude, and geocoded status
- Identifies any locations that could not be located in OpenStreetMap

Outputs:
- data/india_hotspots.csv
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REF_DIR = DATA_DIR / "reference"

PDF_PATH = REF_DIR / "Human_Elephant_Conflict_Westbengal_2000_2025.pdf"
NOMINATIM_CACHE_FILE = CACHE_DIR / "osm_nominatim_hotspots_cache.json"
OUTPUT_HOTSPOTS_CSV = DATA_DIR / "india_hotspots.csv"

TARGET_DISTRICTS = {
    "jhargram": "Jhargram",
    "bankura": "Bankura",
    "purulia": "Purulia",
    "paschim medinipur": "Paschim Medinipur",
    "west medinipur": "Paschim Medinipur",
    "midnapore": "Paschim Medinipur"
}


def extract_hotspots_from_pdf():
    print(f"Reading PDF from: {PDF_PATH}...")
    reader = PdfReader(str(PDF_PATH))
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")

    records = []

    # 1. Appendix I (Elephant Mortality, pages 72-75) & Appendix II (Human Mortality/Injuries, pages 76-93)
    for p in range(71, 94):
        page_num = p + 1
        text = reader.pages[p].extract_text() or ""
        lines = text.split("\n")
        section = "Elephant Mortality (Appendix I)" if page_num <= 75 else "Human Mortality/Injuries (Appendix II)"

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            m = re.match(r"^(\d+)\s+(\d+)\s+(.+)$", line_clean)
            if m:
                s_no = int(m.group(1))
                count = int(m.group(2))
                rest = m.group(3).strip()

                matched_dist = None
                for d_key, d_name in TARGET_DISTRICTS.items():
                    if rest.lower().endswith(d_key):
                        matched_dist = d_name
                        rest_no_dist = rest[: len(rest) - len(d_key)].strip()
                        break

                if matched_dist:
                    parts = rest_no_dist.split()
                    if len(parts) >= 2:
                        if parts[-1] in ["I", "II", "III"] and len(parts) >= 3:
                            subdistrict = parts[-2] + " " + parts[-1]
                            village = " ".join(parts[:-2])
                        else:
                            subdistrict = parts[-1]
                            village = " ".join(parts[:-1])
                    else:
                        village = rest_no_dist
                        subdistrict = ""

                    clean_village = re.sub(r"\s+CT$", "", village)

                    records.append({
                        "name": clean_village,
                        "type": "village",
                        "block": subdistrict,
                        "district": matched_dist,
                        "page_number": page_num,
                        "conflict_count": count,
                        "source": section,
                        "s_no": s_no
                    })

    # 2. Main text high conflict divisions / blocks mentioned
    main_text_entities = [
        {"name": "Bankura North Division", "type": "division", "block": "Bankura North", "district": "Bankura", "page_number": 42, "conflict_count": 120, "source": "Report Section 1.5 (High Conflict Divisions)", "s_no": 0},
        {"name": "Jhargram Division", "type": "division", "block": "Jhargram", "district": "Jhargram", "page_number": 17, "conflict_count": 0, "source": "Report Section 1.1 (Persistent Hotspots)", "s_no": 0},
        {"name": "Rupnarayan Division", "type": "division", "block": "Rupnarayan", "district": "Paschim Medinipur", "page_number": 17, "conflict_count": 0, "source": "Report Section 1.1 (Persistent Hotspots)", "s_no": 0},
        {"name": "Purulia Division", "type": "division", "block": "Purulia", "district": "Purulia", "page_number": 53, "conflict_count": 0, "source": "Report Section 1.7 (Emerging Hotspots)", "s_no": 0},
        {"name": "Panchet Division", "type": "division", "block": "Panchet", "district": "Bankura", "page_number": 67, "conflict_count": 0, "source": "Report Bibliography / Case Study", "s_no": 0},
        {"name": "Barjora Block", "type": "block", "block": "Barjora", "district": "Bankura", "page_number": 66, "conflict_count": 0, "source": "Report Bibliography / Case Study", "s_no": 0}
    ]

    for ent in main_text_entities:
        records.append(ent)

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} total hotspot entries across {df['district'].nunique()} districts.")
    return df


def geocode_all_hotspots(df: pd.DataFrame, max_new_queries: int = 500) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nominatim_cache = {}
    if NOMINATIM_CACHE_FILE.exists():
        try:
            with open(NOMINATIM_CACHE_FILE, "r", encoding="utf-8") as f:
                nominatim_cache = json.load(f)
            print(f"Loaded {len(nominatim_cache)} cached geocoding records.")
        except Exception as e:
            print(f"Cache load warning: {e}")

    headers = {
        "User-Agent": "GajRaksha-HotspotGeocoder/1.0 (contact: dev@gajraksha.local)"
    }

    # Unique entities to geocode: (name, district, type)
    unique_df = df[["name", "district", "type", "block"]].drop_duplicates(subset=["name", "district"])
    print(f"Total unique (name, district) hotspots to geocode: {len(unique_df)}")

    new_queries_run = 0

    for idx, row in unique_df.iterrows():
        name = row["name"]
        dist = row["district"]
        itype = row["type"]
        block = row["block"]
        cache_key = f"{name}___{dist}".lower().strip()

        if cache_key in nominatim_cache:
            continue

        if new_queries_run >= max_new_queries:
            print(f"Reached query limit ({max_new_queries}). Stopping new network queries.")
            break

        # Construct candidate queries
        queries = []
        if itype == "division":
            clean_name = re.sub(r"\s+Division$", "", name)
            queries.append(f"{clean_name}, {dist}, West Bengal, India")
            queries.append(f"{clean_name}, West Bengal, India")
        elif itype == "block":
            clean_name = re.sub(r"\s+Block$", "", name)
            queries.append(f"{clean_name}, {dist}, West Bengal, India")
        else:
            queries.append(f"{name}, {dist}, West Bengal, India")
            if block and block.lower() not in name.lower():
                queries.append(f"{name}, {block}, {dist}, West Bengal, India")
            queries.append(f"{name}, West Bengal, India")

        found = False
        for q in queries:
            url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit=1"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
                        # Check spatial bounds for West Bengal landscape
                        if 21.0 <= lat <= 24.5 and 85.0 <= lon <= 88.5:
                            nominatim_cache[cache_key] = {
                                "lat": lat,
                                "lon": lon,
                                "geocoded": True,
                                "display_name": data[0].get("display_name", ""),
                                "matched_query": q
                            }
                            found = True
                            print(f"[{new_queries_run+1}] ✓ Found: {name} ({dist}) -> ({lat:.4f}, {lon:.4f})")
                            break
            except Exception as e:
                pass
            time.sleep(1.0)
            new_queries_run += 1

        if not found:
            nominatim_cache[cache_key] = {
                "lat": None,
                "lon": None,
                "geocoded": False,
                "display_name": "",
                "matched_query": None
            }
            print(f"[{new_queries_run+1}] ✗ Not found: {name} ({dist})")

        # Save cache every 5 queries
        if new_queries_run % 5 == 0:
            with open(NOMINATIM_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(nominatim_cache, f, indent=2)

    # Final save
    with open(NOMINATIM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(nominatim_cache, f, indent=2)
    print(f"Cache saved to {NOMINATIM_CACHE_FILE} (Total cached entries: {len(nominatim_cache)})")

    # Map geocoding results to original DataFrame
    lats, lons, geocoded_flags, queries_matched = [], [], [], []
    for _, row in df.iterrows():
        cache_key = f"{row['name']}___{row['district']}".lower().strip()
        cached = nominatim_cache.get(cache_key, {"lat": None, "lon": None, "geocoded": False, "matched_query": None})
        lats.append(cached.get("lat"))
        lons.append(cached.get("lon"))
        geocoded_flags.append(cached.get("geocoded", False))
        queries_matched.append(cached.get("matched_query"))

    df["latitude"] = lats
    df["longitude"] = lons
    df["geocoded"] = geocoded_flags
    df["matched_query"] = queries_matched

    # Save to data/india_hotspots.csv
    df.to_csv(OUTPUT_HOTSPOTS_CSV, index=False)
    print(f"✓ Saved {len(df)} records to: {OUTPUT_HOTSPOTS_CSV}")

    located_df = df[df["geocoded"] == True]
    unlocated_df = df[df["geocoded"] == False]

    print(f"\n==========================================")
    print(f"HOTSPOT GEOCODING RESULTS:")
    print(f"==========================================")
    print(f"Total Hotspot Entries:     {len(df)}")
    print(f"Unique Hotspot Names:      {df['name'].nunique()}")
    print(f"Successfully Located:      {len(located_df)} ({len(located_df)/len(df)*100:.1f}%)")
    print(f"Could Not Locate:          {len(unlocated_df)} ({len(unlocated_df)/len(df)*100:.1f}%)")
    print(f"==========================================")

    return df


if __name__ == "__main__":
    df = extract_hotspots_from_pdf()
    geocode_all_hotspots(df)
