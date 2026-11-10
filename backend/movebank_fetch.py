"""
GajRaksha - Movebank Elephant GPS Data Fetcher
==============================================
Downloads elephant GPS tracking data from Movebank using the REST API (direct-read endpoint).
Filters the columns to only the essential fields needed for analyzing elephant presence risk near villages.

Rules & Compliance:
- Reads MOVEBANK_USER, MOVEBANK_PASS, and STUDY_ID from .env.
- If license acceptance is requested, prints the terms and stops without auto-accepting.
- Caches all downloads in /data/cache to avoid re-downloading.
- Saves filtered data to data/raw/movebank_events.csv.
- Strictly keeps: timestamp, location_long, location_lat, individual_local_identifier, tag_local_identifier.
"""

import os
import sys
import csv
import base64
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


# Define directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = RAW_DIR / "movebank_events.csv"


def load_env(env_path: Path) -> dict:
    """
    Simple, zero-dependency .env parser.
    Reads key-value pairs without requiring external third-party libraries.
    """
    config = {}
    if not env_path.exists():
        return config

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines and comments
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip().strip("'\"")
    return config


def check_for_license_terms(content_sample: str) -> bool:
    """
    Checks if the response indicates license terms need to be accepted.
    """
    indicators = [
        "License Terms:",
        "license-md5",
        "Please accept the license terms",
        "terms of use",
        "license terms"
    ]
    lower_sample = content_sample.lower()
    return any(ind.lower() in lower_sample for ind in indicators)


def download_stream(url: str, output_path: Path, auth_header: str = None) -> None:
    """
    Downloads data in chunks with streaming to handle large datasets efficiently.
    """
    headers = {"User-Agent": "GajRaksha/1.0 (Elephant Tracking Research)"}
    if auth_header:
        headers["Authorization"] = auth_header

    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=60) as response:
        # Check first chunk for license agreement
        first_chunk = response.read(4096)
        text_sample = first_chunk.decode("utf-8", errors="replace")

        if check_for_license_terms(text_sample):
            print("\n" + "=" * 70)
            print("🛑 [MOVEBANK LICENSE ACCEPTANCE REQUIRED]")
            print("=" * 70)
            print(text_sample)
            # Read remainder of license if available
            rest = response.read(16384).decode("utf-8", errors="replace")
            print(rest)
            print("=" * 70)
            print("INSTRUCTIONS: Movebank requires users to accept the study license.")
            print("As per project safety rules, license terms are NEVER auto-accepted.")
            print("Please log in to https://www.movebank.org, navigate to the study,")
            print("review and accept the terms, then re-run this script.")
            print("=" * 70)
            sys.exit(0)

        # Write to destination file
        with open(output_path, "wb") as f_out:
            f_out.write(first_chunk)
            while True:
                chunk = response.read(65536)  # 64 KB chunks
                if not chunk:
                    break
                f_out.write(chunk)


def filter_and_format_csv(raw_csv_path: Path, target_csv_path: Path) -> dict:
    """
    Filters the Movebank raw CSV to keep ONLY the requested columns:
    - timestamp
    - location_long
    - location_lat
    - individual_local_identifier
    - tag_local_identifier
    
    Returns summary statistics: row count, number of animals, date range.
    """
    target_csv_path.parent.mkdir(parents=True, exist_ok=True)

    animals = set()
    earliest_date = None
    latest_date = None
    row_count = 0

    with open(raw_csv_path, "r", encoding="utf-8", errors="replace") as f_in:
        reader = csv.reader(f_in)
        headers = next(reader, None)
        if not headers:
            raise ValueError("The downloaded Movebank CSV file is empty.")

        # Normalize header names (lowercase and replace hyphens with underscores)
        header_map = {h.lower().replace("-", "_"): idx for idx, h in enumerate(headers)}

        # Find column indices for requested fields
        time_idx = header_map.get("timestamp")
        long_idx = header_map.get("location_long")
        lat_idx = header_map.get("location_lat")
        animal_idx = header_map.get("individual_local_identifier")
        tag_idx = header_map.get("tag_local_identifier")

        missing = []
        if time_idx is None: missing.append("timestamp")
        if long_idx is None: missing.append("location_long")
        if lat_idx is None: missing.append("location_lat")
        if animal_idx is None: missing.append("individual_local_identifier")
        if tag_idx is None: missing.append("tag_local_identifier")

        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}. Available: {headers}")

        with open(target_csv_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            # Write exact required header
            writer.writerow([
                "timestamp",
                "location_long",
                "location_lat",
                "individual_local_identifier",
                "tag_local_identifier"
            ])

            for row in reader:
                if len(row) <= max(time_idx, long_idx, lat_idx, animal_idx, tag_idx):
                    continue

                ts_str = row[time_idx].strip()
                lng_str = row[long_idx].strip()
                lat_str = row[lat_idx].strip()
                animal_id = row[animal_idx].strip()
                tag_id = row[tag_idx].strip()

                # Skip invalid or empty coordinates
                if not lng_str or not lat_str or not ts_str:
                    continue

                try:
                    lng_val = float(lng_str)
                    lat_val = float(lat_str)
                except ValueError:
                    continue

                # Parse date to track date range
                try:
                    # Movebank timestamps usually formatted like '2007-08-13 00:30:00.000'
                    ts_clean = ts_str.split(".")[0]
                    dt = datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S")
                    if earliest_date is None or dt < earliest_date:
                        earliest_date = dt
                    if latest_date is None or dt > latest_date:
                        latest_date = dt
                except Exception:
                    pass

                animals.add(animal_id)
                row_count += 1

                writer.writerow([ts_str, lng_val, lat_val, animal_id, tag_id])

    return {
        "row_count": row_count,
        "num_animals": len(animals),
        "animals": sorted(list(animals)),
        "earliest_date": earliest_date.strftime("%Y-%m-%d %H:%M:%S") if earliest_date else "N/A",
        "latest_date": latest_date.strftime("%Y-%m-%d %H:%M:%S") if latest_date else "N/A"
    }


def main():
    print("🐘 GajRaksha - Movebank Elephant Data Fetcher")
    print("-" * 50)

    # 1. Ensure required directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Read configuration from .env
    env_file = BASE_DIR / ".env"
    env_vars = load_env(env_file)

    movebank_user = env_vars.get("MOVEBANK_USER", os.environ.get("MOVEBANK_USER", "")).strip()
    movebank_pass = env_vars.get("MOVEBANK_PASS", os.environ.get("MOVEBANK_PASS", "")).strip()
    study_id = env_vars.get("STUDY_ID", os.environ.get("STUDY_ID", "736029750")).strip()

    if not study_id:
        study_id = "736029750"

    print(f"Study ID: {study_id}")
    cached_raw_file = CACHE_DIR / f"movebank_events_raw_{study_id}.csv"

    # 3. Check cache rule: Cache every download in /data/cache and never re-download
    if cached_raw_file.exists() and cached_raw_file.stat().st_size > 0:
        print(f"✓ Found cached download at: {cached_raw_file.relative_to(BASE_DIR)}")
        print("  Skipping re-download to respect cache rule.")
    else:
        # Prepare API URL
        api_url = (
            f"https://www.movebank.org/movebank/service/direct-read"
            f"?entity_type=event&study_id={study_id}&sensor_type=gps"
        )
        print(f"Connecting to Movebank direct-read endpoint for Study ID {study_id}...")

        auth_header = None
        if movebank_user and movebank_pass:
            auth_bytes = f"{movebank_user}:{movebank_pass}".encode("utf-8")
            auth_header = f"Basic {base64.b64encode(auth_bytes).decode('ascii')}"

        try:
            download_stream(api_url, cached_raw_file, auth_header=auth_header)
            print(f"✓ Raw data downloaded and cached at: {cached_raw_file.relative_to(BASE_DIR)}")
        except urllib.error.HTTPError as e:
            if e.code == 401 and not (movebank_user and movebank_pass):
                print(f"ℹ Movebank direct-read requires credentials for Study ID {study_id}.")
                # If Study 736029750, fallback to Movebank Data Repository open direct bitstream
                if study_id == "736029750":
                    print("ℹ Fetching published open-access Movebank repository bitstream for Study 736029750...")
                    repo_url = "https://datarepository.movebank.org/server/api/core/bitstreams/10240da1-0a99-42bf-a34f-eff15883bbfb/content"
                    download_stream(repo_url, cached_raw_file)
                    print(f"✓ Raw data downloaded and cached at: {cached_raw_file.relative_to(BASE_DIR)}")
                else:
                    print(f"Error 401: Please add your MOVEBANK_USER and MOVEBANK_PASS to .env.")
                    sys.exit(1)
            else:
                print(f"HTTP Error {e.code}: {e.reason}")
                sys.exit(1)
        except Exception as e:
            print(f"Error downloading data: {e}")
            sys.exit(1)

    # 4. Filter and save to data/raw/movebank_events.csv
    print(f"Filtering dataset to required columns...")
    stats = filter_and_format_csv(cached_raw_file, OUTPUT_FILE)

    # 5. Output summary statistics
    print("\n" + "=" * 55)
    print("GAJRAKSHA - ELEPHANT TRACKING DATA SUMMARY")
    print("=" * 55)
    print(f"Output File:       {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"Total Row Count:   {stats['row_count']:,}")
    print(f"Number of Animals: {stats['num_animals']}")
    print(f"Animal IDs:        {', '.join(stats['animals'][:10])}{'...' if len(stats['animals']) > 10 else ''}")
    print(f"Date Range:        {stats['earliest_date']} to {stats['latest_date']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
