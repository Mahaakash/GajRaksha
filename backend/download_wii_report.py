import os
import shutil
import urllib.request
import sys

def download_wii_report():
    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    
    url = "https://wii.gov.in/uploads/media/pdf/documents/Human_Elephant_Conflict_Westbengal_2000_2025.pdf"
    cache_path = os.path.join("data", "cache", "Human_Elephant_Conflict_Westbengal_2000_2025.pdf")
    ref_path = os.path.join("data", "reference", "Human_Elephant_Conflict_Westbengal_2000_2025.pdf")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000000:
        print(f"Report already cached at {cache_path} ({os.path.getsize(cache_path)} bytes).")
    else:
        print(f"Downloading from {url}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            total_size = int(resp.headers.get('content-length', 0))
            print(f"Total size: {total_size / (1024*1024):.2f} MB")
            with open(cache_path, 'wb') as f:
                downloaded = 0
                chunk_size = 1024 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloaded {downloaded / (1024*1024):.2f} MB...", end='\r')
        print(f"\nDownload finished: {cache_path} ({os.path.getsize(cache_path)} bytes).")
    
    # Copy to data/reference/
    shutil.copy2(cache_path, ref_path)
    print(f"Copied report to {ref_path} ({os.path.getsize(ref_path)} bytes).")

if __name__ == '__main__':
    download_wii_report()

