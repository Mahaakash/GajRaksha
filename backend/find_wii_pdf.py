import urllib.request
import urllib.parse
import json
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Search WII website for TR-0039 or Human-elephant conflict West Bengal
search_urls = [
    "https://wii.gov.in/publications",
    "https://wii.gov.in/technical_reports",
    "https://wii.gov.in/human_elephant_conflict",
    "https://wii.gov.in/research_reports"
]

for url in search_urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            print(f"URL: {url} -> Status {resp.status}")
            matches = re.findall(r'href=[\'"]([^\'"]*(?:TR[-_]?0039|elephant|bengal)[^\'"]*)[\'"]', content, re.I)
            if matches:
                print(f"Matches in {url}: {matches[:5]}")
    except Exception as e:
        print(f"Error {url}: {e}")

