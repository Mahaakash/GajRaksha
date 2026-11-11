import urllib.request
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

url = "https://wii.gov.in/news/human_elephant_conflict_reports"
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode('utf-8', errors='ignore')
    print("Page length:", len(html))
    # Find all links to pdf or reports
    links = re.findall(r'<a\s+[^>]*href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', html, re.I | re.DOTALL)
    for href, text in links:
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        if 'bengal' in href.lower() or 'bengal' in clean_text.lower() or 'conflict' in href.lower() or 'pdf' in href.lower():
            print(f"HREF: {href} | TEXT: {clean_text}")

