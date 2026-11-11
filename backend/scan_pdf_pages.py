from pypdf import PdfReader
import re

pdf_path = r"data\reference\Human_Elephant_Conflict_Westbengal_2000_2025.pdf"
reader = PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")

# Print title / TOC or search for South Bengal districts
keywords = ["jhargram", "bankura", "purulia", "medinipur", "midnapore", "south bengal"]

found_pages = {}
for idx, page in enumerate(reader.pages):
    page_num = idx + 1
    text = page.extract_text() or ""
    text_lower = text.lower()
    matches = [k for k in keywords if k in text_lower]
    if matches:
        found_pages[page_num] = (matches, text)

print(f"Pages mentioning South Bengal districts: {list(found_pages.keys())}")

for page_num in sorted(found_pages.keys()):
    matches, text = found_pages[page_num]
    print(f"--- PAGE {page_num} (Matches: {matches}) ---")
    # Print lines containing matches or hotspot keywords
    lines = text.split("\n")
    for line in lines:
        if any(k in line.lower() for k in keywords + ["range", "block", "division", "hotspot", "conflict"]):
            print(f"  {line.strip()}")

