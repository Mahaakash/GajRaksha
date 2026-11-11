from pypdf import PdfReader
import re

reader = PdfReader(r"data\reference\Human_Elephant_Conflict_Westbengal_2000_2025.pdf")

target_districts = ["jhargram", "bankura", "purulia", "paschim medinipur", "west medinipur", "midnapore"]

print(f"Total pages: {len(reader.pages)}")

# Check pages 1 to 71 (main report)
for idx in range(71):
    page_num = idx + 1
    text = reader.pages[idx].extract_text() or ""
    text_lower = text.lower()
    
    # Check if any target district or related keywords appear
    for dist in target_districts:
        if dist in text_lower:
            # Look for context: range, block, division, hotspot, beat, village
            print(f"\n--- Page {page_num} mentions {dist} ---")
            lines = text.split("\n")
            for line in lines:
                if any(w in line.lower() for w in [dist, "range", "division", "block", "hotspot", "beat", "mouza", "rupnarayan"]):
                    print(f"  [p.{page_num}] {line.strip()}")

