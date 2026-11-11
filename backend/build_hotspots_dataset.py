from pypdf import PdfReader
import re
import pandas as pd

reader = PdfReader(r"data\reference\Human_Elephant_Conflict_Westbengal_2000_2025.pdf")

target_districts = {
    "jhargram": "Jhargram",
    "bankura": "Bankura",
    "purulia": "Purulia",
    "paschim medinipur": "Paschim Medinipur",
    "west medinipur": "Paschim Medinipur",
    "midnapore": "Paschim Medinipur"
}

records = []

# Scan Appendix I (pages 72-75) and Appendix II (pages 76-93)
for p in range(71, 94):
    page_num = p + 1
    text = reader.pages[p].extract_text() or ""
    lines = text.split("\n")
    section = "Elephant Mortality (Appendix I)" if page_num <= 75 else "Human Mortality and Injuries (Appendix II)"
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        
        m = re.match(r'^(\d+)\s+(\d+)\s+(.+)$', line_clean)
        if m:
            s_no = int(m.group(1))
            count = int(m.group(2))
            rest = m.group(3).strip()
            
            matched_dist = None
            for d_key, d_name in target_districts.items():
                if rest.lower().endswith(d_key):
                    matched_dist = d_name
                    rest_no_dist = rest[:len(rest) - len(d_key)].strip()
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
                
                records.append({
                    "name": village,
                    "type": "village",
                    "block": subdistrict,
                    "district": matched_dist,
                    "page_number": page_num,
                    "conflict_count": count,
                    "source": section,
                    "s_no": s_no
                })

# Main text divisions/blocks
main_text_entities = [
    {"name": "Bankura North Division", "type": "division", "block": "Bankura North", "district": "Bankura", "page_number": 42, "conflict_count": 120, "source": "Report Section 1.5 (High Conflict Divisions)"},
    {"name": "Jhargram Division", "type": "division", "block": "Jhargram", "district": "Jhargram", "page_number": 17, "conflict_count": 0, "source": "Report Section 1.1 (Persistent Hotspots)"},
    {"name": "Rupnarayan Division", "type": "division", "block": "Rupnarayan", "district": "Paschim Medinipur", "page_number": 17, "conflict_count": 0, "source": "Report Section 1.1 (Persistent Hotspots)"},
    {"name": "Purulia Division", "type": "division", "block": "Purulia", "district": "Purulia", "page_number": 53, "conflict_count": 0, "source": "Report Section 1.7 (Emerging Hotspots)"},
    {"name": "Panchet Division", "type": "division", "block": "Panchet", "district": "Bankura", "page_number": 67, "conflict_count": 0, "source": "Report Bibliography / Case Study"},
    {"name": "Barjora Block", "type": "block", "block": "Barjora", "district": "Bankura", "page_number": 66, "conflict_count": 0, "source": "Report Bibliography / Case Study"}
]

for entity in main_text_entities:
    entity["s_no"] = 0
    records.append(entity)

df = pd.DataFrame(records)
print(f"Total entries: {len(df)}")
print(f"By district:\n{df['district'].value_counts()}")
print(f"By type:\n{df['type'].value_counts()}")

