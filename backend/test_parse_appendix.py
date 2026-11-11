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
    
    # Check section title
    section = "Appendix I (Elephant Mortality)" if page_num <= 75 else "Appendix II (Human Mortality/Injuries)"
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Look for rows starting with number: e.g., "36 10 Barjora CT Barjora Bankura"
        m = re.match(r'^(\d+)\s+(\d+)\s+(.+)$', line_clean)
        if m:
            s_no = int(m.group(1))
            count = int(m.group(2))
            rest = m.group(3).strip()
            
            # Check if any target district is at the end of the line
            matched_dist = None
            for d_key, d_name in target_districts.items():
                if rest.lower().endswith(d_key):
                    matched_dist = d_name
                    # strip district from rest
                    rest_no_dist = rest[:len(rest) - len(d_key)].strip()
                    break
            
            if matched_dist:
                # rest_no_dist contains village name and subdistrict
                # Usually: <Village Name> <Subdistrict>
                parts = rest_no_dist.split()
                if len(parts) >= 2:
                    subdistrict = parts[-1]
                    # Handle multi-word subdistricts like "Garbeta I", "Barjora", "Jhargram", "Purulia II", "Jhalda I", "Gopiballavpur II", etc.
                    if parts[-1] in ["I", "II"] and len(parts) >= 3:
                        subdistrict = parts[-2] + " " + parts[-1]
                        village = " ".join(parts[:-2])
                    else:
                        subdistrict = parts[-1]
                        village = " ".join(parts[:-1])
                else:
                    village = rest_no_dist
                    subdistrict = ""
                
                records.append({
                    "s_no": s_no,
                    "count": count,
                    "name": village,
                    "subdistrict_block": subdistrict,
                    "district": matched_dist,
                    "type": "village",
                    "page_number": page_num,
                    "section": section,
                    "raw_line": line_clean
                })

df_appendix = pd.DataFrame(records)
print(f"Total appendix records extracted: {len(df_appendix)}")
print(df_appendix["district"].value_counts())
print(df_appendix.head(10))

