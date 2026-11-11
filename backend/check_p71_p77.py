from pypdf import PdfReader

reader = PdfReader(r"data\reference\Human_Elephant_Conflict_Westbengal_2000_2025.pdf")

for p in range(70, 78):
    text = reader.pages[p].extract_text() or ""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    print(f"Page {p+1}: {lines[:5]}")
    # Also check if any header describes this table
    print(f"  Header/Title on p{p+1}: {lines[:2]}")

