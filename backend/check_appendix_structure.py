from pypdf import PdfReader

reader = PdfReader(r"data\reference\Human_Elephant_Conflict_Westbengal_2000_2025.pdf")

for p in range(60, 96):
    text = reader.pages[p].extract_text() or ""
    first_few = [line.strip() for line in text.split("\n")[:8] if line.strip()]
    print(f"Page {p+1}: {first_few}")

