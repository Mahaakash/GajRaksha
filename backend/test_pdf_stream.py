import re
import zlib

pdf_path = r"data\reference\Human_Elephant_Conflict_Westbengal_2000_2025.pdf"

with open(pdf_path, 'rb') as f:
    data = f.read()

print("File size:", len(data))
# Check if pure text or compressed streams
streams = re.findall(b'stream[\r\n]+(.*?)[\r\n]+endstream', data, re.DOTALL)
print("Total streams found:", len(streams))

decompressed_texts = []
for i, s in enumerate(streams):
    try:
        decomp = zlib.decompress(s)
        # Check if text in decompressed
        # TJ or Tj operators in PDF
        texts = re.findall(rb'\((.*?)\)\s*T[jJ]', decomp)
        if texts:
            line = " ".join([t.decode('latin1', errors='ignore') for t in texts])
            if any(term in line.lower() for term in ['jhargram', 'bankura', 'purulia', 'medinipur', 'hotspot']):
                decompressed_texts.append((i, line[:200]))
    except Exception:
        pass

print(f"Streams with keywords found: {len(decompressed_texts)}")
for idx, sample in decompressed_texts[:10]:
    print(f"Stream {idx}: {sample}")

