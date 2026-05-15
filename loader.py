from pathlib import Path
from pypdf import PdfReader

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

for pdf_path in INPUT_DIR.glob("*.pdf"):
    print(f"Processing {pdf_path.name}...")
    text = extract_text_from_pdf(pdf_path)
    output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
    output_path.write_text(text)
    print(f"Saved to directory {output_path}")

print("Done ! ")

