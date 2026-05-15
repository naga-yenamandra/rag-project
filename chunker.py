import json
from pathlib import Path
import tiktoken

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/chunks")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ENCODING_NAME = "cl100k_base"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

encoding = tiktoken.get_encoding(ENCODING_NAME)

def chunk_text(text:str, chunk_size:int, overlap:int) -> list[dict]:
    tokens = encoding.encode(text, disallowed_special=())
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({
            "chunk_index": chunk_index,
            "token_count": len(chunk_tokens),
            "text": chunk_text
        })
        chunk_index +=1
        start += chunk_size - overlap

    return chunks

for txt_path in INPUT_DIR.glob("*.txt"):
    print(f"Chunking {txt_path.name}...")
    text = txt_path.read_text()
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    output_path = OUTPUT_DIR / f"{txt_path.stem}.jsonl"
    with output_path.open("w") as f:
        for chunk in chunks:
            chunk["source"] =txt_path.name
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        #    print(f" Chunk  {chunk}")

    print(f" Saved {len(chunks)} chunks to {output_path.name}")


print("Done!")




