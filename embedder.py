import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_DIR = Path("data/chunks")
OUTPUT_DIR = Path("data/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading the model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model Loaded..")

def embed_chunks(jsonl_path: Path) -> tuple[list[dict], np.ndarray]:
    chunks = []
    with jsonl_path.open("r") as f:
            for line in f:
                  chunks.append(json.loads(line))
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return chunks, embeddings

for jsonl_path in INPUT_DIR.glob("*.jsonl"):
      print(f"Embeddings {jsonl_path.name}...")
      chunks, embeddings = embed_chunks(jsonl_path)

      chunks_output = OUTPUT_DIR / f"{jsonl_path.stem}.chunks.jsonl"
      embeddings_output = OUTPUT_DIR / f"{jsonl_path.stem}.embeddings.npy"

      with chunks_output.open("w") as f:
            for chunk in chunks:
                  f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
      
      np.save(embeddings_output, embeddings)

      print(f" Saved {len(chunks)} chunks -> {chunks_output.name}")
      print(f" Saved embeddings shape {embeddings.shape} -> {embeddings_output.name}")

print("\nDone !")
