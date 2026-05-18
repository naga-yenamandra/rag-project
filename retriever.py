import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = Path("data/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

print(f"Loading Model {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("Model Loaded..")

print("Loading chunks and embeddings ...")

all_chunks = []
all_embeddings = []

for chunks_file in sorted(EMBEDDINGS_DIR.glob("*.chunks.jsonl")):
    stem = chunks_file.name.replace(".chunks.jsonl", "")
    embeddings_file = EMBEDDINGS_DIR / f"{stem}.embeddings.npy"

    with chunks_file.open("r") as f:
        for line in f:
            all_chunks.append(json.loads(line))
    
    embeddings = np.load(embeddings_file)
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)

print(f"Loaded {len(all_chunks)} chunks with embeddings shape {all_embeddings.shape}")

#print("\n--- Sample chunk index (0)---")
#print(all_chunks[0])
#print("\n--- First 200 chars of chunk text ---")
#print(all_chunks[0]["text"][:200])
#print("\n--- Sample embeddings index (0) ---")
#print(f"Shape: {all_embeddings[0].shape}")
#print(f"Dtype: {all_embeddings[0].dtype}")
#print(f"First 10 values: {all_embeddings[0][:10]}")
#print(f"Min/Max: {all_embeddings[0].min():.4f} / {all_embeddings[0].max():.4f}")

def retrieve(query: str, top_k: int = TOP_K) -> list[tuple[float, dict]]:
    query_embedding = model.encode([query])[0]

    similarities = all_embeddings @ query_embedding / (
        np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [(float(similarities[i]), all_chunks[i]) for i in top_indices]

    return results

query = "What is attention in transformers?"
results = retrieve(query)

print(f"\nQuery: {query}\n")

for score, chunk in results:
    print(f"Score: {score:.4f} | Source: {chunk['source']}")
    print(f"{chunk['text'][:200]}...")
    print()




