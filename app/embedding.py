from sentence_transformers import SentenceTransformer
import faiss
import os


embedder = SentenceTransformer('all-MiniLM-L6-v2')


base_dir = os.path.dirname(__file__)
markdown_file_path = os.path.join(base_dir, "info.md")
faiss_file_path = os.path.join(base_dir, "event_index.faiss")
chunks_file_path = os.path.join(base_dir, "event_chunks.txt")


with open(markdown_file_path, "r", encoding="utf-8") as f:
    data = f.read()

chunks = data.split("\n\n") 
chunk_embeddings = embedder.encode(chunks)


dimension = chunk_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

faiss.write_index(index, faiss_file_path)
with open(chunks_file_path, "w", encoding="utf-8") as f:
    f.write("\n---\n".join(chunks))