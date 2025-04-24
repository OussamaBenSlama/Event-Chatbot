import os
import torch
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Safe model loading
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer: {e}")
    raise

model_name = "google/flan-t5-small"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Optional: Compile model for optimized inference (Torch 2.0+)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
except Exception as e:
    logger.error(f"Failed to load model {model_name}: {e}")
    raise

# Load FAISS index and text chunks
try:
    base_dir = os.path.dirname(__file__)
    faiss_file_path = os.path.join(base_dir, "event_index.faiss")
    chunks_file_path = os.path.join(base_dir, "event_chunks.txt")

    index = faiss.read_index(faiss_file_path)

    with open(chunks_file_path, "r", encoding="utf-8") as f:
        chunks = f.read().split("\n---\n")

    logger.info("FAISS index and chunks loaded successfully.")
except FileNotFoundError as fnf_error:
    logger.error(f"Required file missing: {fnf_error}")
    raise
except Exception as e:
    logger.error(f"Error loading resources: {e}")
    raise

# Main answer function
def get_answer(question, top_k=3):
    try:
        q_embedding = embedder.encode([question])
        D, I = index.search(np.array(q_embedding), top_k)
        context = "\n".join([chunks[i] for i in I[0] if i < len(chunks)])

        prompt = (
            "You are a helpful and friendly assistant responding to questions from event attendees.\n\n"
            "Using the information provided in the context below, answer the following question with a clear, natural, and human-like response. "
            "Begin with a brief introduction if appropriate, and make the answer easy to understand.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_length=100)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Sorry, something went wrong while trying to answer your question. Please try again later."

