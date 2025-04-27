import os
import torch
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm_hf_inference(model_id, max_new_tokens=128, temperature=0.1):
    return HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation", 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token=os.getenv("HF_TOKEN")
    )

try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer: {e}")
    raise

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

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

def get_answer(question, top_k=3):
    try:
        q_embedding = embedder.encode([question])
        D, I = index.search(np.array(q_embedding), top_k)
        context = "\n".join([chunks[i] for i in I[0] if i < len(chunks)])

        prompt = PromptTemplate.from_template(
            "[INST] You are a helpful AI assistant to respond to event attendee's questions about event details based on the provided context."
            "\nContext:\n{context}\n\n"
            "User: {question}.\n [/INST]"
            "\nAI:"
        )
        hf = get_llm_hf_inference(
            max_new_tokens=256,
            temperature=0.1,
            model_id=model_name
        )

        chain = prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key='content')

        response = chain.invoke({
            "question": question,
            "context": context
        })

        if isinstance(response, str):
            response = response.split("AI:")[-1].strip()

        return response

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Sorry, something went wrong while trying to answer your question. Please try again later."
