from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import hashlib
from dotenv import load_dotenv
from functools import lru_cache
from groq import Groq

from utils import (
    download_file, extract_text, chunk_text,
    embed_texts, embed_single, build_faiss
)

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in .env")

# Initialize Groq Client once
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize FastAPI
app = FastAPI(title="HackRx RAG System")

# ---------- Pydantic Schemas ----------
class HackRxRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ---------- Utility Functions ----------
def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

@lru_cache(maxsize=10)
def process_document(url: str):
    local_path = download_file(url)
    full_text = extract_text(local_path)
    chunks = chunk_text(full_text, chunk_size=600, overlap=75)
    embeddings = embed_texts(chunks)
    index = build_faiss(embeddings)
    return chunks, index

# ---------- Main Endpoint ----------
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_submission(payload: HackRxRequest):
    try:
        # 1) Process document (cached if repeated)
        chunks, index = process_document(payload.documents)

        # 2) Batch embed questions
        question_embeddings = embed_texts(payload.questions)

        answers = []

        # 3) Loop through questions and generate answers
        for i, q in enumerate(payload.questions):
            q_emb = np.array([question_embeddings[i]], dtype="float32")
            _, idx = index.search(q_emb, k=3)

            # Join top relevant chunks
            context = " ".join([chunks[j] for j in idx[0]])

            # Prepare prompt
            prompt = (
                "You are an insurance policy assistant. "
                "Answer ONLY using the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\nAnswer:"
            )

            # Get completion
            resp = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400
            )
            answer = resp.choices[0].message.content.strip()
            answers.append(answer)

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
