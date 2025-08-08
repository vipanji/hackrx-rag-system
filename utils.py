import os
import tempfile
import requests
import mimetypes
import pathlib
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

try:
    import docx  # python-docx
except ImportError:
    docx = None  # DOCX extraction will be skipped if not available

# --- Load lightweight transformer once -----------------
print("Loading embedding model (all-MiniLM-L6-v2)...")
_ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded.")

# -------------------------------------------------------
# 1. File Handling
# -------------------------------------------------------

def download_file(url: str) -> str:
    """Download file from URL to a temp file and return the path."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    suffix = pathlib.Path(url.split("?")[0]).suffix or ".tmp"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(filter(None, (page.extract_text() for page in reader.pages)))

def load_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())

def extract_text(path: str) -> str:
    """Detect file type and extract text accordingly."""
    mtype = mimetypes.guess_type(path)[0] or ""
    if path.lower().endswith(".pdf") or "pdf" in mtype:
        return load_pdf(path)
    if path.lower().endswith(".docx") or "word" in mtype:
        return load_docx(path)
    raise ValueError(f"Unsupported file type for {path}")

# -------------------------------------------------------
# 2. Chunking and Embedding
# -------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better context retention."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed multiple texts into float32 vectors."""
    emb = _ST_MODEL.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")

def embed_single(text: str) -> List[float]:
    """Embed a single text (used for single question embedding)."""
    emb = _ST_MODEL.encode([text], normalize_embeddings=True)
    return emb[0].tolist()

# -------------------------------------------------------
# 3. FAISS Indexing
# -------------------------------------------------------

def build_faiss(embeddings: np.ndarray):
    """Create FAISS index using cosine similarity (IP with normalized embeddings)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine sim
    faiss.normalize_L2(embeddings)  # Normalize for cosine distance
    index.add(embeddings)
    return index
