# embeddings.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize model and FAISS index
_model = SentenceTransformer("all-MiniLM-L6-v2")
_dimension = _model.get_sentence_embedding_dimension()
_index = faiss.IndexFlatL2(_dimension)
_store = []  # simple in-memory store: [(id, text)]

def get_embeddings(texts):
    """
    Generate embeddings for a list of text chunks.
    Returns a list of numpy arrays.
    """
    return [np.array(e) for e in _model.encode(texts, convert_to_numpy=True)]

def upsert_embeddings(uid, text, embedding):
    """
    Store (id, text, embedding) in FAISS for later retrieval.
    """
    global _index, _store
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
    _index.add(embedding)
    _store.append((uid, text))

def similarity_search(query_embedding, top_k=3):
    """
    Find the most similar chunks to a query embedding.
    Returns a list of (id, score, text).
    """
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    if len(_store) == 0:
        return []

    D, I = _index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        uid, text = _store[idx]
        results.append((uid, float(score), text))
    return results
