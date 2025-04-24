# ChatBot_BE/document_qa/vector_index.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from document_qa.doc_parser import parse_all_documents
from document_qa.utils import chunk_text

# Load embedding model (small and fast)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Store (text chunks, metadata, embeddings) in memory
text_chunks = []
chunk_sources = []
index = None

def build_index():
    global text_chunks, chunk_sources, index

    text_chunks = []
    chunk_sources = []
    embeddings = []

    docs = parse_all_documents()
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            text_chunks.append(chunk)
            chunk_sources.append(doc["filename"])
            vector = embed_model.encode(chunk)
            embeddings.append(vector)

    embeddings_np = np.vstack(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

def search_index(question, top_k=3):
    global text_chunks, chunk_sources, index

    if index is None:
        build_index()

    query_vec = embed_model.encode([question]).astype("float32")
    D, I = index.search(query_vec, top_k)

    results = []
    for idx in I[0]:
        results.append({
            "chunk": text_chunks[idx],
            "source": chunk_sources[idx]
        })

    return results
