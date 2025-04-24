# ChatBot_BE/document_qa/qa_engine.py

import requests
from document_qa.vector_index import search_index
from functools import lru_cache

OLLAMA_API_URL = "http://localhost:11434/api"
MODEL = "llama3"
MAX_CONTEXT_LENGTH = 1000  # characters to send to LLM

@lru_cache(maxsize=1)
def warm_up():
    """Force FAISS to index documents once."""
    from document_qa.vector_index import build_index
    build_index()

def ask_documents(question: str):
    warm_up()  # ensure index is built

    matches = search_index(question, top_k=3)

    # Build limited-length context
    context = ""
    for match in matches:
        if len(context) + len(match["chunk"]) < MAX_CONTEXT_LENGTH:
            context += f"\n\n[From {match['source']}]:\n{match['chunk']}"
        else:
            break

    if not context:
        return "I couldn't find anything useful in the documents."

    prompt = f"""You are an assistant helping with document Q&A.

Context from internal documents:
{context.strip()}

Question: {question}

Give a helpful, clear, and polite answer. Be specific. Avoid vague or generic replies.
Avoid mentioning you are an AI or referring to documents unless necessary.
"""

    response = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3
        }
    )

    if response.status_code != 200:
        return f"Ollama error: {response.status_code}"

    result = response.json()
    return result.get("response", "").strip()
