# Embeddings via the Jina AI API (no local torch/model) so the app fits free hosting.

import os
import requests

class JinaEmbeddings:
    # Drop-in replacement for HuggingFaceEmbeddings using Jina's hosted multilingual model.
    def __init__(self, model="jina-embeddings-v3"):
        self.api_key = os.getenv("JINA_API_KEY")
        self.model = model
        self.url = "https://api.jina.ai/v1/embeddings"

    def _embed(self, texts):
        # Send a batch of texts to Jina and return their vectors.
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": [{"text": t} for t in texts]}
        resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]

    def embed_documents(self, texts):
        # Embed a list of documents (batched to stay within API limits).
        vectors = []
        for i in range(0, len(texts), 50):
            vectors.extend(self._embed(texts[i:i + 50]))
        return vectors

    def embed_query(self, text):
        # Embed a single query string.
        return self._embed([text])[0]