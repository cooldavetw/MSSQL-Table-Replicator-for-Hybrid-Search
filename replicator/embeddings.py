from __future__ import annotations

from openai import OpenAI

from .config import EmbeddingConfig


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.config.model_name,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        for vector in vectors:
            if len(vector) != self.config.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: configured {self.config.dimensions}, "
                    f"but model '{self.config.model_name}' returned {len(vector)}. "
                    "Update the embedding dimensions setting and recreate the target table."
                )
        return vectors
