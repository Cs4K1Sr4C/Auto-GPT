"""Base class for memory providers."""
import abc
from typing import List, Dict
import openai
from config import AbstractSingleton, Config
import numpy as np

cfg = Config()


def get_ada_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    if cfg.use_azure:
        return openai.Embedding.create(input=[text], engine=cfg.azure_embeddings_deployment_id, model="text-embedding-ada-002")["data"][0]["embedding"]
    else:
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data: Dict) -> bool:
        pass

    @abc.abstractmethod
    def get(self, data: Dict) -> Dict:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass

    @abc.abstractmethod
    def get_relevant(self, data: Dict, num_relevant: int = 5) -> List[Dict]:
        pass

    @abc.abstractmethod
    def get_stats(self) -> Dict:
        pass
    
    @abc.abstractmethod
    def remove(self, data: Dict) -> bool:
        pass
    
    @abc.abstractmethod
    def get_all(self) -> List[Dict]:
        pass
    
    @abc.abstractmethod
    def get_size(self) -> int:
        pass

    @staticmethod
    def get_similarity_score(self, emb1, emb2):
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return score

    @staticmethod
    def get_embeddings(text_list: List[str]) -> List[List[float]]:
        embeddings = []
        for text in text_list:
            embeddings.append(get_ada_embedding(text))
        return embeddings

    @staticmethod
    def get_most_similar(embedding: List[float], embeddings: List[List[float]], num_similar: int) -> List[int]:
        similarities = [MemoryProviderSingleton.get_similarity_score(embedding, emb) for emb in embeddings]
        return [idx for idx in sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:num_similar]]

    @staticmethod
    def get_avg_embedding(embeddings: List[List[float]]) -> List[float]:
        return [sum(x)/len(x) for x in zip(*embeddings)]

    @staticmethod
    def get_text_from_embeddings(embeddings: List[List[float]]) -> List[str]:
        return [openai.Completion.create(model="text-davinci-002", prompt=f"embeddings_to_text({e})", max_tokens=60)["choices"][0]["text"] for e in embeddings]
