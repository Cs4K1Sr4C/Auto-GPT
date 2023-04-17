
import pinecone

from memory.base import MemoryProviderSingleton, get_ada_embedding


class PineconeMemory(MemoryProviderSingleton):
    def __init__(self, cfg):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "auto-gpt"
        self.vec_num = 0
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

    def add(self, data):
        vector = get_ada_embedding(data)
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        self.vec_num += 1
        return f"Inserting data into memory at index: {self.vec_num-1}:\n data: {data}"

    def update(self, index, new_data):
        old_data = self.get_data(index)
        if old_data:
            self.index.delete(ids=[str(index)])
            self.add(new_data)
            return f"Data at index {index} updated from {old_data} to {new_data}."
        else:
            return f"No data found at index {index}."

    def delete(self, index):
        data = self.get_data(index)
        if data:
            self.index.delete(ids=[str(index)])
            return f"Data at index {index} ({data}) deleted from memory."
        else:
            return f"No data found at index {index}."

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        query_embedding = get_ada_embedding(data)
        results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score)
        return [str(item['metadata']["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()

    def get_data(self, index):
        result = self.index.query(ids=[str(index)], include_metadata=True)
        if len(result.matches) > 0:
            return result.matches[0]['metadata']["raw_text"]
        else:
            return None

