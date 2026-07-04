import weaviate
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from config import WEAVIATE_COLLECTION, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT


class VectorStoreConnector:
    """Opens a Weaviate connection and builds a LlamaIndex index over an existing collection."""

    def __init__(self, collection_name: str = WEAVIATE_COLLECTION):
        self.collection_name = collection_name
        self._client = None

    def build_index(self) -> VectorStoreIndex:
        self._client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
        )
        vector_store = WeaviateVectorStore(
            weaviate_client=self._client,
            index_name=self.collection_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

    def close(self):
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
