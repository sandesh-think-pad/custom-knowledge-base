from datetime import datetime
from zoneinfo import ZoneInfo

import weaviate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from weaviate.classes.config import DataType, Property
from weaviate.classes.data import DataObject

from config import WEAVIATE_COLLECTION, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT, EMBED_MODEL


class WeaviateIndexer:
    """Creates the Weaviate collection schema and bulk-inserts document nodes with BGE embeddings."""

    def __init__(
        self,
        collection_name: str = WEAVIATE_COLLECTION,
        host: str = WEAVIATE_HOST,
        port: int = WEAVIATE_PORT,
        grpc_port: int = WEAVIATE_GRPC_PORT,
    ):
        self.collection_name = collection_name
        self._client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
        self._embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            query_instruction="Represent this sentence for searching relevant passages: ",
        )

    def create_schema(self):
        if self._client.collections.exists(self.collection_name):
            self._client.collections.delete(self.collection_name)
        # No vectorizer_config — vectors are provided manually at insert time.
        self._client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="department", data_type=DataType.TEXT),
                Property(name="classification", data_type=DataType.TEXT),
                Property(name="updated_at", data_type=DataType.DATE),
            ],
        )

    def index(self, nodes):
        collection = self._client.collections.get(self.collection_name)
        objects = [
            DataObject(
                properties={
                    "text": node.get_text(),
                    "source": "policy_source",
                    "department": "HR",
                    "classification": "Confidential",
                    "updated_at": datetime.now(ZoneInfo("Asia/Kolkata")),
                },
                vector=self._embed_model.get_text_embedding(node.get_text()),
            )
            for node in nodes
        ]
        collection.data.insert_many(objects)
        print(f"Indexed {len(objects)} nodes.")

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
