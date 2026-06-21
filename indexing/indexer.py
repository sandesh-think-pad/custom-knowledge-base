from datetime import datetime
from zoneinfo import ZoneInfo

import weaviate
from weaviate.classes.config import Configure, DataType, Property

from config import WEAVIATE_COLLECTION, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT


class WeaviateIndexer:
    """Creates the Weaviate collection schema and bulk-inserts document nodes."""

    def __init__(
        self,
        collection_name: str = WEAVIATE_COLLECTION,
        host: str = WEAVIATE_HOST,
        port: int = WEAVIATE_PORT,
        grpc_port: int = WEAVIATE_GRPC_PORT,
    ):
        self.collection_name = collection_name
        self._client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)

    def create_schema(self):
        self._client.collections.create(
            name=self.collection_name,
            vector_config=Configure.Vectors.text2vec_model2vec(),
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
            {
                "text": node.get_text(),
                "source": "policy_source",
                "department": "HR",
                "classification": "Confidential",
                "updated_at": datetime.now(ZoneInfo("Asia/Kolkata")),
            }
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
