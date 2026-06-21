from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from weaviate.classes.query import Filter


class HybridRetriever:
    """Configures hybrid vector + keyword retrieval with department and classification filters."""

    def __init__(self, index: VectorStoreIndex):
        self._retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
            vector_store_query_mode="hybrid",
            alpha=0.75,  # 75% semantic, 25% keyword
            search_kwargs={
                "where_filter": (
                    Filter.by_property("department").equal("HR")
                    & Filter.by_property("classification").equal("confidential")
                )
            },
        )

    @property
    def retriever(self):
        return self._retriever
