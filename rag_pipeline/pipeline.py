from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from config import QUERY_EMBED_MODEL, LLM_MODEL, OLLAMA_HOST
from .connector import VectorStoreConnector
from .retriever import HybridRetriever
from .query_engine import RAGQueryEngine


class KnowledgeBasePipeline:
    """Orchestrates the full RAG pipeline: retrieval, reranking, and answer generation."""

    def __init__(self):
        Settings.embed_model = HuggingFaceEmbedding(model_name=QUERY_EMBED_MODEL)
        Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_HOST, request_timeout=360.0)

    def query(self, question: str) -> str:
        with VectorStoreConnector() as connector:
            index = connector.build_index()
            retriever = HybridRetriever(index).retriever
            return RAGQueryEngine(retriever).query(question)


def get_data(query: str) -> str:
    return KnowledgeBasePipeline().query(query)
