from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.llms.ollama import Ollama

from config import LLM_MODEL, RERANK_MODEL, OLLAMA_HOST

_QA_PROMPT = PromptTemplate(
    """You are a knowledgeable assistant for the internal knowledge base.
Answer the question using only the context provided below.
If the answer is not clearly present in the context, say so honestly and suggest
the employee contact the relevant team directly.
Always end your answer by citing the source document(s) you used.

Context:
{context_str}

Question: {query_str}

Answer:"""
)


class RAGQueryEngine:
    """Composes retriever, sentence-window replacement, reranker, LLM, and prompt into a query engine."""

    def __init__(self, retriever):
        llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.1,
            context_window=8192,
            request_timeout=120.0,
        )
        self._engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=llm,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                SentenceTransformerRerank(model=RERANK_MODEL, top_n=3),
            ],
            text_qa_template=_QA_PROMPT,
        )

    def query(self, question: str) -> str:
        return self._engine.query(question).response
