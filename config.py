import os

WEAVIATE_COLLECTION = "EnterpriseKB"
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.environ.get("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

KNOWLEDGE_SRC_DIR = os.environ.get("KNOWLEDGE_SRC_DIR", "./../knowledgesrc")

# Ollama host — override via OLLAMA_HOST env var when running inside Docker
# (containers reach the Mac host's Ollama via host.docker.internal)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Single model used for both indexing and querying — they must share the same vector space.
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "llama3.1:latest"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
