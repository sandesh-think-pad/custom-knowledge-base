# Enterprise Knowledge Base — RAG System

An AI-powered knowledge base that lets employees ask questions in plain English and get accurate, cited answers grounded in your company's own documents. Built with [LlamaIndex](https://www.llamaindex.ai/), [Weaviate](https://weaviate.io/), and [Ollama](https://ollama.com/).

---

## Why RAG?

Standard LLMs are static — they cannot answer questions about your company's refund policy, the latest HR guidelines, or a product spec that changed last week. Fine-tuning requires hours of retraining every time documents change.

**Retrieval-Augmented Generation (RAG)** solves this by fetching the relevant documents at query time and feeding them to the LLM as context. Knowledge updates take minutes (re-index the changed documents), answers are grounded in real sources, and every response carries a citation trail.

---

## Architecture

The system runs as two independent pipelines that meet at the vector store.

```
── Indexing Pipeline (run once, then on document changes) ────────────────────

  [PDF / DOCX / Confluence / Notion]
            │
            ▼
  [DocumentLoader]  ──  reads files from knowledgesrc/
            │
            ▼
  [NodeParser]  ──  sentence-window chunking (window = 3 sentences)
            │
            ▼
  [BAAI/bge-large-en-v1.5]  ──  generates embeddings
            │
            ▼
  [WeaviateIndexer]  ──  stores text + vectors + metadata in Weaviate
            │
            ▼
  [Weaviate: EnterpriseKB collection]


── Query Pipeline (per user question) ───────────────────────────────────────

  [User Question]
            │
            ▼
  [BAAI/bge-small-zh-v1.5]  ──  embeds the query
            │
            ▼
  [VectorStoreConnector]  ──  connects to existing Weaviate index
            │
            ▼
  [HybridRetriever]  ──  75% semantic + 25% BM25 keyword, top-10 results
            │            filters: department=HR, classification=confidential
            ▼
  [RAGQueryEngine]
    ├─ MetadataReplacementPostProcessor  ──  expands sentence to window context
    ├─ SentenceTransformerRerank         ──  cross-encoder re-scores, keeps top-3
    └─ Ollama / llama3.1 (temp=0.1)      ──  generates grounded answer + citations
            │
            ▼
  [Answer with source citations]
```

---

## Folder Structure

```
knowledgebase/
│
├── config.py                   # All constants: model names, Weaviate host, collection name
│
├── indexing/                   # One-time ingestion pipeline
│   ├── loader.py               # DocumentLoader  — reads PDFs from knowledgesrc/
│   ├── parser.py               # NodeParser      — sentence-window chunking
│   ├── indexer.py              # WeaviateIndexer — schema creation + bulk insert
│   └── ingest.py               # Entry point: run this to index your documents
│
├── rag_pipeline/               # Per-query retrieval and generation pipeline
│   ├── connector.py            # VectorStoreConnector — opens Weaviate, builds index
│   ├── retriever.py            # HybridRetriever      — semantic + keyword search
│   ├── query_engine.py         # RAGQueryEngine       — reranker + LLM + prompt
│   └── pipeline.py             # KnowledgeBasePipeline.query() + get_data() entry point
│
├── ui/
│   └── app.py                  # Streamlit chat interface
│
├── weaviate/
│   └── docker-compose.yml      # Weaviate + model2vec embedding sidecar
│
├── pyproject.toml              # Project metadata and dependencies
└── requirements.txt            # Pip-installable dependency list
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | `zoneinfo` requires 3.9 |
| Docker + Docker Compose | For Weaviate |
| [Ollama](https://ollama.com/) | Local LLM inference |
| 16 GB RAM / GPU memory | For llama3.1:8b; 80 GB for 70b |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd knowledgebase
pip install -e .
```

### 2. Pull the LLM

```bash
ollama pull llama3.1:latest
```

### 3. Start Weaviate

```bash
docker compose -f weaviate/docker-compose.yml up -d
```

This starts two containers:
- **weaviate** — vector database on port `8080` (HTTP) and `50051` (gRPC)
- **text2vec-model2vec** — lightweight embedding sidecar used during insert

### 4. Add your documents

Place PDF files in a `knowledgesrc/` folder one level above the project root:

```
parent-directory/
├── knowledgesrc/       ← put your PDFs here
│   ├── hr-policy.pdf
│   └── onboarding-guide.pdf
└── knowledgebase/      ← this repo
```

To change the source path, edit `KNOWLEDGE_SRC_DIR` in [config.py](config.py).

### 5. Index your documents

```bash
python -m indexing.ingest
```

Or if installed via `pip install -e .`:

```bash
ingest
```

This will:
1. Load all PDFs from `knowledgesrc/`
2. Parse them into sentence-window nodes
3. Create the `EnterpriseKB` collection in Weaviate
4. Embed and insert all nodes

Re-run this command whenever documents change (only changed files need re-indexing in production).

### 6. Start the chat UI

```bash
streamlit run ui/app.py
```

Open [http://localhost:8501](http://localhost:8501) and ask questions about your documents.

---

## Configuration

All tunable parameters live in [config.py](config.py):

| Constant | Default | Notes |
|---|---|---|
| `WEAVIATE_COLLECTION` | `EnterpriseKB` | Weaviate collection name |
| `WEAVIATE_HOST` | `localhost` | Change for remote deployments |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `KNOWLEDGE_SRC_DIR` | `./../knowledgesrc` | Path to source documents |
| `INDEXING_EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model used during indexing |
| `QUERY_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding model used at query time |
| `LLM_MODEL` | `llama3.1:latest` | Ollama model for answer generation |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for re-scoring |

**Important:** `INDEXING_EMBED_MODEL` and `QUERY_EMBED_MODEL` must operate in the same vector space. Only change them together and re-index if you switch models.

Retrieval parameters (in [retriever.py](rag_pipeline/retriever.py)):

| Parameter | Value | Effect |
|---|---|---|
| `similarity_top_k` | `10` | Candidates retrieved before reranking |
| `alpha` | `0.75` | 75% semantic / 25% BM25 keyword blend |
| `top_n` (reranker) | `3` | Final chunks passed to the LLM |
| `temperature` | `0.1` | Low = factual and conservative |

---

## Key Design Decisions

**Sentence-window chunking** — Documents are indexed at sentence granularity for precise retrieval, but the surrounding 3-sentence window is returned at generation time. This avoids fragmented answers while keeping retrieval surgical. Chunking quality has more impact on system performance than the choice of embedding model or LLM.

**Hybrid search** — Pure semantic search misses exact terms (product IDs, ticket numbers, regulatory citations). The 75/25 semantic/keyword blend handles both natural-language questions and jargon lookups.

**Cross-encoder reranking** — The bi-encoder retrieval model scores query and document independently. The cross-encoder in `SentenceTransformerRerank` evaluates them jointly, catching relevance that vector similarity misses. Applied only to the top-10 candidates to keep latency low (~50–100 ms overhead).

**Local inference** — Ollama runs the LLM entirely on-premise. No document content leaves your infrastructure, which is required for confidential enterprise data.

**Temperature 0.1** — Keeps the model conservative and grounded in retrieved context. Values above 0.4 risk the model interpolating beyond what the documents say.

---

## Evaluation

Use [RAGAS](https://docs.ragas.io/) to measure system quality without manually labelling answers:

```bash
pip install ragas
```

Target metrics:

| Metric | Target | What it measures |
|---|---|---|
| Faithfulness | ≥ 0.90 | Answer is supported by retrieved context |
| Answer Relevancy | ≥ 0.85 | Response addresses the question asked |
| Context Recall | ≥ 0.80 | Required information surfaced by retrieval |
| Context Precision | ≥ 0.75 | Retrieved chunks are mostly useful |

**Diagnosing failures:**
- Low **context recall** → fix chunking or embedding (retrieval is not finding the right content)
- Low **faithfulness** → LLM is hallucinating; tighten the prompt or lower temperature
- Low **context precision** → too much noise in retrieved set; lower `similarity_top_k` or tighten filters

---

## Extending the System

**Connect to Confluence / Notion / SharePoint**

Replace `DocumentLoader` with a LlamaIndex connector:

```python
from llama_index.readers.confluence import ConfluenceReader

reader = ConfluenceReader(
    base_url="https://yourcompany.atlassian.net/wiki",
    oauth2={"client_id": "...", "token": "..."}
)
documents = reader.load_data(space_key="HR")
```

**Add more departments**

Update the metadata fields and filter conditions in [indexer.py](indexing/indexer.py) and [retriever.py](rag_pipeline/retriever.py). Weaviate's multi-tenancy can isolate data per department with native access control.

**Switch to Weaviate Cloud**

In [connector.py](rag_pipeline/connector.py), replace:

```python
client = weaviate.connect_to_local()
```

with:

```python
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=weaviate.AuthApiKey("your-api-key")
)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No results returned | Weaviate not running | `docker compose -f weaviate/docker-compose.yml up -d` |
| Wrong or outdated answers | Documents not re-indexed after change | Re-run `python -m indexing.ingest` |
| LLM timeout | Model too large for available memory | Switch to a smaller Ollama model in `config.py` |
| Answers ignore middle context | Lost-in-middle problem | Reduce `similarity_top_k`; verify reranker ordering |
| Collection already exists error | Re-running ingest on existing data | Drop the collection in Weaviate before re-indexing |
