# ADR 002: Technology Stack Selection

**Status:** Proposed  
**Date:** 2025-10-26  
**Deciders:** Development Team  
**Related ADRs:** [ADR 001: Modular Architecture](./001-modular-architecture-with-rag-and-evaluator.md)

## Context and Problem Statement

We need to select appropriate technologies for each component of the career chatbot system. The choices must balance:
- **Cost efficiency** (API costs, infrastructure)
- **Performance** (latency, throughput)
- **Developer experience** (ease of use, documentation)
- **Production readiness** (stability, community support)
- **Deployment constraints** (HuggingFace Spaces compatibility)

## Decision Drivers

- **Budget**: Minimize API costs while maintaining quality
- **Speed**: Response time < 3 seconds target
- **Simplicity**: Prefer embedded solutions over separate services
- **Compatibility**: Must work on HuggingFace Spaces (CPU-only)
- **Maintainability**: Well-documented, active community
- **Type Safety**: Strong typing for reliability
- **Testing**: Easy to mock and test

## Technology Stack Decisions

### 1. Primary Language Model

**Decision:** DeepSeek Chat

**Alternatives Considered:**
- OpenAI GPT-4o-mini
- Anthropic Claude 3.5 Sonnet
- Google Gemini 2.0 Flash
- Local models (Llama, Mistral)

**Comparison:**

| Model | Input Cost | Output Cost | Quality | Speed | Function Calling |
|-------|-----------|-------------|---------|-------|------------------|
| **DeepSeek** | **$0.14/M** | **$0.28/M** | High | Fast | ✅ Excellent |
| GPT-4o-mini | $0.15/M | $0.60/M | High | Fast | ✅ Excellent |
| Claude Sonnet | $3.00/M | $15.00/M | Highest | Medium | ✅ Excellent |
| Gemini Flash | $0.075/M | $0.30/M | Good | Very Fast | ✅ Good |
| Local (Llama 3.2) | Free | Free | Medium | Slow (CPU) | ⚠️ Limited |

**Rationale:**
- ✅ **Cost-effective**: ~50% cheaper output than GPT-4o-mini
- ✅ **Quality**: Comparable to GPT-4o-mini in benchmarks
- ✅ **Function calling**: Excellent support for tool use
- ✅ **API compatibility**: OpenAI-compatible API (easy migration)
- ✅ **Context window**: 64K tokens sufficient for career conversations

**Trade-offs:**
- ⚠️ Less established than OpenAI (newer provider)
- ⚠️ Smaller community compared to OpenAI/Anthropic

---

### 2. Evaluator Model

**Decision:** Google Gemini 2.0 Flash

**Alternatives Considered:**
- DeepSeek (same as primary)
- OpenAI GPT-4o-mini
- Anthropic Claude Haiku

**Comparison:**

| Model | Cost | Speed | Structured Output | Quality |
|-------|------|-------|-------------------|---------|
| **Gemini Flash** | **$0.075/$0.30** | **Very Fast** | ✅ Excellent | Good |
| DeepSeek | $0.14/$0.28 | Fast | ✅ Good | High |
| GPT-4o-mini | $0.15/$0.60 | Fast | ✅ Excellent | High |
| Claude Haiku | $0.80/$4.00 | Very Fast | ✅ Good | Good |

**Rationale:**
- ✅ **Fastest**: Critical for evaluation to not slow down responses
- ✅ **Cheapest**: Lowest cost per evaluation
- ✅ **Structured output**: Native support for Pydantic models
- ✅ **Separation**: Different model provides diverse perspective
- ✅ **Good enough**: Evaluation doesn't need highest quality, just consistency

**Trade-offs:**
- ⚠️ Slightly lower quality than GPT-4o-mini or Claude
- ✅ But speed and cost matter more for evaluation use case

---

### 3. Vector Database (RAG)

**Decision:** Chroma DB

**Alternatives Considered:**
- Pinecone (cloud)
- Weaviate (self-hosted)
- FAISS (in-memory)
- Qdrant (self-hosted)
- LanceDB (embedded)
- PGVector (PostgreSQL extension)
- Milvus (distributed)

**Comparison:**

| Solution | Type | Setup | Cost | HF Spaces | Persistence | Scale | Performance |
|----------|------|-------|------|-----------|-------------|-------|-------------|
| **Chroma** | **Embedded** | **None** | **Free** | ✅ | ✅ Disk | Small-Med | Good |
| LanceDB | Embedded | None | Free | ✅ | ✅ Disk | Medium | Excellent |
| FAISS | In-memory | None | Free | ✅ | ❌ RAM only | Large | Excellent |
| PGVector | Extension | PostgreSQL | Free | ⚠️ Complex | ✅ DB | Medium | Good |
| Qdrant | Self-hosted | Docker | Free | ⚠️ Complex | ✅ Disk | Large | Excellent |
| Milvus | Distributed | K8s/Docker | Free | ❌ | ✅ Cluster | Very Large | Excellent |
| Pinecone | Cloud | API key | $70+/mo | ⚠️ Network | ✅ Cloud | Large | Excellent |
| Weaviate | Self-hosted | Docker | Free | ❌ | ✅ Disk | Large | Excellent |

**Detailed Analysis:**

#### Chroma (Chosen)
**Pros:**
- ✅ **Zero setup**: Python package, no configuration
- ✅ **Persistent**: Saves to disk automatically
- ✅ **LangChain native**: First-class integration
- ✅ **HF Spaces**: Works perfectly on CPU
- ✅ **Simple API**: Easy to use
- ✅ **Sufficient**: Handles 10K-100K vectors easily

**Cons:**
- ⚠️ Not optimized for millions of vectors
- ⚠️ Single-node only
- ⚠️ Limited filtering capabilities

**Best for:** Small to medium knowledge bases, prototypes, HuggingFace deployments

#### LanceDB
**Pros:**
- ✅ **Fast**: Columnar storage, excellent performance
- ✅ **Embedded**: No server needed
- ✅ **Scalable**: Handles larger datasets than Chroma
- ✅ **Modern**: Built on Apache Arrow
- ✅ **Versioning**: Built-in data versioning

**Cons:**
- ⚠️ Newer (less mature than Chroma)
- ⚠️ Smaller community
- ⚠️ Less LangChain integration

**Best for:** Medium datasets, performance-critical applications, when you need versioning

#### PGVector
**Pros:**
- ✅ **PostgreSQL native**: Use existing PostgreSQL skills
- ✅ **ACID compliance**: Full transactional support
- ✅ **Mature**: Built on PostgreSQL
- ✅ **SQL queries**: Can combine with regular SQL

**Cons:**
- ⚠️ Requires PostgreSQL setup
- ⚠️ Not suitable for HuggingFace Spaces (needs server)
- ⚠️ Slower than specialized vector DBs
- ⚠️ Limited to ~1M vectors efficiently

**Best for:** When you already use PostgreSQL, need ACID transactions, complex queries

#### Milvus
**Pros:**
- ✅ **Highly scalable**: Billions of vectors
- ✅ **Fast**: Optimized for large-scale
- ✅ **Feature-rich**: Advanced filtering, hybrid search
- ✅ **Production-ready**: Used by major companies

**Cons:**
- ⚠️ Complex setup (Kubernetes/Docker)
- ⚠️ Overkill for small projects
- ⚠️ Not suitable for HuggingFace Spaces
- ⚠️ Requires significant resources

**Best for:** Large-scale production systems, millions of vectors, enterprise applications

**Rationale for Chroma:**
- ✅ **Perfect for MVP**: Zero configuration, works immediately
- ✅ **HF Spaces compatible**: Embedded, no server needed
- ✅ **Sufficient scale**: Handles career chatbot knowledge base easily
- ✅ **Easy migration**: Can switch to LanceDB or Milvus later if needed
- ✅ **LangChain integration**: Mature, well-tested

**Migration Path:**
```
Chroma (0-100K vectors) → LanceDB (100K-1M) → Milvus (1M+)
```

**Trade-offs:**
- ⚠️ Not suitable for very large datasets (millions of vectors)
- ⚠️ Single-node only (but we don't need distributed)
- ✅ But perfect for our use case and easy to migrate later

---

### 4. SQL Database

**Decision:** SQLite

**Alternatives Considered:**
- PostgreSQL
- MySQL
- DuckDB

**Comparison:**

| Database | Type | Setup | HF Spaces | Features | Use Case |
|----------|------|-------|-----------|----------|----------|
| **SQLite** | **Embedded** | **None** | ✅ | Basic | **Analytics** |
| PostgreSQL | Server | Docker | ❌ | Advanced | Production |
| MySQL | Server | Docker | ❌ | Advanced | Production |
| DuckDB | Embedded | None | ✅ | Analytics | Analytics |

**Rationale:**
- ✅ **Zero configuration**: File-based, no server
- ✅ **Sufficient**: Handles thousands of conversations easily
- ✅ **Reliable**: Battle-tested, stable
- ✅ **Portable**: Single file, easy backup
- ✅ **SQLAlchemy support**: Excellent ORM integration
- ✅ **HF Spaces**: Works perfectly in constrained environment

**Trade-offs:**
- ⚠️ Not suitable for high concurrency (but we have single user at a time)
- ⚠️ Limited analytics features (but DuckDB could be added later)

---

### 5. Embedding Model

**Decision:** sentence-transformers/all-MiniLM-L6-v2

**Alternatives Considered:**
- OpenAI text-embedding-3-small
- Google Gemini text-embedding-004
- Cohere embed-english-v3.0
- Voyage AI voyage-2
- sentence-transformers/all-mpnet-base-v2
- BGE-small-en-v1.5
- E5-small-v2

**Comparison:**

| Model | Provider | Size/Type | Speed | Quality | Cost | Offline | Dimensions |
|-------|----------|-----------|-------|---------|------|---------|------------|
| **all-MiniLM-L6-v2** | **HuggingFace** | **80MB** | **Fast** | Good | **Free** | ✅ | 384 |
| all-mpnet-base-v2 | HuggingFace | 420MB | Medium | Better | Free | ✅ | 768 |
| BGE-small-en-v1.5 | HuggingFace | 130MB | Fast | Good | Free | ✅ | 384 |
| E5-small-v2 | HuggingFace | 130MB | Fast | Good | Free | ✅ | 384 |
| OpenAI ada-002 | OpenAI | API | Fast | Excellent | $0.02/M | ❌ | 1536 |
| OpenAI text-3-small | OpenAI | API | Fast | Excellent | $0.02/M | ❌ | 1536 |
| Gemini text-004 | Google | API | Very Fast | Excellent | **$0.00125/M** | ❌ | 768 |
| Cohere v3 | Cohere | API | Fast | Excellent | $0.10/M | ❌ | 1024 |
| Voyage AI v2 | Voyage | API | Fast | Best | $0.12/M | ❌ | 1024 |

**Detailed Analysis:**

#### sentence-transformers/all-MiniLM-L6-v2 (Chosen)
**Pros:**
- ✅ **Free**: No API costs
- ✅ **Fast**: 80MB model, quick inference on CPU
- ✅ **Offline**: Works without internet
- ✅ **Sufficient quality**: 384 dimensions, good for career docs
- ✅ **Popular**: 50M+ downloads, well-tested
- ✅ **HF Spaces**: Runs perfectly on CPU
- ✅ **No rate limits**: Unlimited usage

**Cons:**
- ⚠️ Lower quality than API-based embeddings
- ⚠️ Smaller dimension space (384 vs 768/1536)

**Best for:** Cost-sensitive projects, offline deployments, HuggingFace Spaces

#### Google Gemini text-embedding-004
**Pros:**
- ✅ **Extremely cheap**: $0.00125/M tokens (16x cheaper than OpenAI!)
- ✅ **Fast**: Very low latency
- ✅ **High quality**: Competitive with OpenAI
- ✅ **768 dimensions**: Good balance
- ✅ **Large context**: 2048 tokens per embedding

**Cons:**
- ⚠️ Requires internet connection
- ⚠️ API dependency
- ⚠️ Rate limits (though generous)

**Best for:** When quality matters, cost-conscious API usage, production systems

**Cost Comparison (10K documents, 500 tokens each):**
```
Gemini:     $0.00625  (5M tokens × $0.00125/M)
OpenAI:     $0.10     (5M tokens × $0.02/M)
Local:      $0.00     (Free)
```

#### OpenAI text-embedding-3-small
**Pros:**
- ✅ **Excellent quality**: State-of-the-art
- ✅ **1536 dimensions**: Rich representation
- ✅ **Fast**: Low latency
- ✅ **Reliable**: Proven at scale

**Cons:**
- ⚠️ $0.02/M tokens (16x more than Gemini)
- ⚠️ API dependency

**Best for:** When quality is paramount, budget allows

#### all-mpnet-base-v2
**Pros:**
- ✅ **Better quality**: Higher than MiniLM
- ✅ **768 dimensions**: Richer representation
- ✅ **Free & offline**

**Cons:**
- ⚠️ Slower: 420MB model, 5x larger
- ⚠️ More memory usage

**Best for:** When you need better quality but want to stay offline

**Rationale for all-MiniLM-L6-v2:**
- ✅ **Perfect for HF Spaces**: Small, fast, CPU-friendly
- ✅ **Zero cost**: Important for MVP and demos
- ✅ **Sufficient quality**: Career docs don't need SOTA embeddings
- ✅ **Offline**: No API dependencies, no rate limits
- ✅ **Easy upgrade path**: Can switch to Gemini API later if needed

**Upgrade Path:**
```
Phase 1: all-MiniLM-L6-v2 (MVP, free)
Phase 2: Gemini text-embedding-004 (production, cheap API)
Phase 3: OpenAI text-3-small (if quality critical)
```

**When to Upgrade:**
- Knowledge base > 10K documents
- Quality issues with retrieval
- Moving off HuggingFace Spaces
- Budget allows API costs

**Trade-offs:**
- ⚠️ Lower quality than API embeddings (but sufficient for use case)
- ⚠️ Smaller dimension space (but faster and free)
- ✅ Perfect balance for MVP and HuggingFace deployment

---

### 6. Memory Management System

**Decision:** Hybrid Multi-Layer Memory (Custom Implementation)

**Alternatives Considered:**
- LangChain Memory classes
- Mem0 (memory management service)
- Zep (long-term memory store)
- Custom implementation (chosen)

**Comparison:**

| Solution | Type | Setup | Cost | Features | Control | HF Spaces |
|----------|------|-------|------|----------|---------|-----------|
| **Custom** | **Library** | **Code** | **Free** | **Tailored** | **Full** | ✅ |
| LangChain Memory | Library | Import | Free | Basic | Medium | ✅ |
| Mem0 | Service | API | $29+/mo | Advanced | Low | ⚠️ API |
| Zep | Self-hosted | Docker | Free | Advanced | Medium | ❌ |

**Detailed Analysis:**

#### Custom Implementation (Chosen)
**Components:**
1. **Working Memory**: Python deque + token counting
2. **Episodic Memory**: Chroma vector store (reuse RAG infrastructure)
3. **Semantic Memory**: SQLite table + JSON fields
4. **Procedural Memory**: SQLite analytics queries

**Pros:**
- ✅ **Full control**: Customize for our exact needs
- ✅ **No dependencies**: Uses existing infrastructure (Chroma + SQLite)
- ✅ **Cost-effective**: No additional services
- ✅ **Integrated**: Seamless with our architecture
- ✅ **Lightweight**: Minimal overhead
- ✅ **HF Spaces**: Works perfectly

**Cons:**
- ⚠️ More code to maintain
- ⚠️ Need to implement ourselves

**Implementation:**
```python
# Working Memory: Simple deque
from collections import deque
working_memory = deque(maxlen=20)  # Last 10 turns

# Episodic Memory: Reuse Chroma
episodic_store = chroma_client.get_or_create_collection("conversations")

# Semantic Memory: SQLite table
CREATE TABLE semantic_facts (
    fact_type TEXT,
    fact_key TEXT,
    fact_value TEXT,
    confidence FLOAT
);

# Procedural Memory: Analytics queries
SELECT question_type, COUNT(*) FROM conversations GROUP BY question_type;
```

#### LangChain Memory
**Types Available:**
- `ConversationBufferMemory`: Simple list
- `ConversationSummaryMemory`: LLM-based summarization
- `ConversationBufferWindowMemory`: Sliding window
- `VectorStoreRetrieverMemory`: Vector-based retrieval

**Pros:**
- ✅ **Quick start**: Pre-built classes
- ✅ **Battle-tested**: Used in production
- ✅ **Documentation**: Well-documented

**Cons:**
- ⚠️ **Limited flexibility**: Hard to customize
- ⚠️ **Not comprehensive**: Missing semantic/procedural layers
- ⚠️ **Overhead**: Brings LangChain dependencies
- ⚠️ **Breaking changes**: LangChain updates frequently

**Best for:** Quick prototypes, when LangChain already used heavily

#### Mem0
**Features:**
- Managed memory service
- Multi-user support
- Automatic fact extraction
- Graph-based memory
- API-based

**Pros:**
- ✅ **Fully managed**: No infrastructure
- ✅ **Advanced features**: Graph memory, auto-extraction
- ✅ **Multi-user**: Built-in user separation

**Cons:**
- ⚠️ **Cost**: $29/month minimum
- ⚠️ **API dependency**: Requires internet
- ⚠️ **Less control**: Black box
- ⚠️ **Overkill**: For single-user chatbot

**Best for:** Multi-user production apps, when budget allows, enterprise

#### Zep
**Features:**
- Open-source memory store
- Long-term memory
- Fact extraction
- Summary generation
- Self-hosted

**Pros:**
- ✅ **Feature-rich**: Comprehensive memory system
- ✅ **Open source**: Free to use
- ✅ **Active development**: Regular updates

**Cons:**
- ⚠️ **Complex setup**: Requires Docker/server
- ⚠️ **Not HF Spaces compatible**: Needs separate service
- ⚠️ **Overkill**: For our use case

**Best for:** Complex multi-agent systems, when you need all features

**Rationale for Custom Implementation:**
- ✅ **Perfect fit**: Exactly what we need, nothing more
- ✅ **Reuses infrastructure**: Chroma + SQLite already there
- ✅ **Cost-effective**: Zero additional cost
- ✅ **HF Spaces**: Works without external services
- ✅ **Learning**: Understand memory management deeply
- ✅ **Control**: Can optimize for our specific use case

**Memory Storage Strategy:**

| Memory Layer | Storage | Retrieval | Persistence |
|--------------|---------|-----------|-------------|
| Working | Python deque | Direct access | In-memory |
| Episodic | Chroma | Vector search | Disk |
| Semantic | SQLite | SQL queries | Disk |
| Procedural | SQLite | Aggregation | Disk |

**Trade-offs:**
- ⚠️ More code to write and maintain
- ⚠️ Need to handle edge cases ourselves
- ✅ But perfect control and zero cost
- ✅ Easy to understand and debug

---

### 7. ORM (Object-Relational Mapping)

**Decision:** SQLAlchemy

**Alternatives Considered:**
- Peewee
- Django ORM
- Raw SQL

**Rationale:**
- ✅ **Industry standard**: Most popular Python ORM
- ✅ **Feature-rich**: Migrations, relationships, complex queries
- ✅ **Well-documented**: Extensive docs and community
- ✅ **Type hints**: Good typing support
- ✅ **Database agnostic**: Easy to switch from SQLite to PostgreSQL later

**Trade-offs:**
- ⚠️ Steeper learning curve than simpler ORMs
- ✅ But worth it for production readiness

---

### 7. Validation & Configuration

**Decision:** Pydantic (v2) + pydantic-settings

**Alternatives Considered:**
- dataclasses + python-dotenv
- attrs
- marshmallow

**Rationale:**
- ✅ **Type safety**: Runtime validation
- ✅ **Developer experience**: Excellent error messages
- ✅ **JSON schema**: Auto-generation for API docs
- ✅ **Settings management**: Built-in .env support
- ✅ **OpenAI integration**: Native support in OpenAI SDK
- ✅ **Performance**: Fast validation with Rust backend

**Trade-offs:**
- None significant - Pydantic is the clear winner

---

### 8. Web Framework

**Decision:** Gradio

**Alternatives Considered:**
- Streamlit
- FastAPI + React
- Flask + HTML

**Comparison:**

| Framework | Setup Time | HF Integration | Features | Learning Curve |
|-----------|-----------|----------------|----------|----------------|
| **Gradio** | **Minutes** | ✅ **Native** | Chat UI | Low |
| Streamlit | Minutes | ✅ Good | General | Low |
| FastAPI | Hours | ⚠️ Manual | Full control | Medium |
| Flask | Hours | ⚠️ Manual | Full control | Medium |

**Rationale:**
- ✅ **HuggingFace native**: `gradio deploy` command
- ✅ **Chat interface**: Built-in ChatInterface component
- ✅ **Fast development**: Minimal code for UI
- ✅ **Automatic API**: REST API generated automatically
- ✅ **Sharing**: Easy to share and embed

**Trade-offs:**
- ⚠️ Less customization than custom frontend
- ✅ But sufficient for MVP and demos

---

### 9. Text Processing

**Decision:** LangChain (selective imports)

**Alternatives Considered:**
- llama-index
- Haystack
- Custom implementation

**Rationale:**
- ✅ **Text splitters**: Excellent chunking algorithms
- ✅ **Document loaders**: Support for PDF, TXT, MD, etc.
- ✅ **Vector store integrations**: Works with Chroma
- ✅ **Mature**: Battle-tested in production
- ✅ **Modular**: Can import only what we need

**Trade-offs:**
- ⚠️ Heavy dependency (but we use selective imports)
- ⚠️ Frequent breaking changes (but we pin versions)

**Usage Strategy:**
```python
# Only import what we need
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
# Don't import the full langchain package
```

---

### 10. Testing Framework

**Decision:** pytest + pytest-asyncio

**Alternatives Considered:**
- unittest
- nose2

**Rationale:**
- ✅ **Industry standard**: Most popular Python testing framework
- ✅ **Fixtures**: Powerful dependency injection for tests
- ✅ **Plugins**: Rich ecosystem (coverage, mocking, etc.)
- ✅ **Async support**: pytest-asyncio for async code
- ✅ **Readable**: Simple, pythonic syntax

---

### 11. HTTP Client

**Decision:** requests (for Pushover) + httpx (for async if needed)

**Rationale:**
- ✅ **Simple**: requests is the standard for sync HTTP
- ✅ **Reliable**: Battle-tested
- ✅ **Async option**: httpx for future async needs

---

### 12. PDF Processing

**Decision:** pypdf

**Alternatives Considered:**
- PyPDF2
- pdfplumber
- pdfminer

**Rationale:**
- ✅ **Modern**: Active development (PyPDF2 fork)
- ✅ **Simple**: Easy text extraction
- ✅ **Lightweight**: Minimal dependencies
- ✅ **Sufficient**: Works well for LinkedIn PDFs

---

## Complete Technology Stack Summary

### Core Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Main language |
| **Primary LLM** | DeepSeek | deepseek-chat | Conversation generation |
| **Evaluator LLM** | Gemini | 2.0-flash | Quality assessment |
| **Vector DB** | Chroma | 0.4+ | RAG embeddings |
| **SQL DB** | SQLite | 3.x | Analytics & history |
| **Embeddings** | HuggingFace | all-MiniLM-L6-v2 | Vector embeddings |
| **ORM** | SQLAlchemy | 2.0+ | Database operations |
| **Validation** | Pydantic | 2.0+ | Type safety & config |
| **Web UI** | Gradio | 4.0+ | User interface |
| **Text Processing** | LangChain | 0.1+ | Document loading & chunking |

### Supporting Libraries

| Library | Purpose |
|---------|---------|
| `python-dotenv` | Environment variable management |
| `requests` | HTTP client (Pushover) |
| `pypdf` | PDF text extraction |
| `pytest` | Testing framework |
| `pytest-asyncio` | Async testing |
| `black` | Code formatting |
| `ruff` | Linting |

### Development Tools

| Tool | Purpose |
|------|---------|
| `uv` | Fast package manager |
| `pytest-cov` | Test coverage |
| `mypy` | Static type checking |

---

## Dependencies File

```txt
# requirements.txt

# Core
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# LLM & AI
openai>=1.0.0  # Used for DeepSeek and Gemini (OpenAI-compatible APIs)

# RAG & Embeddings
langchain-text-splitters>=0.1.0
langchain-community>=0.1.0
langchain-chroma>=0.1.0
langchain-huggingface>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Database
sqlalchemy>=2.0.0

# PDF Processing
pypdf>=3.0.0

# HTTP & Notifications
requests>=2.31.0

# Web UI
gradio>=4.0.0

# Development (optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.1.0
mypy>=1.5.0
```

---

## Cost Analysis

### Monthly Cost Estimate (1000 conversations/month)

**Assumptions:**
- Average conversation: 3 turns
- Average input: 500 tokens/turn
- Average output: 200 tokens/turn
- Evaluation: 300 tokens input, 50 tokens output

**Calculations:**

| Component | Usage | Cost |
|-----------|-------|------|
| **DeepSeek (Primary)** | 1.5M input, 0.6M output | $0.21 + $0.17 = **$0.38** |
| **Gemini (Evaluator)** | 0.9M input, 0.15M output | $0.07 + $0.05 = **$0.12** |
| **Embeddings** | Local (free) | **$0.00** |
| **Storage** | SQLite + Chroma (free) | **$0.00** |
| **Hosting** | HuggingFace Spaces (free tier) | **$0.00** |
| **Total** | | **$0.50/month** |

**Comparison with OpenAI:**
- OpenAI GPT-4o-mini: ~$1.20/month (2.4x more expensive)
- OpenAI + OpenAI embeddings: ~$1.50/month (3x more expensive)

**Scalability:**
- 10,000 conversations/month: ~$5.00
- 100,000 conversations/month: ~$50.00

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Response Time** | < 3s | P95 latency |
| **RAG Retrieval** | < 200ms | Average |
| **Evaluation** | < 1s | Average |
| **Embedding** | < 100ms | Per document |
| **Database Query** | < 50ms | Average |

---

## Deployment Considerations

### HuggingFace Spaces Constraints
- ✅ **CPU-only**: All chosen technologies work on CPU
- ✅ **Memory**: ~16GB available (sufficient)
- ✅ **Storage**: Persistent disk for SQLite and Chroma
- ✅ **Network**: Can call external APIs (DeepSeek, Gemini)

### Migration Path
If we outgrow HuggingFace Spaces:
1. **Move to dedicated server** (AWS, GCP, Azure)
2. **Upgrade SQLite → PostgreSQL** (SQLAlchemy makes this easy)
3. **Consider Chroma → Pinecone** (if dataset grows large)
4. **Add Redis** for caching (optional)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **DeepSeek API downtime** | Fallback to OpenAI (same API format) |
| **Chroma performance issues** | Monitor query times, optimize chunk size |
| **SQLite concurrency** | Acceptable for single-user use case |
| **Dependency conflicts** | Pin all versions, use uv for fast resolution |
| **HF Spaces limitations** | Design for easy migration to dedicated hosting |

---

## Future Considerations

### Potential Upgrades
1. **Add caching layer** (Redis) for frequently asked questions
2. **Upgrade to PostgreSQL** if analytics become complex
3. **Add observability** (Langfuse, LangSmith) for monitoring
4. **Consider fine-tuning** DeepSeek on career-specific data
5. **Add A/B testing** for prompt variations

### Technology Watch List
- **Gemini 2.0 Pro**: When available, might replace DeepSeek
- **Anthropic Claude**: If cost comes down
- **Local LLMs**: Llama 4, Mistral 3 for cost reduction
- **Vector DBs**: Qdrant, Milvus if we need scale

---

## References

### Official Documentation
- [DeepSeek API Docs](https://platform.deepseek.com/docs)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Chroma Documentation](https://docs.trychroma.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/)

### Benchmarks & Comparisons
- [LLM Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [Embedding Model Benchmark](https://huggingface.co/spaces/mteb/leaderboard)

### Related ADRs
- [ADR 001: Modular Architecture](./001-modular-architecture-with-rag-and-evaluator.md)
- ADR 003: Memory Management Strategy (to be created)

---

**Last Updated:** 2025-10-26  
**Authors:** Development Team  
**Reviewers:** TBD

