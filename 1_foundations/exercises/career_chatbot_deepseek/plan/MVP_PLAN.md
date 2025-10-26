# Career Chatbot MVP Implementation Plan

**Project:** Career Chatbot with RAG, Memory, and Evaluator  
**Current State:** Monolithic `app_deepseek.py` (192 lines)  
**Target State:** Modular architecture with 4-layer memory system  
**Methodology:** Incremental delivery with feature-complete sprints  
**Total Duration:** 5 sprints

---

## MVP Scope Definition

### What's IN the MVP ✅
- ✅ **Core conversation** with DeepSeek
- ✅ **Basic RAG** with Chroma (knowledge base retrieval)
- ✅ **Working Memory** (recent turns only)
- ✅ **Tool calling** (record_user_details, record_unknown_question)
- ✅ **Pushover notifications**
- ✅ **Gradio interface**
- ✅ **SQLite persistence** (conversations, contacts)
- ✅ **Basic evaluator** (quality check)
- ✅ **Configuration management**
- ✅ **Unit tests** for core components

### What's OUT of MVP (Future Iterations) ⏭️
- ⏭️ Episodic Memory (semantic search over history)
- ⏭️ Semantic Memory (fact extraction)
- ⏭️ Procedural Memory (pattern learning)
- ⏭️ Advanced analytics dashboard
- ⏭️ Multiple LLM providers
- ⏭️ Streaming responses
- ⏭️ Multi-user support
- ⏭️ Advanced caching strategies

### MVP Success Criteria
1. ✅ All existing functionality from `app_deepseek.py` preserved
2. ✅ Modular architecture with clear separation of concerns
3. ✅ 70%+ test coverage
4. ✅ Successfully deployed to HuggingFace Spaces
5. ✅ Response time < 3 seconds (P95)
6. ✅ Zero breaking changes for end users

---

## Package Management

### Dependencies
This project leverages the **parent project's** package management:
- **Location:** `path/to/agents/`
- **Files:** `pyproject.toml`, `requirements.txt`, `uv.lock`, `.python-version`

### Adding New Dependencies
When you need new packages for this MVP:

```bash
# Navigate to parent project root
cd path/to/agents/

# Add package to requirements.txt
echo "chromadb>=0.4.0" >> requirements.txt

# Install with uv
uv pip install -r requirements.txt

# Commit the change
git add requirements.txt
git commit -m "deps: add chromadb for RAG functionality"
```

### Required MVP Dependencies
Add these to parent `requirements.txt` if not already present:

```txt
# Already in parent requirements.txt:
# - openai (for DeepSeek API)
# - gradio (for UI)
# - pypdf (for PDF reading)
# - python-dotenv (for .env)
# - requests (for Pushover)

# Need to add for MVP:
chromadb>=0.4.22          # Vector database for RAG
pydantic>=2.0.0           # Already present
pydantic-settings>=2.0.0  # Configuration management
sqlalchemy>=2.0.0         # ORM for SQLite
pytest>=7.4.0             # Testing
pytest-cov>=4.1.0         # Coverage
```

---

## Sprint Structure

### Sprint Principles
- **One sprint = One complete, deployable feature**
- **Each sprint is independent** and can be deployed
- **Manual execution** - you implement step by step
- **No CI/CD setup** for now (can add later)

### Definition of Done (DoD)
- [ ] Code written and self-reviewed
- [ ] Unit tests written and passing
- [ ] Code follows clean code principles
- [ ] Documentation updated (inline comments + docstrings)
- [ ] Feature is manually tested and working
- [ ] Can be deployed to HuggingFace Spaces

---

## Sprint Overview

Each sprint is documented in a separate file for detailed implementation:

### Sprint 1: Foundation & Data Models
**File:** `sprint_1_foundation.md`  
**Goal:** Set up project structure, configuration, and data models  
**Deliverable:** Working configuration system + database models  
**Duration:** ~1-2 weeks

### Sprint 2: Services Layer
**File:** `sprint_2_services.md`  
**Goal:** Implement core services (Database, LLM, Notification)  
**Deliverable:** Reusable service layer with tests  
**Duration:** ~1-2 weeks

### Sprint 3: RAG & Tools
**File:** `sprint_3_rag_tools.md`  
**Goal:** Implement RAG service and tool system  
**Deliverable:** Working knowledge base retrieval + tool calling  
**Duration:** ~1-2 weeks

### Sprint 4: Agents & Memory
**File:** `sprint_4_agents.md`  
**Goal:** Implement Career Agent, Evaluator, and Working Memory  
**Deliverable:** Complete agent system with evaluation  
**Duration:** ~1-2 weeks

### Sprint 5: Integration & Deployment
**File:** `sprint_5_integration.md`  
**Goal:** Wire everything together and deploy  
**Deliverable:** Deployed MVP on HuggingFace Spaces  
**Duration:** ~1-2 weeks

---

## How to Use This Plan

### Step 1: Read Architecture Documents
```bash
cd /path/to/career_chatbot_deepseek

# Read these in order:
1. adr/001-modular-architecture-with-rag-and-evaluator.md
2. adr/002-technology-stack-selection.md
3. adr/003-memory-management-strategy.md
4. plan/MVP_PLAN.md (this file)
```

### Step 2: Start Sprint 1
```bash
# Read sprint file
cat plan/sprint_1_foundation.md

# Follow step-by-step instructions
# Implement each task
# Test as you go
```

### Step 3: Continue Through Sprints
- Complete Sprint 1 fully before moving to Sprint 2
- Each sprint builds on the previous one
- Test thoroughly after each sprint
- Can deploy after any sprint (incremental delivery)

### Step 4: Deploy MVP
- After Sprint 5, you'll have a complete MVP
- Deploy to HuggingFace Spaces
- Monitor and iterate

---

## Quick Reference

### Project Structure (Target)
```
career_chatbot_deepseek/
├── config/
│   ├── __init__.py
│   └── settings.py          # Pydantic Settings
├── models/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic models
│   └── database.py          # SQLAlchemy models
├── services/
│   ├── __init__.py
│   ├── database_service.py  # SQL operations
│   ├── llm_service.py       # DeepSeek wrapper
│   ├── rag_service.py       # Chroma RAG
│   └── notification_service.py  # Pushover
├── tools/
│   ├── __init__.py
│   ├── base.py              # BaseTool
│   ├── registry.py          # ToolRegistry
│   ├── contact_tool.py      # Record contact
│   ├── question_tool.py     # Record unknown Q
│   └── search_tool.py       # RAG search
├── memory/
│   ├── __init__.py
│   ├── memory_service.py    # Memory orchestrator
│   └── working_memory.py    # Recent turns
├── core/
│   ├── __init__.py
│   ├── career_agent.py      # Main agent
│   ├── evaluator_agent.py   # Quality checker
│   └── controller.py        # Orchestrator
├── utils/
│   ├── __init__.py
│   ├── file_loader.py       # PDF/TXT loader
│   └── prompt_builder.py    # Prompt templates
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── knowledge_base/
│   ├── linkedin.pdf
│   └── summary.txt
├── data/                    # Created at runtime
│   ├── career_qa.db
│   └── chroma_db/
├── app.py                   # Gradio entry point
├── .env.example
└── README.md
```

### Key Technologies
- **LLM:** DeepSeek (via OpenAI SDK)
- **Evaluator:** Google Gemini 2.0 Flash
- **Vector DB:** Chroma
- **SQL DB:** SQLite + SQLAlchemy
- **Validation:** Pydantic
- **UI:** Gradio
- **Notifications:** Pushover

### Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_database_service.py -v

# Run specific test
pytest tests/unit/test_database_service.py::test_save_conversation -v
```

### Common Git Workflow
```bash
# Start new feature
git checkout -b feature/sprint-1-config

# Work on code
# ... make changes ...

# Test
pytest

# Commit
git add .
git commit -m "feat(config): implement Pydantic Settings"

# Merge to main
git checkout main
git merge feature/sprint-1-config

# Tag sprint completion
git tag -a v0.1.0 -m "Sprint 1: Foundation complete"
```

---

## Progress Tracking

### Sprint Completion Checklist

- [ ] **Sprint 1:** Foundation & Data Models
  - [ ] Project structure created
  - [ ] Configuration management working
  - [ ] Pydantic schemas defined
  - [ ] SQLAlchemy models defined
  - [ ] Unit tests passing

- [ ] **Sprint 2:** Services Layer
  - [ ] Database service implemented
  - [ ] LLM service implemented
  - [ ] Notification service implemented
  - [ ] Utils implemented
  - [ ] Unit tests passing

- [ ] **Sprint 3:** RAG & Tools
  - [ ] RAG service with Chroma working
  - [ ] Tool base & registry implemented
  - [ ] Three concrete tools working
  - [ ] Integration tests passing

- [ ] **Sprint 4:** Agents & Memory
  - [ ] Career agent implemented
  - [ ] Evaluator agent implemented
  - [ ] Working memory implemented
  - [ ] Agent tests passing

- [ ] **Sprint 5:** Integration & Deployment
  - [ ] Controller orchestrating all components
  - [ ] app.py wiring complete
  - [ ] End-to-end tests passing
  - [ ] Deployed to HuggingFace Spaces
  - [ ] Production testing complete

---

## Success Metrics

### Per Sprint
- [ ] All tasks completed
- [ ] Tests passing (70%+ coverage)
- [ ] Code follows clean code principles
- [ ] Documentation updated
- [ ] Feature manually tested

### MVP Complete
- [ ] All 5 sprints done
- [ ] Deployed to HuggingFace Spaces
- [ ] All original features working
- [ ] Response time < 3s
- [ ] No breaking changes

---

## Next Steps

1. **Read Sprint 1 file:** `plan/sprint_1_foundation.md`
2. **Set up environment:** Install dependencies from parent project
3. **Start implementing:** Follow Sprint 1 tasks step by step
4. **Test frequently:** Run tests after each task
5. **Commit often:** Small, focused commits

**Ready to start? Open `plan/sprint_1_foundation.md`!** 🚀
