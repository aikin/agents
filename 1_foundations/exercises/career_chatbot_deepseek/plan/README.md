# Career Chatbot MVP - Implementation Plan

**Version:** 1.0  
**Date:** 2025-10-26  
**Status:** Ready for Implementation

---

## 📁 Plan Structure

This folder contains all planning documents for the MVP implementation:

```
plan/
├── README.md                      # This file - overview and navigation
├── MVP_PLAN.md                    # High-level overview and quick reference
├── QUICK_START.md                 # Getting started guide
├── sprint_1_foundation.md         # Sprint 1: Foundation & Data Models
├── sprint_2_services.md           # Sprint 2: Services Layer
├── sprint_3_rag_tools.md          # Sprint 3: RAG & Tools
├── sprint_4_agents.md             # Sprint 4: Agents & Memory
└── sprint_5_integration.md        # Sprint 5: Integration & Deployment
```

---

## 🎯 MVP Overview

**Goal:** Transform monolithic `app_deepseek.py` into a modular, production-ready career chatbot

**Key Features:**
- ✅ Conversational AI with DeepSeek
- ✅ RAG with Chroma vector database
- ✅ Tool calling (contact, unknown questions, search)
- ✅ Quality evaluation with Gemini
- ✅ Working memory for context
- ✅ SQLite persistence
- ✅ Pushover notifications
- ✅ Gradio interface
- ✅ HuggingFace Spaces deployment

**Duration:** 5-10 weeks (5 sprints)

---

## 🚀 Quick Start

### 1. Read Architecture Documents First

```bash
# Navigate to project
cd path/to/agents/1_foundations/exercises/career_chatbot_deepseek

# Read ADRs (Architectural Decision Records)
cat adr/001-modular-architecture-with-rag-and-evaluator.md
cat adr/002-technology-stack-selection.md
cat adr/003-memory-management-strategy.md
```

### 2. Review Planning Documents

```bash
# Read overview
cat plan/MVP_PLAN.md

# Read quick start guide
cat plan/QUICK_START.md
```

### 3. Start Sprint 1

```bash
# Read Sprint 1 plan
cat plan/sprint_1_foundation.md

# Follow step-by-step instructions
# Implement, test, commit
```

---

## 📋 Sprint Overview

### Sprint 1: Foundation & Data Models (~1-2 weeks)
**File:** `sprint_1_foundation.md`

**Goal:** Set up project structure, configuration, and data models

**Deliverables:**
- Project directory structure
- Pydantic Settings for configuration
- Pydantic schemas for validation
- SQLAlchemy models for database
- Unit tests (80%+ coverage)

**Key Tasks:**
1. Create project structure
2. Implement configuration management
3. Define Pydantic schemas
4. Create SQLAlchemy models

---

### Sprint 2: Services Layer (~1-2 weeks)
**File:** `sprint_2_services.md`

**Goal:** Implement core services

**Deliverables:**
- Database Service (CRUD operations)
- LLM Service (DeepSeek wrapper)
- Notification Service (Pushover)
- File Loader & Prompt Builder utilities
- Unit tests

**Key Tasks:**
1. Database Service with analytics
2. LLM Service wrapper
3. Notification Service
4. Utility functions

---

### Sprint 3: RAG & Tools (~1-2 weeks)
**File:** `sprint_3_rag_tools.md`

**Goal:** Implement RAG and tool system

**Deliverables:**
- RAG Service with Chroma
- Tool base class and registry
- Three concrete tools (Contact, Question, Search)
- Integration tests

**Key Tasks:**
1. RAG Service implementation
2. Tool base & registry
3. Concrete tool implementations

---

### Sprint 4: Agents & Memory (~1-2 weeks)
**File:** `sprint_4_agents.md`

**Goal:** Implement agents and memory

**Deliverables:**
- Career Agent with tool calling
- Evaluator Agent with Gemini
- Working Memory system
- Memory Service orchestrator
- Unit tests

**Key Tasks:**
1. Career Agent with agentic loop
2. Evaluator Agent
3. Working Memory implementation

---

### Sprint 5: Integration & Deployment (~1-2 weeks)
**File:** `sprint_5_integration.md`

**Goal:** Wire everything together and deploy

**Deliverables:**
- Controller with Evaluator-Optimizer pattern
- Gradio app.py with dependency injection
- End-to-end tests
- HuggingFace Spaces deployment
- Production-ready MVP

**Key Tasks:**
1. Controller implementation
2. App.py integration
3. E2E testing
4. HuggingFace deployment

---

## 🏗️ Architecture Principles

### Clean Code
- **SRP (Single Responsibility Principle)** - Each class has one reason to change
- **DIP (Dependency Inversion)** - Depend on abstractions, not concretions
- **DRY (Don't Repeat Yourself)** - Reuse code through abstraction
- **KISS (Keep It Simple)** - Simplicity over complexity

### Agentic Patterns
- **Tool Use** - LLM calls external functions
- **Reflection (Evaluator-Optimizer)** - Quality assessment and regeneration
- **RAG** - Retrieval Augmented Generation
- **Memory** - Context management (Working Memory for MVP)

### Layered Architecture
```
┌─────────────────────────────────────┐
│         Presentation Layer          │
│            (app.py)                 │
├─────────────────────────────────────┤
│           Core Layer                │
│  (Agents, Controller, Memory)       │
├─────────────────────────────────────┤
│         Services Layer              │
│  (Database, LLM, RAG, Notification) │
├─────────────────────────────────────┤
│          Tools Layer                │
│   (Contact, Question, Search)       │
├─────────────────────────────────────┤
│          Models Layer               │
│    (Schemas, Database Models)       │
├─────────────────────────────────────┤
│          Config Layer               │
│         (Settings)                  │
└─────────────────────────────────────┘
```

---

## 📦 Package Management

**Important:** This project uses the **parent project's** package management.

### Location
- **Parent:** `path/to/agents/`
- **Files:** `pyproject.toml`, `requirements.txt`, `uv.lock`, `.python-version`

### Adding Dependencies

```bash
# Navigate to parent project
cd path/to/agents/

# Add to requirements.txt
echo "chromadb>=0.4.22" >> requirements.txt

# Install
uv pip install -r requirements.txt

# Commit
git add requirements.txt
git commit -m "deps: add chromadb for RAG"
```

### Required Dependencies

Already in parent `requirements.txt`:
- openai (DeepSeek API)
- gradio (UI)
- pypdf (PDF reading)
- python-dotenv (.env)
- requests (Pushover)

Need to add for MVP:
- chromadb (Vector DB)
- pydantic-settings (Config)
- sqlalchemy (ORM)
- pytest, pytest-cov (Testing)
- google-generativeai (Evaluator)

---

## 🧪 Testing Strategy

### Unit Tests
- Test each class/function in isolation
- Use mocking for dependencies
- Aim for 80%+ coverage
- Fast execution (< 1 second per test)

### Integration Tests
- Test component interactions
- Use in-memory databases
- Test happy paths and error cases
- Slower execution acceptable

### Manual Testing
- Test conversation flow
- Test tool calling
- Test RAG search
- Test evaluation
- Test notifications

---

## 📝 Commit Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding tests
- `docs`: Documentation
- `chore`: Maintenance

**Example:**
```
feat(services): implement database service with CRUD operations

- Add DatabaseService with context manager
- Implement save_conversation, save_contact, record_unknown_question
- Add analytics queries
- Add comprehensive unit tests

Relates to Sprint 2, Task 2.1
```

---

## ✅ Success Criteria

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
- [ ] Test coverage > 70%

---

## 📚 Additional Resources

### Architecture Documents
- `adr/001-modular-architecture-with-rag-and-evaluator.md` - Main architecture
- `adr/002-technology-stack-selection.md` - Tech stack choices
- `adr/003-memory-management-strategy.md` - Memory system design
- `adr/README.md` - ADR index

### Reference
- Original monolithic code: `../app_deepseek.py`
- Knowledge base: `knowledge_base/` (linkedin.pdf, summary.txt)

---

## 🎯 Next Steps

1. **Read this file** ✅ (you're here!)
2. **Read `MVP_PLAN.md`** - Get high-level overview
3. **Read `QUICK_START.md`** - Understand workflow
4. **Read `sprint_1_foundation.md`** - Start implementing
5. **Follow sprint tasks step-by-step** - Build the MVP!

---

## 💡 Tips for Success

### Do's ✅
- Read architecture documents first
- Follow sprints sequentially
- Test after each task
- Commit often with clear messages
- Update retrospectives
- Ask questions when stuck

### Don'ts ❌
- Skip tests
- Jump between sprints
- Commit without testing
- Ignore linter errors
- Copy-paste without understanding
- Rush through tasks

---

## 🆘 Getting Help

### Common Issues

**Q: Tests failing with import errors**  
A: Ensure you're in the parent project's virtual environment

**Q: Missing dependencies**  
A: Add to parent `requirements.txt` and run `uv pip install -r requirements.txt`

**Q: Task taking longer than estimated**  
A: Break it down into smaller steps or simplify the approach

**Q: Stuck on implementation**  
A: Review ADR documents, check similar code in codebase, or simplify

### Resources
- ADR Documents: `../adr/` folder
- Sprint Files: This folder
- Original Code: `../app_deepseek.py`
- Parent Project: `path/to/agents/`

---

## 📈 Progress Tracking

Track your progress by checking off sprints:

- [ ] Sprint 1: Foundation & Data Models
- [ ] Sprint 2: Services Layer
- [ ] Sprint 3: RAG & Tools
- [ ] Sprint 4: Agents & Memory
- [ ] Sprint 5: Integration & Deployment

**Current Sprint:** ___________  
**Estimated Completion:** ___________

---

**Ready to start?** Open `sprint_1_foundation.md` and begin your MVP journey! 🚀

Good luck! 🍀

