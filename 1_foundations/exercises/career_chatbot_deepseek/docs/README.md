# Career Chatbot Documentation

This directory contains all documentation for the Career Chatbot MVP project, including architectural decisions, implementation plans, and guides.

## 📁 Directory Structure

```
docs/
├── adr/           # Architectural Decision Records
├── plan/          # Sprint plans and implementation guides
└── README.md      # This file
```

## 🏗️ Architectural Decision Records (ADR)

The `adr/` folder contains detailed architectural decisions and design documents:

### Core Architecture
- **[001-modular-architecture-with-rag-and-evaluator.md](adr/001-modular-architecture-with-rag-and-evaluator.md)**
  - Complete system architecture with layered design
  - Mermaid diagrams for architecture and sequence flows
  - Key components and their responsibilities
  - Implementation phases and integration strategy

### Technology Stack
- **[002-technology-stack-selection.md](adr/002-technology-stack-selection.md)**
  - LLM providers comparison (DeepSeek, Gemini)
  - Vector database evaluation (Chroma, LanceDB, PGVector, Milvus)
  - Embedding models comparison
  - SQL database and ORM selection
  - Configuration management approach

### Memory Management
- **[003-memory-management-strategy.md](adr/003-memory-management-strategy.md)**
  - Memory types (Working, Episodic, Semantic, Procedural)
  - Context engineering best practices
  - Memory tech stack selection
  - Implementation strategy for MVP

### Navigation
- **[README.md](adr/README.md)** - ADR index and decision log

## 📋 Implementation Plan

The `plan/` folder contains detailed sprint plans for manual implementation:

### Overview Documents
- **[MVP_PLAN.md](plan/MVP_PLAN.md)** - High-level MVP scope and sprint overview
- **[QUICK_START.md](plan/QUICK_START.md)** - Step-by-step setup and workflow guide
- **[README.md](plan/README.md)** - Plan navigation and usage instructions

### Sprint Plans (Sequential Implementation)

#### Sprint 1: Foundation (Week 1)
- **[sprint_1_foundation.md](plan/sprint_1_foundation.md)**
  - Project structure setup
  - Configuration management with Pydantic Settings
  - Pydantic schemas for type safety
  - SQLAlchemy database models
  - **Status**: ✅ Complete

#### Sprint 2: Core Services (Week 2)
- **[sprint_2_services.md](plan/sprint_2_services.md)**
  - Database service with SQLAlchemy
  - LLM service with DeepSeek/Gemini
  - Notification service with Pushover
  - Utility modules (file loader, prompt builder)

#### Sprint 3: RAG & Tools (Week 3)
- **[sprint_3_rag_tools.md](plan/sprint_3_rag_tools.md)**
  - RAG service with Chroma
  - Tool system architecture
  - Tool registry and base classes
  - Knowledge base ingestion

#### Sprint 4: Agents (Week 4)
- **[sprint_4_agents.md](plan/sprint_4_agents.md)**
  - Career agent implementation
  - Evaluator agent for quality control
  - Working memory management
  - Agent orchestration

#### Sprint 5: Integration & Deployment (Week 5)
- **[sprint_5_integration.md](plan/sprint_5_integration.md)**
  - App.py integration
  - Gradio UI implementation
  - HuggingFace Spaces deployment
  - End-to-end testing

## 🚀 How to Use This Documentation

### For First-Time Setup
1. **Start here**: Read this `README.md` for an overview
2. **Understand the architecture**: Review ADR documents in order (001 → 002 → 003)
3. **Set up your environment**: Follow `plan/QUICK_START.md`
4. **Begin implementation**: Start with `plan/sprint_1_foundation.md`

### For Ongoing Development
1. **Check current sprint**: Refer to `plan/MVP_PLAN.md` for sprint status
2. **Follow sprint tasks**: Work through tasks in the current sprint file
3. **Reference architecture**: Consult ADR documents when making design decisions
4. **Update documentation**: Keep sprint status and TODO.md current

### For Architecture Decisions
1. **Review existing ADRs**: Check if a decision has already been made
2. **Understand context**: Read the "Context" section of relevant ADRs
3. **Follow patterns**: Apply established patterns from ADR 001
4. **Document changes**: Create new ADR if making significant architectural changes

## 📚 Related Documentation

- **Project README**: `../README.md` - Project overview and setup
- **Code Documentation**: Inline docstrings in source files
- **Test Documentation**: Test files in `../tests/` with descriptive names
- **Parent Project**: `/Users/kin/Works/10_Repos/agi/agent/agents/` - Shared dependencies

## 🎯 Documentation Principles

This documentation follows these principles:

1. **KISS (Keep It Simple, Stupid)**: Clear, concise, and easy to understand
2. **DRY (Don't Repeat Yourself)**: Single source of truth for each concept
3. **Actionable**: Provides concrete steps and code examples
4. **Maintainable**: Easy to update as the project evolves
5. **Navigable**: Clear structure with cross-references

## 📝 Contributing to Documentation

When updating documentation:

1. **Keep it current**: Update docs when code changes
2. **Be specific**: Include code examples and file paths
3. **Link related docs**: Cross-reference relevant sections
4. **Follow format**: Maintain consistent markdown formatting
5. **Test examples**: Ensure code snippets are accurate

---

**Last Updated**: Sprint 1 Complete
**Next Review**: Before Sprint 2 begins

For questions or clarifications, refer to the specific ADR or sprint plan file.

