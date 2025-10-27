# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Career Chatbot DeepSeek project.

## What are ADRs?

Architecture Decision Records (ADRs) document important architectural decisions made during the project, including:
- Context and problem statement
- Alternatives considered
- Decision outcome and rationale
- Consequences (positive and negative)

## ADR Index

### [ADR 001: Modular Architecture with RAG and Evaluator Pattern](./001-modular-architecture-with-rag-and-evaluator.md)
**Status:** Proposed  
**Date:** 2025-10-26

**Summary:** Refactor monolithic career chatbot into a production-ready layered architecture with:
- Clean separation of concerns (UI → Controller → Agents → Services → Data)
- RAG for knowledge base retrieval
- Evaluator pattern for quality control
- Multi-agent collaboration
- SOLID principles throughout

**Key Decisions:**
- Layered architecture with dependency injection
- 5 agentic patterns applied
- Tool registry for extensibility
- Comprehensive testing strategy

---

### [ADR 002: Technology Stack Selection](./002-technology-stack-selection.md)
**Status:** Proposed  
**Date:** 2025-10-26

**Summary:** Comprehensive technology choices for each component with detailed comparisons and justifications.

**Key Decisions:**
- **Primary LLM:** DeepSeek (cost-effective, $0.14/$0.28 per M tokens)
- **Evaluator LLM:** Gemini 2.0 Flash (fastest, cheapest)
- **Vector DB:** Chroma (embedded, zero setup) with migration path to LanceDB/Milvus
- **SQL DB:** SQLite (embedded, sufficient)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (free, offline) with upgrade path to Gemini
- **Memory:** Custom multi-layer implementation
- **ORM:** SQLAlchemy
- **Validation:** Pydantic v2
- **Web UI:** Gradio

**Cost Analysis:** ~$0.50/month for 1000 conversations

---

### [ADR 003: Memory Management Strategy](./003-memory-management-strategy.md)
**Status:** Proposed  
**Date:** 2025-10-26

**Summary:** Four-layer memory system following context engineering best practices.

**Key Decisions:**
- **Working Memory:** Last 10 turns (always included)
- **Episodic Memory:** Semantic search over past conversations
- **Semantic Memory:** Extracted facts about user
- **Procedural Memory:** Learned interaction patterns

**Benefits:**
- 30-50% cost reduction through smart context management
- Better response quality with relevant historical context
- Handles unlimited conversation length
- Personalized experience

---

## Future ADRs

### Planned:
- **ADR 004:** Tool System Design
- **ADR 005:** Evaluation Criteria and Metrics
- **ADR 006:** Deployment Strategy and CI/CD
- **ADR 007:** Security and Privacy Considerations
- **ADR 008:** Monitoring and Observability
- **ADR 009:** Testing Strategy and Coverage
- **ADR 010:** Performance Optimization Techniques

---

## ADR Template

When creating new ADRs, use this structure:

```markdown
# ADR NNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded  
**Date:** YYYY-MM-DD  
**Deciders:** Team members involved  
**Related ADRs:** Links to related ADRs

## Context and Problem Statement
[Describe the context and problem]

## Decision Drivers
[List factors influencing the decision]

## Considered Options
[List alternatives considered]

## Decision Outcome
[Describe the chosen option and rationale]

## Positive Consequences
[List benefits]

## Negative Consequences
[List drawbacks and mitigation strategies]

## References
[Links to relevant resources]
```

---

## How to Use ADRs

### For Developers:
1. **Read ADRs** before making architectural changes
2. **Propose new ADRs** for significant decisions
3. **Update ADRs** when decisions change
4. **Reference ADRs** in code comments and PRs

### For Reviewers:
1. **Check ADRs** during code review
2. **Ensure alignment** with documented decisions
3. **Flag deviations** from ADRs
4. **Suggest ADR updates** when needed

### For New Team Members:
1. **Start with ADR 001** for architecture overview
2. **Read ADR 002** for technology stack
3. **Read ADR 003** for memory management
4. **Reference as needed** during development

---

## ADR Lifecycle

```
Proposed → Under Review → Accepted → Implemented
                                   ↓
                              Deprecated/Superseded
```

### Status Definitions:
- **Proposed:** Initial draft, under discussion
- **Accepted:** Approved by team, ready for implementation
- **Implemented:** Decision has been implemented
- **Deprecated:** No longer recommended
- **Superseded:** Replaced by a newer ADR

---

## Contributing

When proposing a new ADR:

1. **Create a new file:** `NNN-short-title.md`
2. **Use the template** above
3. **Number sequentially:** Next available number
4. **Update this README:** Add to index
5. **Get review:** From at least 2 team members
6. **Update status:** From Proposed → Accepted

---

## References

### ADR Resources:
- [ADR GitHub Organization](https://adr.github.io/)
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR Tools](https://github.com/npryce/adr-tools)

### Related Documentation:
- [Project README](../README.md)
- [Implementation Guide](../docs/implementation-guide.md)
- [API Documentation](../docs/api.md)

---

**Last Updated:** 2025-10-26  
**Maintainers:** Development Team

