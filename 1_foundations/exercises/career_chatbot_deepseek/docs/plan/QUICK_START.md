# Quick Start Guide - MVP Implementation

**Goal:** Transform monolithic `app_deepseek.py` into modular architecture  
**Time:** 5-10 weeks (5 sprints)  
**Methodology:** Incremental feature delivery

---

## 📋 Prerequisites

- [ ] Python 3.11+
- [ ] Git
- [ ] DeepSeek API key
- [ ] (Optional) Google API key for evaluator
- [ ] (Optional) Pushover account for notifications

---

## 🚀 Getting Started (Day 1)

### Step 1: Review Documentation (1 hour)
```bash
# Read these files in order:
1. ../adr/001-modular-architecture-with-rag-and-evaluator.md
2. ../adr/002-technology-stack-selection.md
3. ../adr/003-memory-management-strategy.md
4. MVP_PLAN.md (overview)
5. sprint_1_foundation.md (start here)
```

### Step 2: Set Up Environment (30 minutes)
```bash
# Navigate to parent project (uses parent's package management)
cd path/to/agents/

# Activate virtual environment (should already exist)
source .venv/bin/activate  # On macOS/Linux

# Verify Python version
python --version  # Should be 3.12

# Install/update dependencies
uv pip install -r requirements.txt
```

### Step 3: Navigate to Project
```bash
# Go to career chatbot project
cd 1_foundations/exercises/career_chatbot_deepseek

# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

---

## 📅 Sprint Workflow

### Starting a Sprint

1. **Read the sprint file** (e.g., `plan/sprint_1_foundation.md`)
2. **Understand the goal** - What complete feature will you deliver?
3. **Review all tasks** - Understand dependencies
4. **Set up environment** - Add any new dependencies

### During Sprint

- **Work through tasks sequentially** - Each task builds on previous
- **Test frequently** - Run tests after each task
- **Commit often** - Small, focused commits
- **Update sprint file** - Check off completed tasks

### Completing a Sprint

1. **Run all tests** - Ensure everything passes
2. **Manual testing** - Test the complete feature
3. **Update retrospective** - Note what went well/poorly
4. **Tag the sprint** - `git tag -a v0.1.0 -m "Sprint 1 complete"`
5. **Move to next sprint** - Open next sprint file

---

## 💻 Daily Development Workflow

### Morning Routine (15 minutes)
```bash
# 1. Check CI/CD status
# Visit GitHub Actions page

# 2. Review yesterday's work
git log --oneline -5

# 3. Pull latest changes
git pull origin main

# 4. Plan today's tasks
# Update SPRINT_BOARD.md with today's focus
```

### Development Cycle (TDD Approach)

#### 1. Write Test First (Red)
```bash
# Create test file
touch tests/unit/test_my_feature.py

# Write failing test
cat > tests/unit/test_my_feature.py << 'EOF'
def test_my_feature():
    result = my_function()
    assert result == expected_value
EOF

# Run test (should fail)
pytest tests/unit/test_my_feature.py -v
```

#### 2. Write Implementation (Green)
```python
# Create implementation file
# Write minimal code to pass test
# Run test again (should pass)
```

#### 3. Refactor (Refactor)
```python
# Clean up code
# Improve naming
# Extract functions
# Run tests (should still pass)
```

#### 4. Commit
```bash
git add .
git commit -m "feat(scope): add my_feature with tests"
```

### Testing Commands
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_database_service.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run with verbose output
pytest -v

# Run only failed tests
pytest --lf

# Run tests matching pattern
pytest -k "test_database"
```

### Code Quality Checks
```bash
# Format code
black .

# Lint code
ruff check .

# Fix linting issues
ruff check --fix .

# Type check
mypy --ignore-missing-imports .

# Run all checks
black . && ruff check . && mypy --ignore-missing-imports . && pytest
```

### Evening Routine (15 minutes)
```bash
# 1. Run full test suite
pytest --cov=.

# 2. Check coverage
# Should be > 70%

# 3. Commit and push
git push origin feature/story-X.Y-description

# 4. Update progress
# Update SPRINT_BOARD.md with completed tasks

# 5. Plan tomorrow
# Note any blockers or questions
```

---

## 📊 Sprint Review (End of Sprint)

### Preparation (1 day before)
- [ ] Ensure all tests pass
- [ ] Check test coverage meets target
- [ ] Update documentation
- [ ] Prepare demo
- [ ] Collect metrics

### During Sprint Review (1 hour)

#### 1. Demo (30 minutes)
- Show completed stories
- Demonstrate new features
- Run live tests

#### 2. Metrics Review (15 minutes)
- Story points completed vs planned
- Test coverage achieved
- Bugs found and fixed
- Velocity calculation

#### 3. Stakeholder Feedback (15 minutes)
- What worked well?
- What needs improvement?
- Any new requirements?

### After Sprint Review
- [ ] Update `SPRINT_BOARD.md` with metrics
- [ ] Document feedback
- [ ] Update backlog if needed

---

## 🔄 Sprint Retrospective (End of Sprint)

### Format (30 minutes)

#### 1. Set the Stage (5 min)
- Review sprint goal
- Review what was accomplished

#### 2. Gather Data (10 min)
**What went well?**
- 

**What could be improved?**
- 

**What puzzled us?**
- 

#### 3. Generate Insights (10 min)
- Why did things go well/poorly?
- What patterns do we see?
- What can we learn?

#### 4. Decide What to Do (5 min)
**Action Items for Next Sprint:**
1. 
2. 
3. 

### After Retrospective
- [ ] Document in `SPRINT_BOARD.md`
- [ ] Add action items to next sprint planning
- [ ] Share learnings with team (if applicable)

---

## 🐛 Debugging Guide

### Test Failures
```bash
# Run specific test with verbose output
pytest tests/unit/test_database_service.py::test_save_conversation -vv

# Run with print statements
pytest -s tests/unit/test_database_service.py

# Run with debugger
pytest --pdb tests/unit/test_database_service.py
```

### Import Errors
```bash
# Check if module is in PYTHONPATH
python -c "import sys; print('\n'.join(sys.path))"

# Install in editable mode
pip install -e .
```

### Database Issues
```bash
# Delete and recreate database
rm data/career_qa.db
python -c "from models.database import init_db; init_db()"
```

### CI/CD Failures
1. Check GitHub Actions logs
2. Reproduce locally:
   ```bash
   # Run same commands as CI
   ruff check .
   black --check .
   pytest --cov=.
   ```
3. Fix issues
4. Push again

---

## 📈 Progress Tracking

### Daily
- [ ] Update task status in `SPRINT_BOARD.md`
- [ ] Commit code with clear messages
- [ ] Run tests

### Weekly
- [ ] Review sprint progress
- [ ] Check if on track for sprint goal
- [ ] Address blockers

### Sprint End
- [ ] Complete sprint review
- [ ] Complete retrospective
- [ ] Update velocity
- [ ] Plan next sprint

---

## 🎯 Success Criteria Checklist

### Per Story
- [ ] All acceptance criteria met
- [ ] Unit tests written and passing
- [ ] Integration tests (if applicable)
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] CI/CD pipeline green

### Per Sprint
- [ ] Sprint goal achieved
- [ ] All committed stories completed
- [ ] Test coverage target met
- [ ] No critical bugs
- [ ] Demo successful

### MVP Complete
- [ ] All 6 sprints completed
- [ ] Deployed to HuggingFace Spaces
- [ ] All features working
- [ ] Performance targets met (< 3s response)
- [ ] Documentation complete
- [ ] Migration guide written

---

## 🆘 Getting Help

### Common Issues

**Issue:** Tests failing with import errors  
**Solution:** Ensure virtual environment is activated and dependencies installed

**Issue:** CI/CD pipeline failing  
**Solution:** Run same checks locally first

**Issue:** Story taking longer than estimated  
**Solution:** Break it down into smaller tasks or reduce scope

**Issue:** Stuck on implementation  
**Solution:** Review ADR documents, check similar code in codebase, or simplify approach

### Resources
- ADR Documents: `../adr/` folder
- MVP Plan: `MVP_PLAN.md`
- Original Code: `app_deepseek.py`

---

## 📝 Templates

### Daily Standup (Async)
```
**Yesterday:**
- Completed: Story 1.2 - Configuration Management
- Challenges: None

**Today:**
- Plan: Start Story 1.3 - Data Models
- Focus: Pydantic schemas

**Blockers:**
- None
```

### Commit Message
```
feat(services): implement database service

- Add DatabaseService with CRUD operations
- Add analytics queries
- Add comprehensive unit tests

Relates to Sprint 2, Story 2.1
```

### Pull Request
```
## Description
Implements database service with CRUD operations for conversations, contacts, and unknown questions.

## Changes
- Added DatabaseService class
- Implemented save_conversation, save_contact, record_unknown_question
- Added analytics queries
- Added comprehensive unit tests

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Manual testing completed

## Checklist
- [x] Code follows style guidelines
- [x] Tests added
- [x] Documentation updated
- [x] CI/CD passing

Closes #123
```

---

## 🎉 Celebration Milestones

- ✅ Pre-Sprint 0 Complete: CI/CD set up!
- ✅ Sprint 1 Complete: Foundation solid!
- ✅ Sprint 2 Complete: Services working!
- ✅ Sprint 3 Complete: RAG & Tools ready!
- ✅ Sprint 4 Complete: Agents alive!
- ✅ Sprint 5 Complete: Everything integrated!
- ✅ Sprint 6 Complete: MVP DEPLOYED! 🚀

---

**Ready to start?** Begin with Sprint 1 in `sprint_1_foundation.md`!

**Questions?** Review the ADR documents in `../adr/` folder.

**Need details?** Check `MVP_PLAN.md` for complete implementation code.

Good luck! 🍀

