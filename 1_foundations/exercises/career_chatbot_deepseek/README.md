# Career Chatbot MVP

An intelligent career assistant powered by DeepSeek, with RAG, tool calling, and quality evaluation using Gemini.

## Features

- 💬 **Conversational AI** - Natural career Q&A with DeepSeek
- 🔍 **RAG** - Knowledge base retrieval with Chroma vector database
- 🛠️ **Tool Calling** - Contact recording, unknown question tracking, knowledge search
- ✅ **Quality Evaluation** - Gemini-powered response assessment
- 💾 **Persistence** - SQLite database for conversations and analytics
- 🧠 **Memory** - Working memory for conversation context
- 📱 **Notifications** - Pushover integration for real-time alerts

## Architecture

Built with clean code principles and modular design following agentic AI patterns:

```
career_chatbot_deepseek/
├── config/          # Configuration management (Pydantic Settings)
├── models/          # Data models (Pydantic schemas, SQLAlchemy ORM)
├── services/        # Business logic (Database, LLM, RAG, Notifications)
├── tools/           # Function calling tools (Contact, Question, Search)
├── memory/          # Memory management (Working Memory)
├── core/            # Core agents (Career Agent, Evaluator, Controller)
├── utils/           # Utilities (File loader, Prompt builder)
└── tests/           # Unit and integration tests
```

### Agentic Patterns

- **Tool Use** - LLM dynamically calls external functions
- **Reflection (Evaluator-Optimizer)** - Quality assessment and regeneration
- **RAG** - Retrieval Augmented Generation for context-aware responses
- **Memory** - Working memory for conversation context

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

### Installation

#### Option 1: Using uv (Recommended - 10-100x faster)

```bash
# Clone or navigate to the project
cd career_chatbot_deepseek

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies (uses uv.lock for exact versions)
uv sync

# Or install in editable mode for development
uv pip install -e .

# Install development dependencies
uv pip install --group dev
```

#### Option 2: Using pip (Traditional)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install from requirements.txt (exact versions)
pip install -r requirements.txt

# Or install from pyproject.toml
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```bash
# Required
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# Optional
GOOGLE_API_KEY=your-google-api-key-here  # For evaluation
PUSHOVER_USER=your-pushover-user-key     # For notifications
PUSHOVER_TOKEN=your-pushover-app-token
```

3. Add your knowledge base files:
```bash
# Place your files in knowledge_base/
cp /path/to/linkedin.pdf knowledge_base/
cp /path/to/summary.txt knowledge_base/
```

## Usage

### Run Locally

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the Gradio app
python app.py
```

The app will be available at `http://localhost:7860`

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_settings.py -v

# Run specific test
pytest tests/unit/test_database_service.py::test_save_conversation -v
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Fix linting issues
ruff check --fix .

# Type check
mypy .
```

## Development

### Project Structure

- **Config Layer** - Pydantic Settings for type-safe configuration
- **Services Layer** - Database, LLM, RAG, Notification services
- **Tools Layer** - Function calling capabilities
- **Core Layer** - Agents and controller with Evaluator-Optimizer pattern
- **Memory Layer** - Conversation context management

### Adding Dependencies

```bash
# Add a new dependency
uv pip install package-name

# Update pyproject.toml
# Add the package to [project.dependencies]

# Update lock files
uv lock                                    # Update uv.lock
uv pip compile pyproject.toml -o requirements.txt  # Update requirements.txt

# Commit both lock files
git add uv.lock requirements.txt pyproject.toml
git commit -m "chore: add package-name dependency"
```

### Why Two Lock Files?

- **`uv.lock`** - For uv users (native format, faster)
  - Exact versions of all dependencies
  - Cross-platform reproducibility
  - Used by `uv sync`

- **`requirements.txt`** - For pip users (compatibility)
  - Generated from `pyproject.toml`
  - Works with traditional pip workflows
  - Used by CI/CD, HuggingFace, Docker

Both files are committed to version control for reproducibility.

### Running Sprints

This project follows a sprint-based implementation plan. See `plan/` folder for detailed sprint guides:

1. **Sprint 1** - Foundation & Data Models
2. **Sprint 2** - Services Layer
3. **Sprint 3** - RAG & Tools
4. **Sprint 4** - Agents & Memory
5. **Sprint 5** - Integration & Deployment

## Deployment

### HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/new-space)
2. Choose Gradio SDK
3. Clone the Space repository
4. Copy project files
5. Add secrets in Space settings (API keys)
6. Push to deploy

See `plan/sprint_5_integration.md` for detailed deployment instructions.

## Architecture Decision Records

See `adr/` folder for architectural decisions:

- `001-modular-architecture-with-rag-and-evaluator.md` - Main architecture
- `002-technology-stack-selection.md` - Tech stack choices
- `003-memory-management-strategy.md` - Memory system design

## License

MIT

## Contributing

This is a learning project following the implementation plan in `plan/MVP_PLAN.md`.

## Acknowledgments

Built as part of the AI Agents course, demonstrating:
- Clean code principles (SOLID, DRY, KISS)
- Agentic AI design patterns
- Test-driven development
- Modular architecture

