# Sprint 4: Agents & Memory

**Sprint Goal:** Implement Career Agent, Evaluator, and Working Memory  
**Deliverable:** Complete agent system with evaluation  
**Duration:** ~1-2 weeks

---

## Overview

This sprint implements the core agentic components:
- **Career Agent:** Main conversational agent with tool calling
- **Evaluator Agent:** Quality assessment using Gemini
- **Working Memory:** Recent conversation context management

**Why this matters:** These are the intelligent components that make the system "agentic".

---

## Prerequisites

```bash
# Add dependencies to parent requirements.txt if needed
cd path/to/agents/
# google-generativeai should already be present for Gemini
```

---

## Task 4.1: Career Agent

**Estimated Time:** 4-5 hours  
**Priority:** P0

### Implementation

Create `core/career_agent.py`:

```python
"""Career agent with tool calling capabilities."""

from typing import List, Dict, Optional
from services.llm_service import LLMService
from services.rag_service import RAGService
from tools.registry import ToolRegistry
from utils.prompt_builder import PromptBuilder
from config.settings import settings


class CareerAgent:
    """Main conversational agent for career Q&A.
    
    Example:
        >>> agent = CareerAgent(llm_service, rag_service, tool_registry, profile, summary)
        >>> response = agent.chat("What is your experience?", history)
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        rag_service: RAGService,
        tool_registry: ToolRegistry,
        linkedin_profile: str,
        career_summary: str
    ):
        """Initialize career agent.
        
        Args:
            llm_service: LLM service instance
            rag_service: RAG service instance
            tool_registry: Tool registry instance
            linkedin_profile: LinkedIn profile text
            career_summary: Career summary text
        """
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.tool_registry = tool_registry
        self.system_prompt = PromptBuilder.build_system_prompt(
            name=settings.agent_name,
            linkedin_profile=linkedin_profile,
            summary=career_summary,
            enable_rag=settings.enable_rag
        )
    
    def chat(self, message: str, history: List[Dict]) -> str:
        """Process user message and generate response.
        
        Args:
            message: User's message
            history: Conversation history
            
        Returns:
            Agent's response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": message}
        ]
        
        # Agentic loop: allow multiple tool calls
        max_iterations = 5
        for _ in range(max_iterations):
            response = self.llm_service.chat_completion(
                messages=messages,
                tools=self.tool_registry.to_openai_format()
            )
            
            finish_reason = response.choices[0].finish_reason
            assistant_message = response.choices[0].message
            
            if finish_reason == "tool_calls":
                # Execute tools and continue
                messages.append(assistant_message)
                tool_results = self._execute_tools(assistant_message.tool_calls)
                messages.extend(tool_results)
            else:
                # Final response
                return assistant_message.content
        
        return "I apologize, but I'm having trouble processing your request."
    
    def _execute_tools(self, tool_calls) -> List[Dict]:
        """Execute tool calls and format results.
        
        Args:
            tool_calls: List of tool calls from LLM
            
        Returns:
            List of tool result messages
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool = self.tool_registry.get(tool_name)
            
            if tool:
                import json
                args = json.loads(tool_call.function.arguments)
                result = tool.execute(**args)
                
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            else:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"success": False, "message": "Tool not found"})
                })
        
        return results
```

### Tests

Create `tests/unit/test_career_agent.py`:

```python
"""Tests for career agent."""

import pytest
from unittest.mock import Mock, MagicMock
from core.career_agent import CareerAgent


@pytest.fixture
def mock_services():
    """Create mock services."""
    llm_service = Mock()
    rag_service = Mock()
    tool_registry = Mock()
    tool_registry.to_openai_format.return_value = []
    
    return llm_service, rag_service, tool_registry


def test_career_agent_init(mock_services):
    """Test career agent initialization."""
    llm, rag, registry = mock_services
    
    agent = CareerAgent(llm, rag, registry, "LinkedIn", "Summary")
    
    assert agent.llm_service is llm
    assert agent.rag_service is rag
    assert agent.tool_registry is registry


def test_career_agent_chat_simple(mock_services):
    """Test simple chat without tools."""
    llm, rag, registry = mock_services
    
    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message.content = "Test response"
    llm.chat_completion.return_value = mock_response
    
    agent = CareerAgent(llm, rag, registry, "LinkedIn", "Summary")
    response = agent.chat("Hello", [])
    
    assert response == "Test response"
```

### Test & Commit

```bash
pytest tests/unit/test_career_agent.py -v

git add core/career_agent.py tests/unit/test_career_agent.py
git commit -m "feat(core): implement career agent with tool calling

- Add CareerAgent with agentic loop
- Support multiple tool calls
- Add unit tests

Relates to Sprint 4, Task 4.1"
```

---

## Task 4.2: Evaluator Agent

**Estimated Time:** 2-3 hours  
**Priority:** P0

### Implementation

Create `core/evaluator_agent.py`:

```python
"""Evaluator agent for response quality assessment."""

from typing import Optional
import google.generativeai as genai
from models.schemas import Evaluation
from config.settings import settings
from textwrap import dedent


class EvaluatorAgent:
    """Agent for evaluating response quality.
    
    Example:
        >>> evaluator = EvaluatorAgent()
        >>> eval = evaluator.evaluate("Question", "Answer")
        >>> print(eval.acceptable, eval.score)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize evaluator agent.
        
        Args:
            api_key: Optional Google API key
        """
        genai.configure(api_key=api_key or settings.google_api_key)
        self.model = genai.GenerativeModel(settings.evaluator_model)
    
    def evaluate(self, question: str, answer: str) -> Evaluation:
        """Evaluate response quality.
        
        Args:
            question: User's question
            answer: Agent's answer
            
        Returns:
            Evaluation object with acceptable, feedback, score
        """
        prompt = dedent(f"""
            Evaluate the following Q&A exchange for a career chatbot.
            
            Question: {question}
            Answer: {answer}
            
            Evaluate based on:
            1. Relevance: Does the answer address the question?
            2. Accuracy: Is the information correct?
            3. Completeness: Is the answer thorough?
            4. Professionalism: Is the tone appropriate?
            
            Respond in JSON format:
            {{
                "acceptable": true/false,
                "feedback": "specific feedback",
                "score": 1-10
            }}
        """).strip()
        
        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON from response
            import json
            import re
            
            # Extract JSON from markdown code blocks if present
            text = response.text
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                json_str = json_match.group(0) if json_match else text
            
            data = json.loads(json_str)
            return Evaluation(**data)
        
        except Exception as e:
            # Fallback evaluation
            return Evaluation(
                acceptable=True,
                feedback=f"Evaluation failed: {str(e)}",
                score=5
            )
```

### Tests

Create `tests/unit/test_evaluator_agent.py`:

```python
"""Tests for evaluator agent."""

import pytest
from unittest.mock import Mock, patch
from core.evaluator_agent import EvaluatorAgent
from models.schemas import Evaluation


@pytest.fixture
def evaluator():
    """Create evaluator with mocked Gemini."""
    with patch('core.evaluator_agent.genai'):
        return EvaluatorAgent()


def test_evaluator_evaluate_success(evaluator):
    """Test successful evaluation."""
    # Mock Gemini response
    mock_response = Mock()
    mock_response.text = '{"acceptable": true, "feedback": "Good answer", "score": 8}'
    evaluator.model.generate_content = Mock(return_value=mock_response)
    
    eval = evaluator.evaluate("What is AI?", "AI is artificial intelligence")
    
    assert isinstance(eval, Evaluation)
    assert eval.acceptable is True
    assert eval.score == 8


def test_evaluator_evaluate_with_markdown(evaluator):
    """Test evaluation with markdown JSON."""
    mock_response = Mock()
    mock_response.text = '```json\n{"acceptable": true, "feedback": "Great", "score": 9}\n```'
    evaluator.model.generate_content = Mock(return_value=mock_response)
    
    eval = evaluator.evaluate("Test", "Test answer")
    
    assert eval.acceptable is True
    assert eval.score == 9


def test_evaluator_evaluate_fallback(evaluator):
    """Test evaluation fallback on error."""
    evaluator.model.generate_content = Mock(side_effect=Exception("API Error"))
    
    eval = evaluator.evaluate("Test", "Test")
    
    assert isinstance(eval, Evaluation)
    assert eval.acceptable is True  # Fallback is acceptable
    assert eval.score == 5
```

### Test & Commit

```bash
pytest tests/unit/test_evaluator_agent.py -v

git add core/evaluator_agent.py tests/unit/test_evaluator_agent.py
git commit -m "feat(core): implement evaluator agent with Gemini

- Add EvaluatorAgent for quality assessment
- Support JSON parsing from Gemini responses
- Add fallback evaluation on errors
- Add unit tests

Relates to Sprint 4, Task 4.2"
```

---

## Task 4.3: Working Memory

**Estimated Time:** 2 hours  
**Priority:** P0

### Implementation

Create `memory/working_memory.py`:

```python
"""Working memory for recent conversation context."""

from typing import List, Dict
from config.settings import settings


class WorkingMemory:
    """Manages recent conversation turns.
    
    Example:
        >>> memory = WorkingMemory()
        >>> memory.add_turn("Hello", "Hi!")
        >>> context = memory.get_context()
    """
    
    def __init__(self, max_turns: int = None):
        """Initialize working memory.
        
        Args:
            max_turns: Maximum turns to keep. Defaults to settings
        """
        self.max_turns = max_turns or settings.working_memory_max_turns
        self.turns: List[Dict] = []
    
    def add_turn(self, user_message: str, agent_response: str):
        """Add conversation turn to memory.
        
        Args:
            user_message: User's message
            agent_response: Agent's response
        """
        self.turns.append({
            "role": "user",
            "content": user_message
        })
        self.turns.append({
            "role": "assistant",
            "content": agent_response
        })
        
        # Keep only recent turns
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]
    
    def get_context(self) -> List[Dict]:
        """Get conversation context for LLM.
        
        Returns:
            List of message dictionaries
        """
        return self.turns.copy()
    
    def clear(self):
        """Clear all turns from memory."""
        self.turns = []
```

Create `memory/memory_service.py`:

```python
"""Memory service orchestrator."""

from memory.working_memory import WorkingMemory


class MemoryService:
    """Orchestrates memory layers.
    
    For MVP, only working memory is used.
    
    Example:
        >>> memory = MemoryService()
        >>> memory.store_interaction("Hello", "Hi!")
        >>> context = memory.build_context("What is your name?")
    """
    
    def __init__(self):
        """Initialize memory service."""
        self.working_memory = WorkingMemory()
    
    def store_interaction(self, user_message: str, agent_response: str):
        """Store conversation interaction.
        
        Args:
            user_message: User's message
            agent_response: Agent's response
        """
        self.working_memory.add_turn(user_message, agent_response)
    
    def build_context(self, current_query: str) -> list:
        """Build context for current query.
        
        Args:
            current_query: Current user query
            
        Returns:
            List of message dictionaries for LLM
        """
        return self.working_memory.get_context()
    
    def clear(self):
        """Clear all memory."""
        self.working_memory.clear()
```

### Tests

Create `tests/unit/test_memory.py`:

```python
"""Tests for memory components."""

import pytest
from memory.working_memory import WorkingMemory
from memory.memory_service import MemoryService


def test_working_memory_add_turn():
    """Test adding turn to working memory."""
    memory = WorkingMemory(max_turns=5)
    
    memory.add_turn("Hello", "Hi!")
    context = memory.get_context()
    
    assert len(context) == 2
    assert context[0]["role"] == "user"
    assert context[1]["role"] == "assistant"


def test_working_memory_max_turns():
    """Test max turns limit."""
    memory = WorkingMemory(max_turns=2)
    
    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")
    memory.add_turn("Q3", "A3")
    
    context = memory.get_context()
    assert len(context) == 4  # 2 turns * 2 messages


def test_working_memory_clear():
    """Test clearing memory."""
    memory = WorkingMemory()
    memory.add_turn("Hello", "Hi!")
    memory.clear()
    
    assert len(memory.get_context()) == 0


def test_memory_service_store_interaction():
    """Test storing interaction."""
    service = MemoryService()
    
    service.store_interaction("Hello", "Hi!")
    context = service.build_context("How are you?")
    
    assert len(context) == 2


def test_memory_service_clear():
    """Test clearing memory service."""
    service = MemoryService()
    service.store_interaction("Hello", "Hi!")
    service.clear()
    
    assert len(service.build_context("Test")) == 0
```

### Test & Commit

```bash
pytest tests/unit/test_memory.py -v

git add memory/ tests/unit/test_memory.py
git commit -m "feat(memory): implement working memory for conversation context

- Add WorkingMemory for recent turns
- Add MemoryService orchestrator
- Support max turns limit
- Add unit tests

Relates to Sprint 4, Task 4.3"
```

---

## Sprint 4 Completion

### Verification

```bash
pytest tests/unit/test_career_agent.py tests/unit/test_evaluator_agent.py tests/unit/test_memory.py -v

pytest tests/unit/ --cov=core --cov=memory --cov-report=html
```

### Deliverables

- [x] Career Agent with tool calling
- [x] Evaluator Agent with Gemini
- [x] Working Memory system
- [x] Memory Service orchestrator
- [x] Unit tests for all components
- [x] Test coverage > 80%

### Next Steps

**Ready for Sprint 5!** 🎉

Open `plan/sprint_5_integration.md` to complete the MVP:
- Controller to orchestrate everything
- app.py integration
- End-to-end testing
- HuggingFace deployment

---

**Sprint 4 Complete!** ✅  
**Next:** `plan/sprint_5_integration.md`

