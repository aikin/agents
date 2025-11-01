import shutil
import tempfile

import pytest

from services.rag_service import RAGService


@pytest.fixture
def rag_service():
    temp_dir = tempfile.mkdtemp()
    service = RAGService(persist_directory=temp_dir)
    yield service
    shutil.rmtree(temp_dir)


def test_ingest_and_search(rag_service):
    rag_service.ingest_text("I have 10 years of experience in software engineering.", "doc1")

    results = rag_service.search("experience")
    assert len(results) > 0
    assert "experience" in results[0].lower()


def test_search_no_results(rag_service):
    results = rag_service.search("nonexistent query")
    assert results == []


def test_clear(rag_service):
    rag_service.ingest_text("Test document", "doc1")
    rag_service.clear()

    results = rag_service.search("Test")
    assert results == []
