"""RAG service using Chroma vector database."""

import chromadb

from config.settings import settings


class RAGService:
    def __init__(self, persist_directory: str | None = None):
        self.persist_directory = persist_directory or settings.chroma_path
        # Use PersistentClient for better isolation in tests
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="career_knowledge", metadata={"hnsw:space": "cosine"}
        )

    def ingest_text(self, text: str, document_id: str, metadata: dict | None = None):
        # Simple chunking by paragraphs
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

        # ChromaDB requires non-empty metadata
        default_metadata = {"source": document_id}
        if metadata:
            default_metadata.update(metadata)

        for i, chunk in enumerate(chunks):
            chunk_metadata = default_metadata.copy()
            chunk_metadata["chunk_index"] = str(i)
            self.collection.add(
                documents=[chunk], ids=[f"{document_id}_chunk_{i}"], metadatas=[chunk_metadata]
            )

    def search(self, query: str, top_k: int = 3) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)

        if results and results["documents"]:
            return results["documents"][0]
        return []

    def clear(self):
        """Clear all documents from collection."""
        self.client.delete_collection("career_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="career_knowledge", metadata={"hnsw:space": "cosine"}
        )
