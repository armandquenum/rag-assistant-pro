"""
Package SRC : Core logic du pipeline Advanced RAG.
Regroupe les modules de chargement, découpage, indexation et recherche.
"""

from .document_loader import DocumentLoader
from .text_splitter import SemanticParentChildSplitter
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .reranker import Reranker
from .retriever import Retriever
from .rag_pipeline import RAGPipeline
from .hyde import HyDE
from .multi_query import MultiQueryGenerator


# Définit ce qui est exposé lors d'un "from src import *"
__all__ = [
    "DocumentLoader",
    "SemanticParentChildSplitter",
    "EmbeddingModel",
    "VectorStore",
    "Reranker",
    "Retriever",
    "RAGPipeline",
    "HyDE",
    "MultiQueryGenerator",
]