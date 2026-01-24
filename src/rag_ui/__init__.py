"""
RAG Question UI

Simple web interface for asking questions against ingested SEC filings.
Uses Streamlit for the chat interface.
"""

from src.rag_ui.config import RAGUIConfig
from src.rag_ui.query_handler import QueryResult, RAGQueryHandler

__all__ = ["RAGUIConfig", "RAGQueryHandler", "QueryResult"]
