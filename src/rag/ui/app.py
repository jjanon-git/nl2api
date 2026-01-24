"""
SEC Filing Q&A - Streamlit Application

Chat interface for asking questions about ingested SEC filings.

Run with:
    streamlit run src/rag/ui/app.py
    # or
    python scripts/run_rag_ui.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file before other imports that depend on environment variables
from dotenv import load_dotenv  # noqa: I001

load_dotenv(PROJECT_ROOT / ".env")

import asyncpg
import streamlit as st

from src.rag.ui.config import RAGUIConfig
from src.rag.ui.query_handler import RAGQueryHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config() -> RAGUIConfig:
    """Get configuration, loading from environment."""
    anthropic_key = (
        os.getenv("RAG_UI_ANTHROPIC_API_KEY")
        or os.getenv("NL2API_ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or ""
    )
    openai_key = (
        os.getenv("RAG_UI_OPENAI_API_KEY")
        or os.getenv("NL2API_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    database_url = (
        os.getenv("RAG_UI_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql://nl2api:nl2api@localhost:5432/nl2api"
    )

    return RAGUIConfig(
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
        database_url=database_url,
    )


async def get_stats_async(config: RAGUIConfig) -> dict:
    """Get system stats with a fresh connection."""
    pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            sec_count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_documents WHERE document_type = 'sec_filing'"
            )
            companies = await conn.fetch(
                """
                SELECT DISTINCT metadata->>'ticker' as ticker
                FROM rag_documents
                WHERE document_type = 'sec_filing'
                AND metadata->>'ticker' IS NOT NULL
                """
            )
        return {
            "sec_filing_chunks": sec_count,
            "unique_companies": len(companies),
            "companies": sorted([r["ticker"] for r in companies if r["ticker"]]),
        }
    finally:
        await pool.close()


async def query_async(config: RAGUIConfig, question: str, top_k: int) -> dict:
    """Run a query with a fresh connection pool."""
    pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=3)
    try:
        handler = RAGQueryHandler(pool=pool, config=config)
        await handler.initialize()
        result = await handler.query(question, top_k=top_k)
        return {
            "answer": result.answer,
            "sources": [
                {
                    "content": s.content,
                    "score": s.score,
                    "metadata": s.metadata,
                }
                for s in result.sources
            ],
        }
    finally:
        await pool.close()


def render_source(source: dict, index: int):
    """Render a source chunk in an expandable section."""
    metadata = source.get("metadata") or {}
    ticker = metadata.get("ticker", "Unknown")
    filing_type = metadata.get("filing_type", "10-K")
    fiscal_year = metadata.get("fiscal_year", "")
    section = metadata.get("section", "")

    title = f"{ticker} {filing_type}"
    if fiscal_year:
        title += f" FY{fiscal_year}"
    if section:
        title += f" - {section}"
    title += f" (Score: {source.get('score', 0):.2f})"

    with st.expander(f"Source {index}: {title}"):
        st.markdown(source.get("content", ""))


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="SEC Filing Q&A",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 SEC Filing Q&A")
    st.markdown(
        "Ask questions about SEC 10-K and 10-Q filings. "
        "Answers are generated from retrieved filing excerpts."
    )

    config = get_config()

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")

        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources may provide better context but slower responses",
        )

        show_sources = st.checkbox(
            "Show source excerpts",
            value=True,
            help="Display the SEC filing excerpts used to generate the answer",
        )

        st.divider()

        # System status
        st.header("System Status")
        try:
            stats = asyncio.run(get_stats_async(config))

            st.success("Connected")
            st.metric("SEC Filing Chunks", f"{stats['sec_filing_chunks']:,}")
            st.metric("Companies", stats["unique_companies"])

            if stats.get("companies"):
                with st.expander("Available Companies"):
                    st.write(", ".join(stats["companies"][:50]))
                    if len(stats["companies"]) > 50:
                        st.write(f"... and {len(stats['companies']) - 50} more")

        except Exception as e:
            st.error(f"Connection error: {e}")
            st.info(
                "Make sure PostgreSQL is running and SEC filings are ingested. "
                "See scripts/ingest_sec_filings.py"
            )
            return

        st.divider()
        st.caption(f"Model: {config.llm_model}")
        st.caption(f"Embedder: {config.embedding_provider}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message.get("sources") and show_sources:
                st.divider()
                st.caption(f"Based on {len(message['sources'])} source(s):")
                for i, source in enumerate(message["sources"], 1):
                    render_source(source, i)

    # Chat input
    if prompt := st.chat_input("Ask about SEC filings..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching SEC filings..."):
                try:
                    result = asyncio.run(query_async(config, prompt, top_k))

                    st.write(result["answer"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"] if show_sources else None,
                        }
                    )

                    if show_sources and result["sources"]:
                        st.divider()
                        st.caption(f"Based on {len(result['sources'])} source(s):")
                        for i, source in enumerate(result["sources"], 1):
                            render_source(source, i)

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    logger.exception("Query failed")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                        }
                    )

    # Example questions
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("**Example questions to try:**")

        examples = [
            "What are Apple's main risk factors?",
            "What is Microsoft's total revenue?",
            "How does Tesla describe its competitive advantages?",
            "What are the key business segments for Amazon?",
            "What does Google say about AI in their filings?",
        ]

        cols = st.columns(len(examples))
        for col, example in zip(cols, examples):
            if col.button(example[:25] + "...", key=example):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


if __name__ == "__main__":
    main()
