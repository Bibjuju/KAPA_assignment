from pathlib import Path

import streamlit as st

from src.agent.rag_agent import RAGAgent
from src.chunker.markdown_section_chunker import MarkdownSectionChunker
from src.converter.pymu import PymuConverter
from src.loader.pdf_loader import DirectoryPDFLoader
from src.vector_store.in_memory import InMemoryVectorStore

DATA_DIR = Path(__file__).parent.parent / "data" / "pdfs"

# ────────────────────────────────────────────────────────────────
# Page config (browser-tab title stays constant)
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chat over PDFs", layout="wide")

# ────────────────────────────────────────────────────────────────
# Session-state defaults
# ────────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "Browse"
if "agent" not in st.session_state:
    st.session_state.agent = RAGAgent(
        loader=DirectoryPDFLoader(DATA_DIR),
        converter=PymuConverter(),
        chunker=MarkdownSectionChunker(),
        store=InMemoryVectorStore(),
    )
agent = st.session_state.agent  # shorthand

# ────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    # One-off toast after indexing
    if "index_msg" in st.session_state:
        st.success(st.session_state.index_msg)
        del st.session_state.index_msg

    # Index / reset actions — show ONE button at a time
    if agent.docs:
        if st.button("♻️ Reset index", use_container_width=True):
            st.session_state.agent = RAGAgent(
                loader=DirectoryPDFLoader(DATA_DIR),
                converter=PymuConverter(),
                chunker=MarkdownSectionChunker(),
                store=InMemoryVectorStore(),
            )
            st.session_state.mode = "Browse"
            st.rerun()
    else:
        if st.button("📄 Load & index PDFs", use_container_width=True):
            agent.index()
            st.session_state.index_msg = f"Indexed {len(agent.docs)} PDF(s)"
            st.session_state.mode = "Browse"
            st.rerun()  # → sidebar immediately shows Reset button

    st.markdown("---")

    # Navigation – stacked buttons (Browse first)
    if st.button("📚 Browse", key="nav_browse", use_container_width=True):
        st.session_state.mode = "Browse"
    if st.button("💬 Chat", key="nav_chat", use_container_width=True):
        st.session_state.mode = "Chat"

# ────────────────────────────────────────────────────────────────
# Dynamic page title (rendered *after* sidebar)
# ────────────────────────────────────────────────────────────────
if st.session_state.mode == "Chat":
    st.title("💬 Chat over indexed PDFs")
else:
    st.title("📚 Browse indexed PDFs")

# ────────────────────────────────────────────────────────────────
# Mode A: Chat
# ────────────────────────────────────────────────────────────────
if st.session_state.mode == "Chat":

    if not agent.docs:
        st.info("Index some PDFs first.")
    else:
        # 🆕 Explanation of how Chat works
        st.info(
            "When you ask a question, the app first **retrieves** the most chunks from the indexed PDFs "
            "sections using semantic search, then it **generates** an answer from those "
            "top-ranked chunks.\n\n"
            "* The assistant’s answer appears above.\n"
            "* The exact chunks that were used are listed underneath."
        )

        query = st.chat_input("Ask a question …")

        if query:
            answer, chunks = agent.answer(query)
            
            st.chat_message("user").markdown(query)
            st.chat_message("assistant").markdown(answer)

            if chunks:
                sources_md = "\n\n---\n\n".join(
                    f"**Chunk {i+1} (score ≈ {score:.2f})**\n\n{chunk}"
                    for i, (chunk, score) in enumerate(chunks)
                )
                st.markdown(f"---\n\n### Source chunks\n\n{sources_md}")

# ────────────────────────────────────────────────────────────────
# Mode B: Browse converted Markdown
# ────────────────────────────────────────────────────────────────
else:
    if not agent.docs:
        st.info("Index some PDFs first.")
    else:
        st.info(
            "This view lets you inspect the output of the PDF→Markdown conversion **and** "
            "see how that Markdown was chunked for retrieval:\n\n"
            "• **View Markdown** – full converted document.\n\n"
            "• **View Chunks** – the smaller sections the chat engine actually searches."
        )

        fname = st.selectbox(
            "Select a document",
            list(agent.docs.keys()),
            key="doc_select",
        )
        doc = agent.docs[fname]

        with st.expander("View Markdown", expanded=True):
            st.markdown(doc.markdown)

        with st.expander("View Chunks", expanded=True):
            chunk_string = "\n\n---\n\n".join([c.content for c in doc.chunks])
            st.markdown(chunk_string)
