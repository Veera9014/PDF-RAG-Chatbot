
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")
st.caption("Hugging Face Embeddings + FAISS Vector Search")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

with st.spinner("Loading Hugging Face Embedding Model..."):
    embedding_model = load_embedding_model()

st.success("Embedding Model Ready!")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    reader = PdfReader(uploaded_file)

    pages = []
    for page_num, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        if content.strip():
            pages.append(content)

    full_text = "\n".join(pages)

    st.success(f"PDF uploaded! Total pages: {len(reader.pages)}")

    if not full_text.strip():
        st.error("No readable text found in this PDF.")
        st.stop()

    # Text Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(full_text)

    st.subheader("1. Text Chunking")
    st.write(f"Total chunks: {len(chunks)}")

    with st.expander("View PDF Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}**")
            st.write(chunk)

    # Generate Embeddings
    with st.spinner("Generating Hugging Face Embeddings..."):

        embeddings = embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    st.subheader("2. Hugging Face Embeddings")
    st.write("Embedding Shape:", embeddings.shape)

    # Create FAISS Index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(embeddings, dtype="float32"))

    st.subheader("3. FAISS Vector Database")
    st.success(f"FAISS Index Created! Vectors: {index.ntotal}")

    # Semantic Search
    st.subheader("4. Ask Questions")

    question = st.text_input(
        "Enter your question about this PDF:"
    )

    if st.button("Search PDF"):

        if question.strip():

            with st.spinner("Searching relevant PDF content..."):

                query_embedding = embedding_model.encode(
                    [question],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                query_embedding = np.asarray(
                    query_embedding,
                    dtype="float32"
                )

                scores, indices = index.search(
                    query_embedding,
                    min(3, len(chunks))
                )

            st.subheader("5. Relevant Search Results")

            for rank, (score, idx) in enumerate(
                zip(scores[0], indices[0]), start=1
            ):
                if idx >= 0:
                    with st.expander(
                        f"Result {rank} | Similarity: {score:.3f}",
                        expanded=True
                    ):
                        st.write(chunks[idx])

        else:
            st.warning("Please enter a question.")
