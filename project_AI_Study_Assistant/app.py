import re
import shutil
from pathlib import Path
from typing import List

import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "store"
INDEX_FILE = STORE_DIR / "index.faiss"
METADATA_FILE = STORE_DIR / "index.pkl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="AI-Powered Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("AI-Powered Study Assistant")
st.markdown(
    """
Upload PDF or TXT files, build a local FAISS index, and ask questions about your study material.

This version works without any API key. It uses document retrieval plus local text extraction only.
"""
)


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_documents_from_folder(folder_path: Path):
    documents = []

    if not folder_path.exists():
        return documents

    for file_path in folder_path.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
        except Exception as exc:
            st.warning(f"Skipping {file_path.name}: {exc}")

    return documents


def load_uploaded_documents(uploaded_files) -> List[str]:
    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = DATA_DIR / Path(uploaded_file.name).name
        with open(file_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))

    return saved_paths


def build_vector_store():
    documents = load_documents_from_folder(DATA_DIR)

    if not documents:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        return None, 0

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(STORE_DIR))

    return vectorstore, len(chunks)


def load_vector_store():
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        return None

    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def clear_directory_contents(directory: Path):
    if not directory.exists():
        return

    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def clear_workspace():
    clear_directory_contents(STORE_DIR)
    clear_directory_contents(DATA_DIR)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def build_short_snippet(text: str, limit: int = 1000) -> str:
    snippet = normalize_text(text)
    if len(snippet) <= limit:
        return snippet
    return snippet[:limit].rsplit(" ", 1)[0] + "..."


def answer_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": [],
        }

    question_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", question)
        if len(term) > 2
    }

    scored_sentences = []

    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        for sentence in sentences:
            clean_sentence = normalize_text(sentence)
            if not clean_sentence:
                continue

            score = sum(
                1 for term in question_terms if term in clean_sentence.lower()
            )
            if score > 0:
                scored_sentences.append((score, clean_sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    selected_sentences = []
    seen_sentences = set()

    for _, sentence in scored_sentences:
        if sentence in seen_sentences:
            continue
        selected_sentences.append(sentence)
        seen_sentences.add(sentence)
        if len(selected_sentences) == 3:
            break

    if selected_sentences:
        answer_text = "Based on the most relevant text I found:\n\n"
        answer_text += "\n\n".join(f"- {sentence}" for sentence in selected_sentences)
    else:
        answer_text = (
            "I could not extract a direct sentence match, so here is the closest "
            "passage from your documents:\n\n"
            f"{build_short_snippet(docs[0].page_content)}"
        )

    return {
        "answer": answer_text,
        "sources": docs,
    }


with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Build Index"):
            if uploaded_files:
                load_uploaded_documents(uploaded_files)

            vectorstore, chunk_count = build_vector_store()
            if vectorstore is not None:
                st.success(f"Vector store created with {chunk_count} chunks.")
            else:
                st.error("No valid documents found to index.")

    with col2:
        if st.button("Clear Store"):
            clear_workspace()
            st.cache_resource.clear()
            st.success("Vector store and uploaded documents cleared.")

    st.markdown("---")
    st.caption("Upload files, build the index, then ask questions from the main page.")


vectorstore = load_vector_store()

if vectorstore is None:
    st.info("No vector store found. Upload documents and build the index first.")
else:
    st.success("Vector store loaded successfully.")

    question = st.text_input(
        "Ask a question about your study material",
        placeholder="What is the main idea of the document?",
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your documents..."):
                result = answer_question(vectorstore, question)

            st.subheader("Answer")
            st.write(result["answer"])

            if result["sources"]:
                st.subheader("Sources")
                for i, source in enumerate(result["sources"], start=1):
                    with st.expander(f"Source {i}"):
                        st.write(build_short_snippet(source.page_content, 2000))
                        if source.metadata:
                            st.json(source.metadata)

st.markdown("---")
st.markdown("Built for local retrieval with Streamlit and FAISS.")