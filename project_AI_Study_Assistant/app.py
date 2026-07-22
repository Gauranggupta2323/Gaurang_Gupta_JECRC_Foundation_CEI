import os
import shutil
from pathlib import Path
from typing import List

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "store"
INDEX_FILE = STORE_DIR / "index.faiss"
METADATA_FILE = STORE_DIR / "index.pkl"

DATA_DIR.mkdir(exist_ok=True)
STORE_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="AI-Powered Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("AI-Powered Study Assistant")
st.markdown(
    """
The AI-Powered Study Assistant is an intelligent system that helps students interact with their study materials efficiently.
Using Retrieval-Augmented Generation, it allows users to upload documents and ask questions in natural language.
The system processes documents by extracting, chunking, and converting text into embeddings stored in a vector database.
When a query is asked, it retrieves relevant content and generates accurate, context-based answers using a language model.
This reduces manual searching, saves time, and improves learning.
The system is scalable, user-friendly, and can be extended with features like summarization, voice interaction, and personalized recommendations.
"""
)


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_documents_from_folder(folder_path: Path):
    documents = []

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
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
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


def clear_store():
    if STORE_DIR.exists():
        for item in STORE_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if ChatOpenAI is not None and api_key:
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=api_key,
        )

    return None


def answer_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "I could not find relevant information in the uploaded documents."

    context = "\n\n".join(
        [f"Source {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    llm = get_llm()

    if llm is None:
        return (
            "No language model is configured. "
            "Set OPENAI_API_KEY to enable answer generation.\n\n"
            f"Relevant context:\n{context}"
        )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa_chain({"query": question})
    return result["result"], result.get("source_documents", [])


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
            clear_store()
            st.cache_resource.clear()
            st.success("Vector store cleared.")

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
            with st.spinner("Searching and generating answer..."):
                result = answer_question(vectorstore, question)

            if isinstance(result, tuple):
                answer, sources = result
                st.subheader("Answer")
                st.write(answer)

                if sources:
                    st.subheader("Sources")
                    for i, source in enumerate(sources, start=1):
                        with st.expander(f"Source {i}"):
                            st.write(source.page_content)
                            if source.metadata:
                                st.json(source.metadata)
            else:
                st.subheader("Answer")
                st.write(result)


st.markdown("---")
st.markdown("Built for Retrieval-Augmented Generation with Streamlit and FAISS.")