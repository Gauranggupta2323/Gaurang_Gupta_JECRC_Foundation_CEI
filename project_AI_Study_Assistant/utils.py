from pathlib import Path
from typing import List, Optional

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_documents(folder_path: Path):
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
            print(f"Skipping {file_path.name}: {exc}")

    return documents


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vector_store(documents, store_dir: Path):
    if not documents:
        return None

    chunks = split_documents(documents)
    if not chunks:
        return None

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    store_dir.mkdir(exist_ok=True)
    vectorstore.save_local(str(store_dir))
    return vectorstore


def load_vector_store(store_dir: Path):
    index_file = store_dir / "index.faiss"
    metadata_file = store_dir / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        return None

    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def clear_store(store_dir: Path):
    if not store_dir.exists():
        return

    for item in store_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            import shutil
            shutil.rmtree(item)


def get_relevant_context(vectorstore, question: str, k: int = 4) -> str:
    if vectorstore is None:
        return ""

    docs = vectorstore.similarity_search(question, k=k)
    return "\n\n".join(
        [f"Source {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )


def format_answer(question: str, context: str) -> str:
    return f"""
You are a helpful AI study assistant.
Answer the question only using the provided context.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:
""".strip()