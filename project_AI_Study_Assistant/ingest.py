import argparse
from pathlib import Path

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_STORE_DIR = BASE_DIR / "store"


def load_documents(folder_path: Path):
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
            print(f"Skipping {file_path.name}: {exc}")

    return documents


def build_vector_store(data_dir: Path, store_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(data_dir)
    if not documents:
        print("No supported documents found in data/")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        print("No text chunks created from documents.")
        return 0

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(store_dir))

    print(f"Saved vector store with {len(chunks)} chunks to {store_dir}")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Build vector store from documents in data/"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Folder containing source documents",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default=str(DEFAULT_STORE_DIR),
        help="Folder where the FAISS index will be saved",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    store_dir = Path(args.store_dir)

    build_vector_store(data_dir, store_dir)


if __name__ == "__main__":
    main()