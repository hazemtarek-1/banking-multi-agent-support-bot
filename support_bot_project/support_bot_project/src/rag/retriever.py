# AI-GENERATED SECTION START
"""Local FAISS RAG index using Sentence Transformers embeddings."""
from __future__ import annotations

from pathlib import Path

KB_DIR = Path("knowledge_base")
INDEX_DIR = Path("artifacts/faiss_index")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index() -> None:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = DirectoryLoader(str(KB_DIR), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    if len(docs) < 10:
        raise RuntimeError("The assignment asks for 10-30 KB documents; add at least 10 markdown files.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, _embeddings())
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Indexed {len(docs)} documents into {len(chunks)} chunks at {INDEX_DIR}")


def get_retriever():
    from langchain_community.vectorstores import FAISS

    if not (INDEX_DIR / "index.faiss").exists():
        build_index()
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


if __name__ == "__main__":
    build_index()
# AI-GENERATED SECTION END
