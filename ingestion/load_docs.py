import hashlib
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


from retrieval.retriever import get_collection

DOCS_DIR = "data/"          
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def load_raw_docs() -> list[dict]:
    """Read every .md file in DOCS_DIR into memory as {content, source}."""
    docs = []
    for filepath in Path(DOCS_DIR).glob("*.md"):
        text = filepath.read_text(encoding="utf-8").strip()
        docs.append({"content": text, "source": filepath.name})
    return docs


def chunk_document(doc: dict)->list[dict]:
    text = _splitter.split_text(doc["content"])
    return [
        {"content": chunk_text , "source": doc["source"], "chunk_index": i}
        for i, chunk_text in enumerate(text)
    ]


def make_chunk_id(source: str, chunk_index: int) -> str:
    """
    Deterministic ID = same doc + same chunk index always produces the
    same ID. This is what makes re-running ingestion safe (upsert below).
    """
    raw = f"{source}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ingest():
    collection = get_collection()
    docs = load_raw_docs()

    all_ids, all_texts, all_metas = [], [], []
    for doc in docs:
        for chunk in chunk_document(doc):
            all_ids.append(make_chunk_id(chunk["source"], chunk["chunk_index"]))
            all_texts.append(chunk["content"])
            all_metas.append({"source": chunk["source"]})

    collection.upsert(ids=all_ids, documents=all_texts, metadatas=all_metas)
    print(f"Ingested {len(all_texts)} chunks from {len(docs)} docs")


if __name__ == "__main__":
    ingest()
