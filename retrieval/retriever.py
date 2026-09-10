import chromadb
from chromadb.utils import embedding_functions

from flashrank import Ranker, RerankRequest

from core.schemas import SourceChunk

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "support_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ENABLE_RERANKER = False

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            configuration={"hnsw": {"space": "cosine"}},
        )
    return _collection

_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


def retrieve(query: str, top_k: int = 3, candidate_k: int = 10) -> list[SourceChunk]:
    collection = get_collection()
    n = candidate_k if ENABLE_RERANKER else top_k
    results = collection.query(query_texts=[query], n_results=n)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not ENABLE_RERANKER:
        chunks = []
        for doc, meta, dist in zip(docs, metas, dists):
            similarity = max(0.0, min(1.0, 1 - dist))
            chunks.append(SourceChunk(content=doc, source=meta["source"], score=similarity))
        return chunks

    passages = [{"id": i, "text": doc, "meta": meta} for i, (doc, meta) in enumerate(zip(docs, metas))]
    ranker = get_ranker()
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))

    return [
        SourceChunk(content=item["text"], source=item["meta"]["source"], score=item["score"])
        for item in reranked[:top_k]
    ]    