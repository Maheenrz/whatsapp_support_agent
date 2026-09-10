from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.chain import run

app = FastAPI(title="WhatsApp Support Agent", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest,session_id: str = "default"):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    try:
        result = run(request.query, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent error: {e}")
    return QueryResponse(query=result["query"], answer=result["answer"], sources=result["sources"])
