# WhatsApp Support Agent

An agentic RAG support assistant that answers questions from WhatsApp's help documentation, escalates to a human when it can't, and asks clarifying questions when a query is ambiguous — built as a portfolio project to demonstrate production-style RAG engineering, not just a working demo.

WhatsApp's public help-center docs are used as a stand-in knowledge base, since this was built without access to a real company's proprietary support corpus.

## What makes this different from a tutorial RAG project

Every component here is evaluated, not assumed. That includes a case where the evaluation said "don't ship this" and it wasn't shipped:

- **Retrieval**: 94% hit rate @ 3, MRR@3 = 0.843, measured against a 17-query test set covering paraphrases, near-duplicates, and out-of-scope questions.
- **Generation**: faithfulness 1.0, answer relevancy 0.82, via RAGAS with an LLM judge — including a documented case where a 0.0 faithfulness score turned out to be a real corpus gap (a "how it works" doc without the matching "how to enable" doc), not a hallucination, traced down to root cause instead of guessed at.
- **Reranking**: implemented with FlashRank, measured before/after, and **disabled** after the data showed a net-neutral-to-slightly-negative effect on this corpus size. The code stays in place behind a feature flag — a documented, rejected optimization, not a hidden one.
- **Tool routing**: the agentic layer's decision-making (search vs. escalate vs. clarify) is checked against a labeled test set, not just spot-checked by eye.

Full reasoning for each decision — including the ones that got reversed — is in [`decisions.md`](./decisions.md).

## Architecture

```
                 ┌─────────────┐
   User query ──▶│    Agent    │──▶ tool_calls? ──No──▶ Final answer
                 │ (GPT-4o-mini)│         │
                 └─────────────┘        Yes
                        ▲                 │
                        │                 ▼
                        │         ┌───────────────┐
                        └─────────│     Tools     │
                                  │ search_docs   │
                                  │ create_ticket │
                                  └───────────────┘
```

The LLM decides which tool to call, if any — not a hardcoded if/else. `search_docs` queries a Chroma vector store; `create_support_ticket` escalates when the retrieved context doesn't answer the question. Conversation state persists per session via a LangGraph checkpointer, so a clarifying question ("iPhone or Android?") and the user's reply are handled as one continuous exchange.

Two entry points exist on purpose:
- `agent/chain.py` — the plain retrieve-then-generate pipeline, used by the generation eval. Kept stable and untouched.
- `agent/graph.py` — the agentic layer on top, used by the deployed API.

## Stack

| Layer | Tool |
|---|---|
| LLM | GPT-4o-mini (OpenAI API) |
| Agent orchestration | LangGraph |
| Vector store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (SentenceTransformers) |
| Reranking (available, off by default) | FlashRank (ms-marco-MiniLM-L-12-v2) |
| Evaluation | RAGAS (faithfulness, answer relevancy), custom retrieval + routing eval |
| Tracing | LangSmith |
| API | FastAPI |
| Deployment | Docker → Render |

## Project structure

```
support_agent/
├── agent/
│   ├── chain.py        # evaluated RAG core (retrieve → generate)
│   ├── graph.py         # agentic layer (tool routing, multi-turn state)
│   └── prompts.py
├── retrieval/
│   └── retriever.py     # Chroma query + optional reranking
├── core/
│   ├── config.py
│   └── schemas.py
├── ingestion/
│   └── load_docs.py
├── evaluation/
│   ├── retrieval_eval.py
│   ├── generation_eval.py
│   └── routing_eval.py
├── main.py               # FastAPI app
├── Dockerfile
└── decisions.md
```

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -e .

# .env
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=...       # optional, for tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=support-agent

python -m ingestion.load_docs      # build the vector store
uvicorn main:app --reload --port 8000
```

Interactive API docs at `http://localhost:8000/docs`.

## Evaluation

```bash
python -m evaluation.retrieval_eval    # hit rate + MRR
python -m evaluation.generation_eval   # RAGAS faithfulness + answer relevancy
python -m evaluation.routing_eval      # tool-routing accuracy
```

## API

`POST /ask`
```json
{ "query": "how do i backup my chats", "session_id": "optional-thread-id" }
```
```json
{ "query": "...", "answer": "...", "session_id": "..." }
```

## Known limitations

- Corpus is 10 source docs — small enough that some optimizations (reranking) don't show measurable benefit; documented in `decisions.md` rather than hidden.
- `MemorySaver` checkpointer is in-memory — conversation state resets on process restart. Fine for a demo, not for multi-user production without a persistent backing store.
- Single-agent tool-calling, not multi-agent — one LLM choosing between two tools, not multiple coordinating specialized agents. Stated accurately rather than oversold.