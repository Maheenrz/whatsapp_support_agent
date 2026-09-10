import os
from openai import OpenAI
from langsmith import traceable
from dotenv import load_dotenv

from retrieval.retriever import retrieve
from core.schemas import SourceChunk

load_dotenv()

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

SYSTEM_PROMPT = """You are a WhatsApp support assistant. Answer the user's question 
using ONLY the provided context below. If the context does not contain the answer, 
say you don't have that information — do not guess or invent steps."""


def format_context(chunks: list[SourceChunk]) -> str:
    return "\n\n".join(f"[Source: {c.source}]\n{c.content}" for c in chunks)


@traceable(name="generate_answer")
def generate_answer(query: str, chunks: list[SourceChunk]) -> str:
    context = format_context(chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content



@traceable(name="support_agent_chain")
def run(query: str)-> dict:
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)
    return {
        "query": query,
        "answer": answer,
        "sources": [c.source for c in chunks],
        "context_text": [c.content for c in chunks],
    }


if __name__ == "__main__":
    result = run("how do i backup my chats")
    print(result["answer"])
    print("\nSources:", result["sources"])