# agent/graph.py — new file, doesn't touch agent/chain.py
import os, uuid
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

from retrieval.retriever import retrieve
from core.schemas import SourceChunk


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are a WhatsApp support assistant.

Use search_docs to find information before answering factual questions.
Use create_support_ticket only when search_docs results genuinely don't answer
the question — do not guess or invent steps.
If a question is ambiguous in a way that changes the answer (e.g. chat transfer
steps differ for iPhone vs Android), ask the user directly in plain text instead
of guessing — wait for their reply."""


@tool
def search_docs(query: str) -> str:
    """Search WhatsApp help documentation for relevant information."""
    chunks: list[SourceChunk] = retrieve(query)
    if not chunks:
        return "No relevant documentation found."
    return "\n\n".join(f"[Source: {c.source}]\n{c.content}" for c in chunks)


@tool
def create_support_ticket(reason: str, original_query: str) -> str:
    """Escalate to a human agent when documentation doesn't answer the question."""
    ticket_id = str(uuid.uuid4())[:8]
    print(f"[TICKET {ticket_id}] {original_query} — {reason} — {datetime.now(timezone.utc).isoformat()}")
    return f"Ticket {ticket_id} created. A human agent will follow up."


tools = [search_docs, create_support_ticket]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.environ["GITHUB_TOKEN"],
    openai_api_base="https://models.inference.ai.azure.com",
    temperature=0,
).bind_tools(tools)


def call_model(state: AgentState):
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

checkpointer = MemorySaver()  # in-memory, resets on restart 
app = graph.compile(checkpointer=checkpointer)


@traceable(name="support_agent_graph")
def run(query: str, session_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke({"messages": [("user", query)]}, config=config)
    return {"query": query, "answer": result["messages"][-1].content, "session_id": session_id}