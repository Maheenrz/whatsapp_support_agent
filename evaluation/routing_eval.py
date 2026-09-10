from agent.graph import app
from rag_agent import app


TEST_CASES = [
    ("how do i backup my chats", "answer"),
    ("how do i turn on two step verification", "answer"),
    ("can whatsapp read my private messages", "ticket"),
    ("how do i move my chats to a new phone", "clarify"),
]

def classify_run(query: str) -> str:
    result = app.invoke(
        {"messages": [("user", query)]},
        config={"configurable": {"thread_id": f"eval-{hash(query)}"}},
    )
    tool_names = [tc["name"] for m in result["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
    if "create_support_ticket" in tool_names:
        return "ticket"
    if "search_docs" not in tool_names:
        return "clarify"
    return "answer"

if __name__ == "__main__":
    correct = 0
    for query, expected in TEST_CASES:
        got = classify_run(query)
        correct += got == expected
        print(f"[{'PASS' if got == expected else 'FAIL'}] {query!r} — expected={expected}, got={got}")
    print(f"\nRouting accuracy: {correct}/{len(TEST_CASES)}")