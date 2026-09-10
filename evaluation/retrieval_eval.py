"""
Baseline retrieval eval. Run BEFORE deciding on reranking, routing,
or any query translation technique. The numbers here decide what gets built next.
"""

from retrieval.retriever import retrieve

# (query, expected_source_filename)
# Written the way a real user would type — NOT copied from doc titles.
# Deliberately includes near-duplicate pairs predicted to fail.
EVAL_SET = [
    ("how do i turn on two step verification", "about_two_step_verification.md"),
    ("tips to keep my account secure", "account_security_tips.md"),
    ("how do i save a copy of my chats", "back_up_chat_history.md"),
    ("my backup keeps failing what do i do", "cant_backup_or_restore_chat_history.md"),
    ("how do i download my chat history as a file", "export_chat_history.md"),
    ("someone hacked my whatsapp how do i get it back", "recover_compromised_account.md"),
    ("general safety tips for whatsapp", "stay_safe_on_whatsapp.md"),
    ("moving my chats to a new samsung phone", "transfer_chat_history_android.md"),
    ("how to move whatsapp chats to my new iphone", "transfer_chat_history_iphone.md"),
    ("why is my backup not working", "cant_backup_or_restore_chat_history.md"),
    ("how do i move my chats to a new phone and i have an iphone", "transfer_chat_history_iphone.md"),
    ("what are the steps to move my chats to a new phone and i have an android", "transfer_chat_history_android.md"),
    ("how do i restore my chat history from a backup", "cant_backup_or_restore_chat_history.md"),
    ("i have given instagram access to my whatsapp account how do i remove it", "account_security_tips.md"),
    ("how do i save my chats so i don't lose them", "back_up_chat_history.md"),
    ("how to save a file of my messages", "export_chat_history.md"),
    ("i want to keep a copy of my messages off whatsapp", "export_chat_history.md"),
]


def hit_rate_at_k(k: int = 3):
    hits = 0
    for query, expected in EVAL_SET:
        results = retrieve(query, top_k=k)
        retrieved_sources = [r.source for r in results]
        hit = expected in retrieved_sources
        hits += hit
        status = "HIT " if hit else "MISS"
        print(f"[{status}] '{query}'")
        print(f"       expected: {expected}")
        print(f"       got:      {retrieved_sources}\n")

    print(f"Hit rate @ {k}: {hits}/{len(EVAL_SET)} = {hits/len(EVAL_SET):.0%}")



def mrr_at_k(k: int = 3):  # mean reciprocal rank -> tells how high a doc rank
    reciprocal_ranks = []
    for query, expected in EVAL_SET:
        results = retrieve(query, top_k=k)
        retrieved_sources = [r.source for r in results]
        if expected in retrieved_sources:
            rank = retrieved_sources.index(expected) + 1  # finding the position of the source using .index() - 1-indexed
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"MRR @ {k}: {mrr:.3f}")
    return mrr

if __name__ == "__main__":
    hit_rate_at_k(k=3)
    mrr_at_k(k=3)