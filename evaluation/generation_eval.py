"""
Generation eval: faithfulness + answer relevancy via RAGAS.
100% GitHub Models (No Hugging Face required).
"""

import os
import sys
import types

# 1. Dummy patch for VertexAI import bug in RAGAS
try:
    import langchain_community.chat_models.vertexai
except ImportError:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy

from agent.chain import run
from ragas.run_config import RunConfig

run_config = RunConfig(
    max_workers=2,     # match the real concurrent-request cap, don't trust the default of 16
    timeout=120,
    max_retries=8,
    max_wait=90,
)

GEN_EVAL_QUERIES = [
    "how do i backup my chats",
    "my backup keeps failing what do i do",
    "how do i move my chats to a new iphone",
    "someone hacked my whatsapp how do i get it back",
    "how do i turn on two step verification",
]


def get_evaluator_models():
    github_token = os.environ["GITHUB_TOKEN"]
    endpoint = "https://models.inference.ai.azure.com"

    eval_llm = ChatOpenAI(
        model="gpt-4o-mini",   
        openai_api_key=github_token,
        openai_api_base=endpoint,
        max_tokens=4096,
        temperature=0,
    )

    github_embed = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=github_token,
        openai_api_base=endpoint,
    )

    return LangchainLLMWrapper(eval_llm), LangchainEmbeddingsWrapper(github_embed)

def build_eval_dataset() -> Dataset:
    questions, answers, contexts = [], [], []

    for q in GEN_EVAL_QUERIES:
        result = run(q)
        questions.append(result["query"])
        answers.append(result["answer"])

        retrieved_texts = result.get("context_text", result.get("sources", []))
        contexts.append(retrieved_texts)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })


if __name__ == "__main__":
    dataset = build_eval_dataset()
    eval_llm, eval_embeddings = get_evaluator_models()

    scores = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=eval_llm,
    embeddings=eval_embeddings,
    run_config=run_config,
    raise_exceptions=True,   
    )

    print("\n=== Generation Evaluation Results ===")
    print(scores)

    # debugging how the scores are structured
    df = scores.to_pandas()
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string())

    df = scores.to_pandas()
    print(df.columns.tolist())  

    answer_col = "response" if "response" in df.columns else "answer"
    context_col = "retrieved_contexts" if "retrieved_contexts" in df.columns else "contexts"

    for i, row in df.iterrows():
        print(f"Q: {row['user_input']}")
        print(f"faithfulness={row['faithfulness']:.2f}  relevancy={row['answer_relevancy']:.2f}")
        print(f"ANSWER: {row[answer_col]}")
        print(f"CONTEXT: {str(row[context_col])[:300]}...")
        print("-" * 70)