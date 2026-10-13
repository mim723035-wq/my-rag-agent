import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from search import SearchTool, tokenize


def main():
    data_dir = ROOT / "data/hotpot_bridge_v1/smoke"

    with (data_dir / "questions.jsonl").open(encoding="utf-8") as file:
        question = json.loads(next(line for line in file if line.strip()))

    tool = SearchTool(data_dir / "documents.jsonl")

    print("Token check:", tokenize("Shirley Temple's career in 1945"))
    print("Question ID:", question["id"])
    print("Question:", question["question"])

    # 诊断阶段检索原问题，不使用答案或标准证据生成查询。
    results = tool.search(question["question"], top_k=3)

    print("\nRetrieved documents:")
    for doc in results:
        print(doc["id"], doc["title"], doc["score"])

    gold_ids = set(question["supporting_doc_ids"])
    retrieved_ids = {doc["id"] for doc in results}

    print("\nGold documents retrieved:",
          len(gold_ids & retrieved_ids), "/", len(gold_ids))
    print("Missing gold IDs:", sorted(gold_ids - retrieved_ids))


if __name__ == "__main__":
    main()