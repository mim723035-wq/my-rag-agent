import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from search import SearchTool

RUN_DIR = ROOT / "outputs/baselines_20261013T092302Z_9a649338"
OUTPUT_DIR = ROOT / "data/sft_recovery_v1"

PLANS = {
    "5a76b7d15542993735360186": {
        "intermediate_entity": "Whitehaven",
        "second_query": "Whitehaven rugby league stadium",
    },
    "5ae7d6e85542993210983f7f": {
        "intermediate_entity": "Romania",
        "second_query": "capital of Romania",
    },
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # 使用原始运行保存的同一文档库。
    documents_path = RUN_DIR / "documents.jsonl"
    tool = SearchTool(documents_path)

    questions = {}
    with (RUN_DIR / "questions.jsonl").open(
        encoding="utf-8-sig"
    ) as file:
        for line in file:
            if line.strip():
                question = json.loads(line)
                questions[question["id"]] = question

    packets = []

    for qid, plan in PLANS.items():
        question = questions[qid]
        if question.get("split") != "train":
            raise ValueError(f"禁止使用非训练题：{qid}")

        source_path = RUN_DIR / "agent" / f"{qid}.json"
        trace = load_json(source_path)

        if trace["question"] != question["question"]:
            raise ValueError(f"源轨迹问题不一致：{qid}")

        first_event = trace["events"][0]
        if first_event["action"].get("type") != "search":
            raise ValueError(f"首次动作不是搜索：{qid}")

        first_query = first_event["action"]["query"]
        first_results = tool.search(first_query, top_k=3)

        # 确认检索重放一致，避免悄悄更换检索条件。
        if first_results != first_event["observation"]["results"]:
            raise ValueError(f"首次检索与原始结果不一致：{qid}")

        first_text = "\n".join(
            doc["text"] for doc in first_results
        )
        if plan["intermediate_entity"] not in first_text:
            raise ValueError(f"中间实体尚未出现在首次证据中：{qid}")

        second_results = tool.search(
            plan["second_query"], top_k=3
        )

        seen_ids = {
            doc["id"]
            for doc in first_results + second_results
        }
        gold_ids = question["supporting_doc_ids"]

        packet = {
            "question_id": qid,
            "question": question["question"],
            # 仅用于审核，不传入搜索或后续模型消息。
            "gold_answer_for_review": question["answer"],
            "gold_document_ids_for_review": gold_ids,
            "source_trace": str(source_path),
            "source_trace_sha256": sha256(source_path),
            "documents_sha256": sha256(documents_path),
            "construction": "assistant_designed_search_revision",
            "model_generated": False,
            "search_budget": 3,
            "retrieval_top_k": 3,
            "search_steps": [
                {
                    "step": 1,
                    "query": first_query,
                    "results": first_results,
                    "remaining_searches": 2,
                },
                {
                    "step": 2,
                    "query": plan["second_query"],
                    "results": second_results,
                    "remaining_searches": 1,
                },
            ],
            "all_gold_evidence_retrieved": (
                set(gold_ids).issubset(seen_ids)
            ),
            "review_status": "pending",
            "raw_trace_modified": False,
        }
        packets.append(packet)

        print(f"\nQuestion: {qid}")
        print(question["question"])
        for step in packet["search_steps"]:
            print(f"\nStep {step['step']}: {step['query']}")
            for doc in step["results"]:
                print(f"{doc['id']} | {doc['title']}")
                print(doc["text"])

        print(
            "\nAll gold evidence retrieved:",
            packet["all_gold_evidence_retrieved"],
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "search_revision_candidates.json"
    if output_path.exists():
        raise FileExistsError(f"避免覆盖：{output_path}")

    output_path.write_text(
        json.dumps(packets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {output_path}")
    print("Candidates only; not exported as SFT training data.")


if __name__ == "__main__":
    main()