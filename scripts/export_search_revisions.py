import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/sft_recovery_v1"
SOURCE = DATA_DIR / "search_revision_candidates.json"
OUTPUT = DATA_DIR / "search_revision_train.jsonl"

APPROVED = {
    "5a76b7d15542993735360186": {
        "answer": "Recreation Ground",
        "titles": ["Siose Muliumu", "Whitehaven R.L.F.C."],
        "entity": "Whitehaven",
        "query": "Whitehaven rugby league stadium",
        "notes": "人物到俱乐部，再到球场；删除原轨迹多余查询。",
    },
    "5ae7d6e85542993210983f7f": {
        "answer": "Bucharest",
        "titles": ["Larga River (Humor)", "Romania"],
        "entity": "Romania",
        "query": "capital of Romania",
        "notes": "首次同名河流结果均位于Romania；直接查询首都。",
    },
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"避免覆盖：{OUTPUT}")

    packets = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    if (
        len(packets) != 2
        or {p["question_id"] for p in packets} != set(APPROVED)
    ):
        raise ValueError("候选集合与审核配置不一致")

    samples = []

    for packet in packets:
        qid = packet["question_id"]
        approved = APPROVED[qid]
        trace_path = Path(packet["source_trace"])

        if sha256(trace_path) != packet["source_trace_sha256"]:
            raise ValueError(f"源轨迹发生变化：{qid}")

        documents_path = trace_path.parent.parent / "documents.jsonl"
        if sha256(documents_path) != packet["documents_sha256"]:
            raise ValueError(f"源文档库发生变化：{qid}")

        trace = json.loads(
            trace_path.read_text(encoding="utf-8-sig")
        )
        if trace["question"] != packet["question"]:
            raise ValueError(f"问题不一致：{qid}")
        if packet["gold_answer_for_review"] != approved["answer"]:
            raise ValueError(f"审核答案不一致：{qid}")

        steps = packet["search_steps"]
        if len(steps) != 2:
            raise ValueError("预期两次搜索")
        if steps[0]["query"] != trace["events"][0]["action"]["query"]:
            raise ValueError("首次查询发生变化")
        if steps[0]["results"] != (
            trace["events"][0]["observation"]["results"]
        ):
            raise ValueError("首次工具结果发生变化")
        if steps[1]["query"] != approved["query"]:
            raise ValueError("修订查询与审核配置不一致")

        first_text = "\n".join(
            d["text"] for d in steps[0]["results"]
        )
        if approved["entity"] not in first_text:
            raise ValueError("第二次查询使用了尚未出现的实体")

        # 只保留原始系统提示词及初始问题。
        messages = json.loads(json.dumps(trace["messages"][:2]))
        if [m["role"] for m in messages] != ["system", "user"]:
            raise ValueError("初始消息格式不符合预期")

        documents = {}

        for index, step in enumerate(steps, start=1):
            if step["remaining_searches"] != 3 - index:
                raise ValueError("剩余搜索预算不一致")

            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "type": "search",
                    "query": step["query"],
                }, ensure_ascii=False),
            })

            observation = {
                "results": step["results"],
                "remaining_searches": step["remaining_searches"],
            }
            messages.append({
                "role": "user",
                "content": (
                    "Controller result:\n"
                    + json.dumps(observation, ensure_ascii=False)
                    + "\nChoose your next action."
                ),
            })

            for doc in step["results"]:
                documents[doc["id"]] = doc

        evidence_ids = []
        for title in approved["titles"]:
            matches = [
                doc_id for doc_id, doc in documents.items()
                if doc["title"] == title
            ]
            if len(matches) != 1:
                raise ValueError(f"证据标题缺失或不唯一：{title}")
            evidence_ids.append(matches[0])

        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "type": "finish",
                "answer": approved["answer"],
                "evidence_ids": evidence_ids,
            }, ensure_ascii=False),
        })

        samples.append({
            "messages": messages,
            "metadata": {
                "question_id": qid,
                "model": trace["model"],
                "source_trace": str(trace_path),
                "source_trace_sha256": packet["source_trace_sha256"],
                "source_candidates_sha256": sha256(SOURCE),
                "documents_sha256": packet["documents_sha256"],
                "construction": "assistant_designed_search_revision",
                "model_generated": False,
                "review_basis": "assistant review of returned evidence",
                "review_notes": approved["notes"],
                "search_count": 2,
                "queries_and_tool_results_modified": True,
                "modification": (
                    "second query revised; actual retrieval replayed; "
                    "redundant searches removed; final citations revised"
                ),
                "raw_trace_modified": False,
                "error_recovery_example": False,
                "purpose": "query progression demonstration",
            },
        })

    # 全部检查通过后才写入。
    with OUTPUT.open("x", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Exported trajectories:", len(samples))
    print("Action examples:", sum(
        m["role"] == "assistant"
        for sample in samples for m in sample["messages"]
    ))
    print("Saved:", OUTPUT)
    print("Raw traces unchanged. No training started.")


if __name__ == "__main__":
    main()
