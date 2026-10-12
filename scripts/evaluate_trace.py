import argparse
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    """统一大小写、全半角，去除空白和标点；不做语义判断。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char
        for char in text
        if not char.isspace()
        and not unicodedata.category(char).startswith("P")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--question-id", required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))

    with (ROOT / "data" / "questions.jsonl").open(
        encoding="utf-8"
    ) as file:
        questions = [
            json.loads(line) for line in file if line.strip()
        ]

    matches = [
        question for question in questions
        if question["id"] == args.question_id
    ]
    if len(matches) != 1:
        raise ValueError("问题 ID 必须唯一且存在")

    question = matches[0]

    # 防止把一个问题的轨迹拿去与另一个问题的答案比较。
    if trace["question"].strip() != question["question"].strip():
        raise ValueError("轨迹问题与指定问题不一致，请检查 question-id")

    gold_ids = set(question["supporting_doc_ids"])
    if not gold_ids:
        raise ValueError("当前评测器要求标准证据非空")

    cited_ids = set(trace.get("evidence_ids", []))
    retrieved_ids = set()

    for event in trace.get("events", []):
        observation = event.get("observation", {})
        for document in observation.get("results", []):
            retrieved_ids.add(document["id"])

    predicted_answer = trace.get("final_answer") or ""
    cited_gold_ids = cited_ids & gold_ids

    report = {
        "question_id": question["id"],
        "question": question["question"],
        "predicted_answer": predicted_answer,
        "gold_answer": question["answer"],
        "finished": trace["status"] == "finished",
        "answer_exact_match": (
            normalize(predicted_answer) == normalize(question["answer"])
        ),
        "gold_evidence_retrieval_recall": (
            len(retrieved_ids & gold_ids) / len(gold_ids)
        ),
        "all_gold_evidence_retrieved": gold_ids <= retrieved_ids,
        "citation_gold_id_precision": (
            len(cited_gold_ids) / len(cited_ids) if cited_ids else 0.0
        ),
        "citation_gold_id_recall": (
            len(cited_gold_ids) / len(gold_ids)
        ),
        "citations_all_seen": cited_ids <= retrieved_ids,
        "retrieved_ids": sorted(retrieved_ids),
        "cited_ids": sorted(cited_ids),
        "gold_ids": sorted(gold_ids),
        "missing_gold_ids": sorted(gold_ids - retrieved_ids),
        "search_count": trace["search_count"],
        "total_output_tokens": trace.get("total_output_tokens"),
        "end_status": trace["status"],
        "limitations": (
            "答案使用严格匹配，证据使用标准ID匹配；"
            "不判断语义支持，也不识别其他有效证据。"
        ),
    }

    report_path = args.trace.with_name(
        args.trace.stem + ".evaluation.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n评测结果已保存：{report_path}")


if __name__ == "__main__":
    main()