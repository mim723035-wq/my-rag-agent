import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs/baselines_20261013T092302Z_9a649338"
OUTPUT_DIR = RUN_DIR / "sft_review"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "sft_review目录已存在，停止生成，避免覆盖核验记录"
        )

    rows = read_json(RUN_DIR / "evaluation/details.json")

    # 排除已经用于pilot的题目，不重复作为新增样本。
    pilot_ids = set()
    pilot_path = ROOT / "data/sft_pilot_v1/train.jsonl"
    if pilot_path.exists():
        with pilot_path.open(encoding="utf-8-sig") as file:
            for line in file:
                if line.strip():
                    sample = json.loads(line)
                    pilot_ids.add(sample["metadata"]["question_id"])

    candidates = [
        row for row in rows
        if row["answer_exact_match"]
        and row["all_gold_evidence_retrieved"]
        and row["action_error_count"] == 0
        and row["repeated_search_count"] == 0
        and not row["all_gold_evidence_cited"]
    ]

    excluded_pilot = [
        row["question_id"] for row in candidates
        if row["question_id"] in pilot_ids
    ]
    candidates = [
        row for row in candidates
        if row["question_id"] not in pilot_ids
    ]

    packets = []

    for row in sorted(candidates, key=lambda r: r["question_id"]):
        trace = read_json(
            RUN_DIR / "agent" / f"{row['question_id']}.json"
        )

        seen_ids = set()
        search_steps = []

        for event in trace["events"]:
            action = event["action"]
            if not isinstance(action, dict):
                continue
            if action.get("type") != "search":
                continue

            documents = event["observation"].get("results", [])
            returned_ids = {doc["id"] for doc in documents}
            new_ids = returned_ids - seen_ids
            seen_ids.update(returned_ids)

            search_steps.append({
                "step": event["step"],
                "query": action["query"],
                "new_document_count": len(new_ids),
                "new_gold_document_ids": sorted(
                    new_ids & set(row["gold_ids"])
                ),
                "documents": documents,
            })

        packets.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "model_answer": row["predicted_answer"],
            "gold_document_ids": row["gold_ids"],
            "original_citation_ids": row["cited_ids"],
            "search_steps": search_steps,
            "search_count": trace["search_count"],
            "zero_new_document_search_count": sum(
                step["new_document_count"] == 0
                for step in search_steps
            ),
            "trace_file": str(
                RUN_DIR / "agent" / f"{row['question_id']}.json"
            ),
            "review": {
                "decision": "pending",
                "answer_supported": None,
                "query_progression_valid": None,
                "stopping_reasonable": None,
                "corrected_citation_ids": [],
                "notes": "",
            },
        })

    if not packets:
        raise ValueError("没有剩余候选")

    # 第一批兼顾单次搜索与多次搜索。
    # 这是人工核验安排，不是最终训练数据抽样规则。
    single = [p for p in packets if p["search_count"] == 1]
    multi = [p for p in packets if p["search_count"] > 1]

    first_batch = multi[:3] + single[:2]
    selected_ids = {p["question_id"] for p in first_batch}
    for packet in packets:
        if len(first_batch) >= 5:
            break
        if packet["question_id"] not in selected_ids:
            first_batch.append(packet)
            selected_ids.add(packet["question_id"])

    OUTPUT_DIR.mkdir()
    with (OUTPUT_DIR / "all_packets.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for packet in packets:
            file.write(json.dumps(packet, ensure_ascii=False) + "\n")

    (OUTPUT_DIR / "batch_001.json").write_text(
        json.dumps(first_batch, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "source_run": str(RUN_DIR),
        "excluded_existing_pilot_ids": excluded_pilot,
        "remaining_review_count": len(packets),
        "single_search_count": len(single),
        "multi_search_count": len(multi),
        "first_batch_ids": [
            p["question_id"] for p in first_batch
        ],
        "note": (
            "All decisions pending. No raw trace modified. "
            "No new documents does not automatically mean useless search."
        ),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Excluded existing pilot questions:", len(excluded_pilot))
    print("Remaining review candidates:", len(packets))
    print("Single-search candidates:", len(single))
    print("Multi-search candidates:", len(multi))
    print("First batch size:", len(first_batch))
    print("First batch:", OUTPUT_DIR / "batch_001.json")


if __name__ == "__main__":
    main()