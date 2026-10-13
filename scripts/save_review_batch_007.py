import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT / "outputs"
    / "baselines_20261013T092302Z_9a649338"
    / "sft_review"
)
SOURCE = REVIEW_DIR / "batch_007.json"
DESTINATION = REVIEW_DIR / "batch_007.reviewed.json"

DECISIONS = {
    "5ae6c0b055429908198fa6ad": (
        "accept_original",
        "Dazzle camouflage文档直接说明名称、归功于Norman Wilkinson"
        "及战争使用背景。原引用足以支持答案，无需机械补齐标准ID。"
        "人物文档存在格式噪声，但不影响当前引用的证据支持。",
    ),
    "5ae825e55542997ec2727732": (
        "accept_citation_revision",
        "Gondolin文档支持其与Troy存在可能的对应关系；"
        "Troy文档支持现址Hisarlik。补充Gondolin引用。"
        "保留possible parallels的不确定性，不扩展为确定历史关系。",
    ),
}


def main():
    packets = json.loads(SOURCE.read_text(encoding="utf-8-sig"))

    if (
        len(packets) != 2
        or {p["question_id"] for p in packets} != set(DECISIONS)
    ):
        raise ValueError("源批次与审核配置不一致。")

    if DESTINATION.exists():
        raise FileExistsError(f"审核文件已存在：{DESTINATION}")

    counts = Counter()

    for packet in packets:
        decision, notes = DECISIONS[packet["question_id"]]
        corrected_ids = list(
            packet["original_citation_ids"]
            if decision == "accept_original"
            else packet["gold_document_ids"]
        )

        seen_ids = {
            doc["id"]
            for step in packet["search_steps"]
            for doc in step["documents"]
        }
        if not set(corrected_ids).issubset(seen_ids):
            raise ValueError(
                f"{packet['question_id']}: 引用包含未检索文档。"
            )

        packet["review"] = {
            "decision": decision,
            "answer_supported": True,
            "query_progression_valid": True,
            "stopping_reasonable": True,
            "corrected_citation_ids": corrected_ids,
            "notes": notes,
            "reviewer": "assistant_evidence_review",
            "raw_trace_modified": False,
        }
        counts[decision] += 1

    DESTINATION.write_text(
        json.dumps(packets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    print(f"Saved: {DESTINATION}")
    print("Raw traces unchanged.")


if __name__ == "__main__":
    main()