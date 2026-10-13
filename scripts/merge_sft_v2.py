import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/sft_mixed_v2"

SOURCES = [
    ("reviewed", ROOT / "data/sft_reviewed_v1/train.jsonl"),
    (
        "search_revision",
        ROOT / "data/sft_recovery_v1/search_revision_train.jsonl",
    ),
    (
        "citation_recovery",
        ROOT / "data/sft_recovery_v1/citation_recovery/train.jsonl",
    ),
]


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"避免覆盖：{OUTPUT_DIR}")

    samples = []
    source_info = []
    sample_ids = set()
    supervised_actions = 0

    for source_type, path in SOURCES:
        raw = path.read_bytes()
        rows = [
            json.loads(line)
            for line in raw.decode("utf-8-sig").splitlines()
            if line.strip()
        ]

        source_info.append({
            "type": source_type,
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "count": len(rows),
        })

        for row in rows:
            metadata = row["metadata"]
            qid = metadata["question_id"]
            sample_id = f"{source_type}:{qid}"

            if sample_id in sample_ids:
                raise ValueError(f"重复样本：{sample_id}")
            sample_ids.add(sample_id)

            messages = row["messages"]
            indices = metadata.get("supervised_message_indices")

            if source_type == "citation_recovery" and indices is None:
                raise ValueError("恢复样本缺少监督索引")

            if indices is None:
                target_count = sum(
                    m["role"] == "assistant" for m in messages
                )
            else:
                if (
                    not isinstance(indices, list)
                    or not indices
                    or len(indices) != len(set(indices))
                ):
                    raise ValueError("监督索引异常")

                for index in indices:
                    if (
                        type(index) is not int
                        or not 0 <= index < len(messages)
                        or messages[index]["role"] != "assistant"
                    ):
                        raise ValueError("监督索引无效")

                if source_type == "citation_recovery":
                    if indices != [len(messages) - 1]:
                        raise ValueError("恢复样本应只监督最后的修复动作")
                    if metadata["injected_message_index"] in indices:
                        raise ValueError("错误动作被选为监督目标")

                target_count = len(indices)

            metadata["sample_id"] = sample_id
            metadata["source_type"] = source_type
            supervised_actions += target_count
            samples.append(row)

    unique_questions = {
        row["metadata"]["question_id"] for row in samples
    }
    if (
        len(samples) != 45
        or len(unique_questions) != 43
        or supervised_actions != 99
    ):
        raise ValueError("样本数量与预期不一致，请检查来源数据")

    OUTPUT_DIR.mkdir(parents=True)
    output_path = OUTPUT_DIR / "train.jsonl"
    with output_path.open("x", encoding="utf-8") as file:
        for row in samples:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "records": len(samples),
        "unique_questions": len(unique_questions),
        "supervised_actions": supervised_actions,
        "source_counts": dict(Counter(
            row["metadata"]["source_type"] for row in samples
        )),
        "sources": source_info,
        "train_sha256": hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest(),
        "note": (
            "Two training questions have both clean-search and "
            "synthetic recovery versions. Recovery errors are "
            "context only, not supervised targets."
        ),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()