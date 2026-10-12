import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/raw/hotpotqa/hotpot_dev_distractor_hf.json"


def main():
    content = PATH.read_bytes()
    records = json.loads(content)

    if not isinstance(records, list) or not records:
        raise ValueError("原始文件应为非空JSON列表")

    ids = set()
    invalid_evidence = []

    for record in records:
        question_id = record["_id"]
        if question_id in ids:
            raise ValueError(f"Duplicate ID: {question_id}")
        ids.add(question_id)

        for field in ("question", "answer"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Invalid {field}: {question_id}")

        contexts = {}
        for title, sentences in record["context"]:
            if title in contexts:
                raise ValueError(f"Duplicate context title: {question_id}")
            if (
                not isinstance(sentences, list)
                or not all(isinstance(s, str) for s in sentences)
            ):
                raise ValueError(f"Invalid context: {question_id}")
            contexts[title] = sentences

        for title, sentence_id in record["supporting_facts"]:
            if title not in contexts:
                invalid_evidence.append({
                    "question_id": question_id,
                    "title": title,
                    "sentence_id": sentence_id,
                    "reason": "missing_supporting_document",
                })
                continue

            sentence_count = len(contexts[title])
            if (
                not isinstance(sentence_id, int)
                or not 0 <= sentence_id < sentence_count
            ):
                invalid_evidence.append({
                    "question_id": question_id,
                    "title": title,
                    "sentence_id": sentence_id,
                    "sentence_count": sentence_count,
                    "reason": "invalid_supporting_sentence",
                })

    print("File:", PATH)
    print("SHA256:", hashlib.sha256(content).hexdigest())
    print("Questions:", len(records))
    print("Types:", dict(Counter(r.get("type") for r in records)))
    print("Levels:", dict(Counter(r.get("level") for r in records)))
    invalid_ids = {
        item["question_id"] for item in invalid_evidence
    }

    report = {
        "source_file": str(PATH),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "total_questions": len(records),
        "invalid_question_count": len(invalid_ids),
        "valid_evidence_question_count": len(records) - len(invalid_ids),
        "invalid_question_ids": sorted(invalid_ids),
        "invalid_evidence": invalid_evidence,
    }

    report_path = PATH.parent / "inspection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Invalid evidence entries:", len(invalid_evidence))
    print("Invalid questions:", len(invalid_ids))
    print("Valid evidence questions:", len(records) - len(invalid_ids))
    print("Inspection report:", report_path)

    # 仅展示一个样本；它之后只能用于开发检查。
    example = next(
        (r for r in records if r.get("type") == "bridge"),
        records[0],
    )

    print("\nExample ID:", example["_id"])
    print("Question:", example["question"])
    print("Answer:", example["answer"])
    print("Supporting facts:", example["supporting_facts"])
    print("Context paragraphs:", len(example["context"]))


if __name__ == "__main__":
    main()