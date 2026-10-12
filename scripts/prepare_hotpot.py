import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/hotpotqa"
SOURCE = RAW_DIR / "hotpot_dev_distractor_hf.json"
REPORT = RAW_DIR / "inspection_report.json"
OUTPUT = ROOT / "data/hotpot_bridge_v1"

SEED = 42
DEV_SIZE = 100
TEST_SIZE = 200
SMOKE_SIZE = 30

# 已展示过内容的样本只能进入开发集。
SEEN_ID = "5a8c7595554299585d9e36b6"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def supporting_titles(record):
    return {title for title, _ in record["supporting_facts"]}


def validate_record(record):
    contexts = dict(record["context"])
    for title, sentence_id in record["supporting_facts"]:
        if title not in contexts:
            raise ValueError(f"Missing support: {record['_id']}")
        if (
            type(sentence_id) is not int
            or not 0 <= sentence_id < len(contexts[title])
        ):
            raise ValueError(f"Invalid support index: {record['_id']}")


def document_id(title, sentences):
    # 同标题但正文不同的段落，保留为不同文档版本。
    content = json.dumps(
        [title, sentences], ensure_ascii=False
    ).encode("utf-8")
    return "hp_doc_" + sha256(content)


def convert_question(record, split):
    contexts = dict(record["context"])
    gold_ids = []
    facts = []

    for title, sentence_id in record["supporting_facts"]:
        doc_id = document_id(title, contexts[title])
        if doc_id not in gold_ids:
            gold_ids.append(doc_id)
        facts.append({
            "document_id": doc_id,
            "sentence_id": sentence_id,
        })

    return {
        "id": record["_id"],
        "question": record["question"],
        "answer": record["answer"],
        "answer_aliases": [],
        "supporting_doc_ids": gold_ids,
        "supporting_facts": facts,
        "split": split,
        "task_type": record["type"],
        "level": record["level"],
        "answerable": True,
        "provenance": "HotpotQA distractor validation",
    }


def encode_jsonl(records):
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"目标目录已存在，停止生成以避免覆盖：{OUTPUT}"
        )

    source_bytes = SOURCE.read_bytes()
    records = json.loads(source_bytes)
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))

    # 检查报告若保存了哈希，必须与当前原始文件一致。
    reported_hash = report.get("source_sha256")
    if reported_hash and reported_hash != sha256(source_bytes):
        raise ValueError("检查报告与原始文件哈希不一致，请重新检查")

    invalid_ids = set(report["invalid_question_ids"])
    # 保留已确认的异常ID，即使旧报告遗漏也不进入候选集。
    invalid_ids.add("5ae61bfd5542992663a4f261")

    candidates = [
        record for record in records
        if record["type"] == "bridge"
        and record["_id"] not in invalid_ids
    ]

    if len({r["_id"] for r in candidates}) != len(candidates):
        raise ValueError("候选问题ID重复")

    for record in candidates:
        validate_record(record)

    # 先排序再随机化，避免原始文件排序改变抽样。
    candidates.sort(key=lambda record: record["_id"])
    random.Random(SEED).shuffle(candidates)

    seen = next(
        (record for record in candidates if record["_id"] == SEEN_ID),
        None,
    )
    if seen is None:
        raise ValueError("已查看样本不在有效候选集中，请核查")

    dev = [seen] + [
        record for record in candidates if record["_id"] != SEEN_ID
    ][:DEV_SIZE - 1]

    dev_ids = {record["_id"] for record in dev}
    dev_support_titles = set().union(
        *(supporting_titles(record) for record in dev)
    )

    # 排除与开发集共享标准支持文档标题的测试候选。
    test_candidates = [
        record for record in candidates
        if record["_id"] not in dev_ids
        and supporting_titles(record).isdisjoint(dev_support_titles)
    ]
    test = test_candidates[:TEST_SIZE]

    if len(dev) != DEV_SIZE or len(test) != TEST_SIZE:
        raise ValueError(
            "满足隔离条件的题目不足，停止生成，不静默缩小规模"
        )

    test_support_titles = set().union(
        *(supporting_titles(record) for record in test)
    )
    assert dev_support_titles.isdisjoint(test_support_titles)

    # 使用所有选中问题的全部context，共享检索库。
    # 不按题目限制工具能访问哪些文档。
    documents = {}
    title_versions = {}

    for record in dev + test:
        for title, sentences in record["context"]:
            doc_id = document_id(title, sentences)
            documents[doc_id] = {
                "id": doc_id,
                "title": title,
                "text": " ".join(s.strip() for s in sentences),
                "sentences": sentences,
                "provenance": "HotpotQA distractor validation context",
            }
            title_versions.setdefault(title, set()).add(doc_id)

    dev_questions = [convert_question(r, "dev") for r in dev]
    test_questions = [convert_question(r, "test") for r in test]
    smoke_questions = dev_questions[:SMOKE_SIZE]

    for question in dev_questions + test_questions:
        if not set(question["supporting_doc_ids"]) <= documents.keys():
            raise ValueError(f"Missing gold document: {question['id']}")

    document_bytes = encode_jsonl([
        documents[doc_id] for doc_id in sorted(documents)
    ])
    dev_bytes = encode_jsonl(dev_questions)
    test_bytes = encode_jsonl(test_questions)
    smoke_bytes = encode_jsonl(smoke_questions)

    OUTPUT.mkdir(parents=True)
    for directory, question_bytes in (
        (OUTPUT, dev_bytes),
        (OUTPUT / "smoke", smoke_bytes),
        (OUTPUT / "test", test_bytes),
    ):
        directory.mkdir(exist_ok=True)
        (directory / "documents.jsonl").write_bytes(document_bytes)
        (directory / "questions.jsonl").write_bytes(question_bytes)

    provenance_path = RAW_DIR / "download_provenance.json"
    download_provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        if provenance_path.exists() else None
    )

    manifest = {
        "source": "HotpotQA distractor validation",
        "license": "CC BY-SA 4.0",
        "download_provenance": download_provenance,
        "source_sha256": sha256(source_bytes),
        "seed": SEED,
        "selection": "bridge; structurally valid supporting facts",
        "known_invalid_ids": sorted(invalid_ids),
        "dev_count": len(dev),
        "smoke_count": len(smoke_questions),
        "test_count": len(test),
        "document_count": len(documents),
        "titles_with_multiple_versions": sum(
            len(versions) > 1 for versions in title_versions.values()
        ),
        "dev_ids": [q["id"] for q in dev_questions],
        "smoke_ids": [q["id"] for q in smoke_questions],
        "test_ids": [q["id"] for q in test_questions],
        "isolation": "dev/test gold supporting titles do not overlap",
        "corpus": "pooled contexts of selected dev and test questions",
        "sha256": {
            "documents": sha256(document_bytes),
            "dev_questions": sha256(dev_bytes),
            "smoke_questions": sha256(smoke_bytes),
            "test_questions": sha256(test_bytes),
        },
        "limitations": [
            "Adapted controlled retrieval task, not official fullwiki.",
            "Distractor documents and entities may overlap across splits.",
            "No claim of absence of pretraining contamination.",
            "Gold document IDs are not sentence-level citation scores.",
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Dev questions:", len(dev))
    print("Smoke questions:", len(smoke_questions))
    print("Held-out questions:", len(test))
    print("Shared documents:", len(documents))
    print("Gold-title overlap: 0")
    print("Saved:", OUTPUT)
    print("Do not inspect held-out answers during development.")


if __name__ == "__main__":
    main()