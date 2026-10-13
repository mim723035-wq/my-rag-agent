import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/hotpotqa/hotpot_train_distractor_hf.json"
EVAL_DIR = ROOT / "data/hotpot_bridge_v1"
OUTPUT = ROOT / "data/hotpot_train_v1"

SEED = 42
TRAIN_SIZE = 300
TRIAL_SIZE = 30


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_key(text):
    return " ".join(text.casefold().split())


def title_key(title):
    return text_key(title)


def document_id(title, sentences):
    content = json.dumps(
        [title, sentences], ensure_ascii=False
    ).encode("utf-8")
    return "hp_doc_" + hashlib.sha256(content).hexdigest()


def validate_record(record):
    """结构检查；返回错误原因，不修改原始数据。"""
    for field in ("_id", "question", "answer"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            return f"invalid_{field}"

    contexts = {}
    raw_context = record.get("context")
    if not isinstance(raw_context, list) or not raw_context:
        return "invalid_context"

    for paragraph in raw_context:
        if not isinstance(paragraph, list) or len(paragraph) != 2:
            return "invalid_paragraph"

        title, sentences = paragraph
        if not isinstance(title, str) or not title.strip():
            return "invalid_title"
        if title in contexts:
            return "duplicate_context_title"
        if (
            not isinstance(sentences, list)
            or not sentences
            or not all(isinstance(s, str) for s in sentences)
        ):
            return "invalid_sentences"
        contexts[title] = sentences

    facts = record.get("supporting_facts")
    if not isinstance(facts, list) or not facts:
        return "invalid_supporting_facts"

    for fact in facts:
        if not isinstance(fact, list) or len(fact) != 2:
            return "invalid_supporting_fact"
        title, sentence_id = fact
        if not isinstance(title, str) or title not in contexts:
            return "missing_supporting_document"
        if (
            type(sentence_id) is not int
            or not 0 <= sentence_id < len(contexts[title])
        ):
            return "invalid_supporting_sentence"

    return None


def encode_jsonl(records):
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"目标目录已存在，停止生成，避免覆盖：{OUTPUT}"
        )

    eval_documents = read_jsonl(EVAL_DIR / "documents.jsonl")
    eval_document_map = {doc["id"]: doc for doc in eval_documents}

    eval_questions = (
        read_jsonl(EVAL_DIR / "questions.jsonl")
        + read_jsonl(EVAL_DIR / "test/questions.jsonl")
    )
    eval_ids = {q["id"] for q in eval_questions}
    eval_question_keys = {text_key(q["question"]) for q in eval_questions}
    eval_support_titles = {
        title_key(eval_document_map[doc_id]["title"])
        for q in eval_questions
        for doc_id in q["supporting_doc_ids"]
    }

    print("Loading training JSON...", flush=True)
    with RAW_PATH.open(encoding="utf-8") as file:
        records = json.load(file)

    # 重复ID涉及的全部记录均排除。
    id_counts = Counter(
        r.get("_id") for r in records if isinstance(r, dict)
    )
    eligible = []
    exclusions = []

    for row_number, record in enumerate(records):
        if not isinstance(record, dict):
            reason = "not_an_object"
            question_id = None
        else:
            question_id = record.get("_id")
            reason = validate_record(record)

            if reason is None and id_counts[question_id] > 1:
                reason = "duplicate_id"
            if reason is None and record.get("type") != "bridge":
                reason = "not_bridge"
            if reason is None and (
                question_id in eval_ids
                or text_key(record["question"]) in eval_question_keys
            ):
                reason = "eval_question_overlap"
            if reason is None:
                titles = {
                    title_key(title)
                    for title, _ in record["supporting_facts"]
                }
                if not titles.isdisjoint(eval_support_titles):
                    reason = "eval_support_title_overlap"

        if reason is not None:
            exclusions.append({
                "row": row_number,
                "question_id": question_id,
                "reason": reason,
            })
        else:
            eligible.append(record)

    # 对剩余候选进行简单文本去重，不等于语义去重。
    unique = {}
    for record in sorted(eligible, key=lambda r: r["_id"]):
        key = text_key(record["question"])
        if key in unique:
            exclusions.append({
                "question_id": record["_id"],
                "reason": "duplicate_question_text",
            })
        else:
            unique[key] = record

    candidates = sorted(unique.values(), key=lambda r: r["_id"])
    if len(candidates) < TRAIN_SIZE:
        raise ValueError(
            f"合格候选只有{len(candidates)}题，不足{TRAIN_SIZE}题"
        )

    random.Random(SEED).shuffle(candidates)
    selected = candidates[:TRAIN_SIZE]

    documents = {}
    questions = []

    for record in selected:
        contexts = dict(record["context"])

        # 加入全部context，包括干扰文档。
        for title, sentences in record["context"]:
            doc_id = document_id(title, sentences)
            documents[doc_id] = {
                "id": doc_id,
                "title": title,
                "text": " ".join(s.strip() for s in sentences),
                "sentences": sentences,
                "provenance": "HotpotQA distractor train context",
            }

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

        questions.append({
            "id": record["_id"],
            "question": record["question"],
            "answer": record["answer"],
            "answer_aliases": [],
            "supporting_doc_ids": gold_ids,
            "supporting_facts": facts,
            "split": "train",
            "task_type": record["type"],
            "level": record["level"],
            "answerable": True,
            "provenance": "HotpotQA distractor train",
        })

    selected_support_titles = {
        title_key(title)
        for r in selected
        for title, _ in r["supporting_facts"]
    }
    assert selected_support_titles.isdisjoint(eval_support_titles)
    assert not {q["id"] for q in questions} & eval_ids

    document_bytes = encode_jsonl([
        documents[key] for key in sorted(documents)
    ])
    question_bytes = encode_jsonl(questions)
    trial_bytes = encode_jsonl(questions[:TRIAL_SIZE])

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "documents.jsonl").write_bytes(document_bytes)
    (OUTPUT / "questions.jsonl").write_bytes(question_bytes)
    (OUTPUT / "trial_questions.jsonl").write_bytes(trial_bytes)
    (OUTPUT / "excluded_records.json").write_text(
        json.dumps(exclusions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "source": "HotpotQA distractor train",
        "source_sha256": file_sha256(RAW_PATH),
        "seed": SEED,
        "raw_count": len(records),
        "eligible_count": len(candidates),
        "train_count": len(questions),
        "trial_count": TRIAL_SIZE,
        "document_count": len(documents),
        "exclusions": dict(Counter(e["reason"] for e in exclusions)),
        "train_ids": [q["id"] for q in questions],
        "trial_ids": [q["id"] for q in questions[:TRIAL_SIZE]],
        "sha256": {
            "documents": hashlib.sha256(document_bytes).hexdigest(),
            "questions": hashlib.sha256(question_bytes).hexdigest(),
            "trial_questions": hashlib.sha256(trial_bytes).hexdigest(),
        },
        "eval_input_sha256": {
            name: file_sha256(EVAL_DIR / name)
            for name in (
                "documents.jsonl",
                "questions.jsonl",
                "test/questions.jsonl",
            )
        },
        "license": "CC BY-SA 4.0",
        "limitations": [
            "Support-title isolation is not complete entity isolation.",
            "Distractor documents may overlap with evaluation documents.",
            "Structural checks do not validate answer semantics.",
            "This is a question pool, not a verified SFT trajectory dataset.",
        ],
    }

    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Raw training questions:", len(records))
    print("Eligible bridge questions:", len(candidates))
    print("Selected training questions:", len(questions))
    print("Trial questions:", TRIAL_SIZE)
    print("Shared training documents:", len(documents))
    print("Eval gold-title overlap: 0")
    print("Exclusions:", manifest["exclusions"])
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()