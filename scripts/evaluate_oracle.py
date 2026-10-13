import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from hotpot_evaluate_v1 import exact_match_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    with (run_dir / "questions.jsonl").open(
        encoding="utf-8-sig"
    ) as file:
        questions = [
            json.loads(line) for line in file if line.strip()
        ]

    if not questions:
        raise ValueError("问题集为空")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("问题ID重复")

    rows = []

    # 按问题集遍历，缺失文件也计入总题数。
    for question in questions:
        path = run_dir / f"{question['id']}.json"
        trace = None
        read_error = None

        try:
            trace = json.loads(
                path.read_text(encoding="utf-8-sig")
            )
            if trace["question_id"] != question["id"]:
                raise ValueError("问题ID不一致")
            if trace["question"] != question["question"]:
                raise ValueError("问题内容不一致")
            if trace["experiment"] != "oracle_evidence_diagnostic":
                raise ValueError("不是标准证据诊断轨迹")
        except (OSError, ValueError, KeyError, TypeError) as error:
            read_error = str(error)
            trace = None

        status = (
            trace.get("status", "unknown")
            if trace is not None else "missing_or_invalid_trace"
        )
        finished = status == "finished"
        answer = (trace.get("final_answer") or "") if trace else ""
        cited = set(trace.get("evidence_ids", [])) if trace else set()
        gold_ids = set(question["supporting_doc_ids"])

        # 未正常提交的输出不计为成功答案。
        em = float(exact_match_score(answer, question["answer"])) \
            if finished else 0.0
        f1 = f1_score(answer, question["answer"])[0] \
            if finished else 0.0

        complete = finished and gold_ids <= cited
        recall = (
            len(cited & gold_ids) / len(gold_ids)
            if gold_ids else None
        )

        rows.append({
            "question_id": question["id"],
            "question": question["question"],
            "predicted_answer": answer,
            "gold_answer": question["answer"],
            "answer_em": em,
            "answer_f1": f1,
            "status": status,
            "complete_gold_document_citation": complete,
            "gold_document_citation_recall": recall,
            "cited_ids": sorted(cited),
            "gold_ids": sorted(gold_ids),
            "trace_file": str(path),
            "error": read_error or (trace.get("error") if trace else None),
            "needs_review": not em or not complete,
            "manual_verdict": None,
            "manual_notes": "",
        })

    summary = {
        "experiment": "oracle_evidence_diagnostic",
        "question_count": len(rows),
        "finished_count": sum(r["status"] == "finished" for r in rows),
        "answer_em_count": sum(r["answer_em"] for r in rows),
        "answer_em_rate": mean(r["answer_em"] for r in rows),
        "answer_f1_mean": mean(r["answer_f1"] for r in rows),
        "complete_gold_document_citation_count": sum(
            r["complete_gold_document_citation"] for r in rows
        ),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "scorer_sha256": hashlib.sha256(
            (ROOT / "scripts/hotpot_evaluate_v1.py").read_bytes()
        ).hexdigest(),
        "limitations": [
            "Oracle evidence diagnostic, not autonomous retrieval.",
            "EM and F1 are lexical metrics, not semantic judgements.",
            "Document citation coverage is not official sentence-level scoring.",
            "Ambiguous or supported alternative answers require manual review.",
        ],
    }

    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)

    for filename, data in (
        ("details.json", rows),
        ("summary.json", summary),
        ("review_candidates.json", [
            r for r in rows if r["needs_review"]
        ]),
    ):
        (evaluation_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nNon-matching or abnormal answers:")
    for row in rows:
        if not row["answer_em"]:
            print(json.dumps({
                "question_id": row["question_id"],
                "predicted": row["predicted_answer"],
                "gold": row["gold_answer"],
                "status": row["status"],
            }, ensure_ascii=False))

    print("\nSaved:", evaluation_dir)


if __name__ == "__main__":
    main()