import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from hotpot_evaluate_v1 import exact_match_score, f1_score
ROOT = Path(__file__).resolve().parents[1]
METHODS = ("no_search", "rag", "agent")


def normalize(text):
    """严格匹配前的文本规范化，不进行语义判断。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char for char in text
        if not char.isspace()
        and not unicodedata.category(char).startswith("P")
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_questions(path):
    questions = {}

    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            question = json.loads(line)
            question_id = question["id"]

            if question_id in questions:
                raise ValueError(f"重复问题ID：{question_id}")

            if not isinstance(question.get("answer"), str):
                raise ValueError(
                    f"第{line_number}行：answer必须是字符串"
                )

            gold_ids = question.get("supporting_doc_ids")
            if (
                not isinstance(gold_ids, list)
                or not all(isinstance(item, str) for item in gold_ids)
            ):
                raise ValueError(
                    f"{question_id}：supporting_doc_ids格式错误"
                )

            questions[question_id] = question

    if not questions:
        raise ValueError("问题集为空")

    return questions


def evaluate(trace, question, answer_language="zh"):
    """返回逐题指标，不加载模型、不执行搜索。"""
    if trace["question"].strip() != question["question"].strip():
        raise ValueError("轨迹问题与标准问题不一致")

    if (
        trace.get("question_id") is not None
        and trace["question_id"] != question["id"]
    ):
        raise ValueError("轨迹question_id与标准问题ID不一致")

    gold_ids = set(question["supporting_doc_ids"])
    cited_ids = set(trace.get("evidence_ids", []))
    retrieved_ids = set()
    executed_queries = []
    action_errors = 0

    for event in trace.get("events", []):
        action = event.get("action", {})
        observation = event.get("observation", {})

        if "error" in observation:
            action_errors += 1

        if isinstance(action, dict) and action.get("type") == "search":
            # 有结果（包括空列表）或工具失败，才算已执行的搜索。
            # 预算拒绝、非法查询不计入重复搜索统计。
            executed = (
                "results" in observation
                or observation.get("error") == "search_failed"
            )
            if executed:
                executed_queries.append(
                    " ".join(action.get("query", "").split()).casefold()
                )

        for document in observation.get("results", []):
            retrieved_ids.add(document["id"])

    answer = trace.get("final_answer") or ""
    finished = trace.get("status") == "finished"

    # 后续可在数据中提供人工审定的等价答案。
    aliases = question.get("answer_aliases", [])
    if (
        not isinstance(aliases, list)
        or not all(isinstance(alias, str) for alias in aliases)
    ):
        raise ValueError("answer_aliases必须是字符串列表")

    accepted_answers = [question["answer"], *aliases]

    if answer_language == "en":
        exact_match = finished and any(
            exact_match_score(answer, candidate)
            for candidate in accepted_answers
        )
        answer_f1 = (
            max(
                f1_score(answer, candidate)[0]
                for candidate in accepted_answers
            )
            if finished else 0.0
        )
    else:
        exact_match = finished and any(
            normalize(answer) == normalize(candidate)
            for candidate in accepted_answers
        )
        answer_f1 = None

    matching_citations = cited_ids & gold_ids
    repeated_searches = (
        len(executed_queries) - len(set(executed_queries))
    )

    return {
        "question_id": question["id"],
        "method": trace.get("method", "unknown"),
        "question": question["question"],
        "predicted_answer": answer,
        "gold_answer": question["answer"],
        "answer_exact_match": exact_match,
        "answer_f1": answer_f1,
        "answer_language": answer_language,
        "finished": finished,
        "end_status": trace.get("status"),
        # 标准证据为空时，这些指标不适用，不记成满分。
        "gold_evidence_retrieval_recall": (
            len(retrieved_ids & gold_ids) / len(gold_ids)
            if gold_ids else None
        ),
        "all_gold_evidence_retrieved": (
            gold_ids <= retrieved_ids if gold_ids else None
        ),
        "citation_gold_id_precision": (
            len(matching_citations) / len(cited_ids)
            if gold_ids and cited_ids else (0.0 if gold_ids else None)
        ),
        "citation_gold_id_recall": (
            len(matching_citations) / len(gold_ids)
            if gold_ids else None
        ),
        "all_gold_evidence_cited": (
            gold_ids <= cited_ids if gold_ids else None
        ),
        "unsupported_citation_count": len(cited_ids - retrieved_ids),
        "retrieved_ids": sorted(retrieved_ids),
        "cited_ids": sorted(cited_ids),
        "gold_ids": sorted(gold_ids),
        "missing_gold_ids": sorted(gold_ids - retrieved_ids),
        "search_count": trace.get("search_count"),
        "repeated_search_count": repeated_searches,
        "action_error_count": action_errors,
        "total_input_tokens": trace.get("total_input_tokens"),
        "total_output_tokens": trace.get("total_output_tokens"),
        "run_seconds_excluding_loading": trace.get(
            "run_seconds_excluding_loading"
        ),
        "runtime_error": trace.get("error"),
    }


def average(rows, field):
    values = [
        row[field] for row in rows if row.get(field) is not None
    ]
    return mean(values) if values else None


def summarize(rows):
    return {
        "evaluated_count": len(rows),
        "answer_exact_match_rate": average(rows, "answer_exact_match"),
        "mean_answer_f1": average(rows, "answer_f1"),
        "finished_rate": average(rows, "finished"),
        "mean_gold_evidence_retrieval_recall": average(
            rows, "gold_evidence_retrieval_recall"
        ),
        "mean_citation_gold_id_precision": average(
            rows, "citation_gold_id_precision"
        ),
        "mean_citation_gold_id_recall": average(
            rows, "citation_gold_id_recall"
        ),
        "complete_gold_citation_rate": average(
            rows, "all_gold_evidence_cited"
        ),
        "mean_search_count": average(rows, "search_count"),
        "mean_repeated_search_count": average(
            rows, "repeated_search_count"
        ),
        "mean_input_tokens": average(rows, "total_input_tokens"),
        "mean_output_tokens": average(rows, "total_output_tokens"),
        "mean_seconds_excluding_loading": average(
            rows, "run_seconds_excluding_loading"
        ),
        "end_status_counts": dict(
            Counter(row["end_status"] for row in rows)
        ),
    }


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            })


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--run-dir", type=Path)

    parser.add_argument("--question-id")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "questions.jsonl",
    )
    parser.add_argument(
        "--answer-language",
        choices=["zh", "en"],
        default="zh",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)

    if args.trace is not None:
        trace = read_json(args.trace)
        question_id = args.question_id or trace.get("question_id")

        if question_id not in questions:
            raise ValueError(
                "无法确定标准问题，请提供正确的--question-id"
            )

        report = evaluate(
            trace, questions[question_id], args.answer_language
        )
        path = args.trace.with_name(
            args.trace.stem + ".evaluation.json"
        )
        write_json(path, report)

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n评测文件：{path}")
        return

    if args.question_id:
        parser.error("--question-id仅用于单份轨迹评测")

    if not args.run_dir.is_dir():
        raise ValueError("运行目录不存在")

    # 用本次运行的summary确定预期任务，避免漏掉失败或缺失文件。
    manifest = read_json(args.run_dir / "summary.json")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("summary.json必须是非空运行任务列表")

    tasks = {}
    for item in manifest:
        key = (item["method"], item["question_id"])
        if key in tasks:
            raise ValueError(f"summary中存在重复任务：{key}")
        if key[0] not in METHODS or key[1] not in questions:
            raise ValueError(f"未知方法或问题：{key}")
        tasks[key] = item

    rows = []
    errors = []

    for method, question_id in tasks:
        path = args.run_dir / method / f"{question_id}.json"

        try:
            trace = read_json(path)
            if trace.get("method") != method:
                raise ValueError("轨迹method与目录不一致")

            report = evaluate(
                trace, questions[question_id], args.answer_language
            )
            report["trace_file"] = str(path)
            rows.append(report)

        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({
                "method": method,
                "question_id": question_id,
                "trace_file": str(path),
                "error": str(error),
            })

    summaries = []
    for method in METHODS:
        expected_ids = {
            question_id
            for task_method, question_id in tasks
            if task_method == method
        }
        if not expected_ids:
            continue

        method_rows = [
            row for row in rows if row["method"] == method
        ]
        evaluated_ids = {row["question_id"] for row in method_rows}

        summaries.append({
            "method": method,
            "expected_count": len(expected_ids),
            "missing_or_invalid_count": len(
                expected_ids - evaluated_ids
            ),
            **summarize(method_rows),
        })

    method_sets = {
        summary["method"]: {
            qid for method, qid in tasks
            if method == summary["method"]
        }
        for summary in summaries
    }
    same_questions = (
        len(method_sets) >= 2
        and all(
            ids == next(iter(method_sets.values()))
            for ids in method_sets.values()
        )
    )

    evaluation_dir = args.run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)

    write_json(evaluation_dir / "details.json", rows)
    write_csv(evaluation_dir / "details.csv", rows)
    write_json(evaluation_dir / "errors.json", errors)
    write_csv(evaluation_dir / "summary.csv", summaries)
    write_json(evaluation_dir / "summary.json", {
        "questions_file": str(args.questions.resolve()),
        "methods_use_same_questions": same_questions,
        "evaluation_complete": not errors,
        "methods": summaries,
        "limitations": [
            "英文答案采用HotpotQA官方EM和词级F1；中文沿用原规则。两者都不判断语义。",
            "证据指标使用标准文档ID，不等于证据语义支持度。",
            "汇总均值仅包含成功读取的轨迹；必须同时检查缺失数。",
            "相同题目不代表模型、提示词和预算配置已验证一致。",
        ],
    })

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"\n方法题目集合一致：{same_questions}")
    print(f"缺失或损坏轨迹：{len(errors)}")
    print(f"评测目录：{evaluation_dir}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()