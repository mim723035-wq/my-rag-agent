import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs/baselines_20261013T092302Z_9a649338"
REVIEW_DIR = RUN_DIR / "sft_review"
OUTPUT_DIR = ROOT / "data/sft_reviewed_v1"
ACCEPTED = {"accept_original", "accept_citation_revision"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"避免覆盖已有数据：{OUTPUT_DIR}")

    questions = {}
    with (RUN_DIR / "questions.jsonl").open(
        encoding="utf-8-sig"
    ) as file:
        for line in file:
            if line.strip():
                question = json.loads(line)
                if question["id"] in questions:
                    raise ValueError("训练问题ID重复")
                questions[question["id"]] = question

    candidates = {}
    with (REVIEW_DIR / "all_packets.jsonl").open(
        encoding="utf-8-sig"
    ) as file:
        for line in file:
            if line.strip():
                packet = json.loads(line)
                qid = packet["question_id"]
                if qid in candidates:
                    raise ValueError("候选ID重复")
                candidates[qid] = packet

    review_files = sorted(
        REVIEW_DIR.glob("batch_*.reviewed.json")
    )
    if len(review_files) != 7:
        raise ValueError("应有7个已审核批次")

    reviewed = {}
    review_sources = {}
    for path in review_files:
        for packet in load_json(path):
            qid = packet["question_id"]
            if qid in reviewed:
                raise ValueError(f"审核ID重复：{qid}")
            reviewed[qid] = packet
            review_sources[qid] = path

    if set(reviewed) != set(candidates):
        raise ValueError("审核题目与候选集合不一致，存在遗漏或额外题目")
    if len(reviewed) != 57:
        raise ValueError("本轮应有57条审核记录")

    counts = Counter()
    samples = []
    search_counts = Counter()

    for qid, packet in sorted(reviewed.items()):
        review = packet["review"]
        decision = review["decision"]
        if decision == "pending":
            raise ValueError(f"尚未审核：{qid}")
        counts[decision] += 1

        if decision not in ACCEPTED:
            continue

        for field in (
            "answer_supported",
            "query_progression_valid",
            "stopping_reasonable",
        ):
            if review.get(field) is not True:
                raise ValueError(f"{qid}: 接受状态与{field}不一致")

        question = questions[qid]
        if question.get("split") != "train":
            raise ValueError(f"禁止导出非训练题：{qid}")

        trace_path = RUN_DIR / "agent" / f"{qid}.json"
        trace = load_json(trace_path)
        if trace["status"] != "finished":
            raise ValueError(f"轨迹未完成：{qid}")
        if trace["question"] != question["question"]:
            raise ValueError(f"问题不一致：{qid}")
        if packet["question"] != trace["question"]:
            raise ValueError(f"审核问题不一致：{qid}")
        if packet["model_answer"] != trace["final_answer"]:
            raise ValueError(f"审核答案与轨迹不一致：{qid}")
        if trace["model"].get("adapter_dir"):
            raise ValueError(f"源轨迹使用了适配器：{qid}")

        seen_ids = set()
        query_list = []
        for event in trace["events"]:
            observation = event["observation"]
            if "error" in observation:
                raise ValueError(f"轨迹含动作错误：{qid}")
            seen_ids.update(
                doc["id"]
                for doc in observation.get("results", [])
            )
            if event["action"].get("type") == "search":
                query_list.append(event["action"]["query"])

        if len(query_list) != len(set(query_list)):
            raise ValueError(f"轨迹含完全相同的重复查询：{qid}")
        if len(query_list) != packet["search_count"]:
            raise ValueError(f"搜索次数与审核记录不一致：{qid}")

        # 深复制，原始轨迹文件不变。
        messages = json.loads(json.dumps(trace["messages"]))
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"最后一条消息不是助手动作：{qid}")

        for message in messages:
            if message["role"] == "assistant":
                action = json.loads(message["content"])
                if action.get("type") not in {"search", "finish"}:
                    raise ValueError(f"非法助手动作：{qid}")

        final_action = json.loads(messages[-1]["content"])
        if final_action.get("type") != "finish":
            raise ValueError(f"最后动作不是finish：{qid}")
        if final_action["answer"] != trace["final_answer"]:
            raise ValueError(f"最终答案不一致：{qid}")

        original_ids = final_action.get("evidence_ids", [])
        if original_ids != packet["original_citation_ids"]:
            raise ValueError(f"原始引用与审核记录不一致：{qid}")

        corrected_ids = review["corrected_citation_ids"]
        if (
            not isinstance(corrected_ids, list)
            or not corrected_ids
            or not all(isinstance(x, str) for x in corrected_ids)
            or len(corrected_ids) != len(set(corrected_ids))
        ):
            raise ValueError(f"修订引用格式不合法：{qid}")
        if not set(corrected_ids).issubset(seen_ids):
            raise ValueError(f"引用了未返回的文档：{qid}")

        if decision == "accept_original":
            if corrected_ids != original_ids:
                raise ValueError(f"原样接受却修改引用：{qid}")
        else:
            final_action["evidence_ids"] = corrected_ids
            messages[-1]["content"] = json.dumps(
                final_action, ensure_ascii=False
            )

        samples.append({
            "messages": messages,
            "metadata": {
                "question_id": qid,
                "source_trace": str(trace_path),
                "source_trace_sha256": sha256(trace_path),
                "source_review": str(review_sources[qid]),
                "source_review_sha256": sha256(review_sources[qid]),
                "model": trace["model"],
                "review_basis": "assistant review of returned evidence",
                "review_decision": decision,
                "review_notes": review["notes"],
                "modification": (
                    "none" if decision == "accept_original"
                    else "final citation IDs corrected"
                ),
                "original_evidence_ids": original_ids,
                "corrected_evidence_ids": corrected_ids,
                "queries_and_tool_results_modified": False,
                "search_count": len(query_list),
                "purpose": "reviewed SFT candidate dataset",
            },
        })
        search_counts[
            "single_search" if len(query_list) == 1 else "multi_search"
        ] += 1

    if len(samples) != 41:
        raise ValueError(f"预期41条接受样本，实际{len(samples)}")

    summary = {
        "source_run": str(RUN_DIR),
        "reviewed_count": len(reviewed),
        "decision_counts": dict(counts),
        "exported_count": len(samples),
        "search_distribution": dict(search_counts),
        "reviewer": "assistant_evidence_review",
        "pilot_merged": False,
        "raw_traces_modified": False,
        "limitations": [
            "Assistant evidence review, not human adjudication.",
            "Selected successful traces; not representative of all tasks.",
            "Mostly single-search examples.",
            "No new recovery trajectories added.",
        ],
    }

    # 所有检查通过后才创建输出。
    OUTPUT_DIR.mkdir(parents=True)
    output_path = OUTPUT_DIR / "train.jsonl"
    with output_path.open("x", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary["train_sha256"] = sha256(output_path)
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")
    print("Next: tokenize and verify supervised targets; do not train yet.")


if __name__ == "__main__":
    main()