import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs/baselines_20261013T044720Z_97d36a36"
OUTPUT_DIR = ROOT / "data/sft_pilot_v1"

SELECTED_IDS = [
    "5a873b215542994846c1cd24",  # Indian Grand Prix
    "5a802dc35542992bc0c4a6c5",  # Galgameth
    "5a89ec4755429946c8d6ea10",  # Javier Hernandez
    "5a8e58cb5542990e94052ae0",  # Simone Kirby
    "5a7c580f55429907fabeef63",  # Danish actor
]


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as file:
        return [
            json.loads(line) for line in file if line.strip()
        ]


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "输出目录已经存在，停止导出，避免覆盖"
        )

    questions = {
        q["id"]: q
        for q in read_jsonl(RUN_DIR / "questions.jsonl")
    }

    samples = []

    for question_id in SELECTED_IDS:
        question = questions[question_id]
        if question["split"] != "train":
            raise ValueError("禁止导出非训练集问题")

        trace_path = RUN_DIR / "agent" / f"{question_id}.json"
        trace_bytes = trace_path.read_bytes()
        trace = json.loads(trace_bytes)

        if trace["status"] != "finished":
            raise ValueError(f"轨迹未完成：{question_id}")
        if trace["question"] != question["question"]:
            raise ValueError(f"问题不一致：{question_id}")
        if trace["final_answer"] != question["answer"]:
            raise ValueError(f"答案需重新核验：{question_id}")

        seen_ids = set()
        for event in trace["events"]:
            observation = event["observation"]
            if "error" in observation:
                raise ValueError(f"轨迹含动作错误：{question_id}")
            seen_ids.update(
                doc["id"] for doc in observation.get("results", [])
            )

        gold_ids = question["supporting_doc_ids"]
        if not set(gold_ids) <= seen_ids:
            raise ValueError(f"标准证据未实际返回：{question_id}")

        # 深复制对话，不改动原始轨迹。
        messages = json.loads(json.dumps(trace["messages"]))

        if messages[-1]["role"] != "assistant":
            raise ValueError("最后一条消息不是助手回答")

        final_action = json.loads(messages[-1]["content"])
        if final_action.get("type") != "finish":
            raise ValueError("最后一条消息不是finish动作")
        if final_action["answer"] != trace["final_answer"]:
            raise ValueError("最终动作与轨迹答案不一致")

        original_ids = final_action.get("evidence_ids", [])
        final_action["evidence_ids"] = gold_ids

        # 所补引用已经在此前搜索中返回，不提前注入未来信息。
        messages[-1]["content"] = json.dumps(
            final_action, ensure_ascii=False
        )

        samples.append({
            "messages": messages,
            "metadata": {
                "question_id": question_id,
                "source_trace": str(trace_path),
                "source_trace_sha256": hashlib.sha256(
                    trace_bytes
                ).hexdigest(),
                "model": trace["model"],
                "review_basis": "assistant review of returned evidence",
                "modification": "final citation IDs corrected",
                "original_evidence_ids": original_ids,
                "corrected_evidence_ids": gold_ids,
                "queries_and_tool_results_modified": False,
                "purpose": "SFT format and loss-mask pilot only",
            },
        })

    OUTPUT_DIR.mkdir(parents=True)
    output_path = OUTPUT_DIR / "train.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

    print("Exported samples:", len(samples))
    print("Saved:", output_path)
    print("Raw traces unchanged.")
    print("Next: verify chat template and supervised tokens before training.")


if __name__ == "__main__":
    main()