import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kzq_agent import AgentController
from search import SearchTool

DATA_DIR = ROOT / "data/sft_recovery_v1"
SOURCE = DATA_DIR / "search_revision_train.jsonl"
OUTPUT = DATA_DIR / "citation_recovery_train.jsonl"
AUDIT = DATA_DIR / "citation_recovery_audit.json"
RUN_DIR = ROOT / "outputs/baselines_20261013T092302Z_9a649338"


def feedback_message(observation):
    return {
        "role": "user",
        "content": (
            "Controller result:\n"
            + json.dumps(observation, ensure_ascii=False)
            + "\nChoose your next action."
        ),
    }


def action_message(action):
    return {
        "role": "assistant",
        "content": json.dumps(action, ensure_ascii=False),
    }


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("输出已存在，停止以避免覆盖")

    source_bytes = SOURCE.read_bytes()
    rows = [
        json.loads(line)
        for line in source_bytes.decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(rows) != 2:
        raise ValueError("本步预期使用两条已审核的查询修订示范")

    tool = SearchTool(RUN_DIR / "documents.jsonl")
    questions = {}
    with (RUN_DIR / "questions.jsonl").open(
        encoding="utf-8-sig"
    ) as file:
        for line in file:
            if line.strip():
                q = json.loads(line)
                questions[q["id"]] = q

    samples = []
    audits = []

    for row in rows:
        qid = row["metadata"]["question_id"]
        question = questions[qid]
        if question.get("split") != "train":
            raise ValueError("禁止使用非训练题")

        original = row["messages"]
        controller = AgentController(
            question["question"], tool, max_searches=3
        )
        messages = copy.deepcopy(original[:2])

        if [m["role"] for m in original] != [
            "system", "user", "assistant", "user",
            "assistant", "user", "assistant",
        ]:
            raise ValueError(f"源对话结构异常：{qid}")

        documents_path = RUN_DIR / "documents.jsonl"
        if hashlib.sha256(documents_path.read_bytes()).hexdigest() != (
            row["metadata"]["documents_sha256"]
        ):
            raise ValueError("文档库发生变化")

        # 实际执行搜索，核对工具结果。
        for index in (2, 4):
            action = json.loads(original[index]["content"])
            if action.get("type") != "search":
                raise ValueError("预期搜索动作")

            observation = controller.step(action)

            saved_text = original[index + 1]["content"]
            saved_observation = json.loads(
                saved_text.split("Controller result:\n", 1)[1]
                .rsplit("\nChoose your next action.", 1)[0]
            )
            if observation != saved_observation:
                raise ValueError(f"检索重放不一致：{qid}")

            messages.append(action_message(action))
            messages.append(feedback_message(observation))

        correct = json.loads(original[-1]["content"])
        if (
            correct.get("type") != "finish"
            or not correct.get("evidence_ids")
        ):
            raise ValueError("源示范缺少有效finish动作")

        # 故意拼错一个ID，获取真实控制器错误反馈。
        bad = copy.deepcopy(correct)
        corrupted_id = bad["evidence_ids"][0] + "4"
        while corrupted_id in controller.seen_document_ids:
            corrupted_id += "4"
        bad["evidence_ids"][0] = corrupted_id

        error_observation = controller.step(bad)
        if (
            error_observation.get("unseen_ids") != [corrupted_id]
            or "error" not in error_observation
            or controller.state["status"] != "running"
        ):
            raise ValueError("错误引用未按预期被拒绝")

        messages.append(action_message(bad))
        injected_index = len(messages) - 1
        messages.append(feedback_message(error_observation))

        # 提交正确引用，验证恢复动作被接受。
        accepted = controller.step(correct)
        if (
            accepted.get("accepted") is not True
            or controller.state["status"] != "finished"
        ):
            raise ValueError("恢复动作未通过控制器校验")

        messages.append(action_message(correct))
        target_index = len(messages) - 1

        metadata = copy.deepcopy(row["metadata"])
        metadata.update({
            "construction": "assistant_designed_citation_recovery",
            "model_generated": False,
            "source_demonstrations_sha256": hashlib.sha256(
                source_bytes
            ).hexdigest(),
            "error_recovery_example": True,
            "injected_error": "one unseen citation ID",
            "injected_message_index": injected_index,
            "supervised_message_indices": [target_index],
            "purpose": "repair citation after controller feedback",
            "review_basis": (
                "previous evidence review and actual controller execution"
            ),
        })

        samples.append({
            "messages": messages,
            "metadata": metadata,
        })
        audits.append({
            "question_id": qid,
            "events": controller.state["events"],
            "status": controller.state["status"],
            "injected_message_index": injected_index,
            "supervised_message_indices": [target_index],
        })

        print(f"{qid}: rejected bad ID, accepted repaired ID")
        print(
            f"Injected index: {injected_index}; "
            f"supervised index: {target_index}"
        )

    with OUTPUT.open("x", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    AUDIT.write_text(
        json.dumps(audits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRecovery samples:", len(samples))
    print("Supervised action examples:", len(samples))
    print("Saved:", OUTPUT)
    print("Audit:", AUDIT)


if __name__ == "__main__":
    main()