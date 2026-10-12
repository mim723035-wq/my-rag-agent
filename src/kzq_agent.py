import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from search import SearchTool

ROOT = Path(__file__).resolve().parents[1]


class AgentController:
    def __init__(self, question, search_tool, max_searches=2):
        if max_searches < 0:
            raise ValueError("搜索预算不能为负数")

        self.search_tool = search_tool
        self.seen_document_ids = set()
        self.state = {
            "question": question,
            "max_searches": max_searches,
            "search_count": 0,
            "status": "running",
            "final_answer": None,
            "evidence_ids": [],
            "events": [],
        }

    def record(self, action, observation):
        self.state["events"].append({
            "step": len(self.state["events"]) + 1,
            "action": action,
            "observation": observation,
        })
        return observation

    def step(self, action):
        if self.state["status"] != "running":
            raise ValueError("任务已经结束")

        if not isinstance(action, dict):
            return self.record(action, {
                "error": "动作必须是 JSON 对象"
            })

        action_type = action.get("type")

        if action_type == "search":
            query = action.get("query")

            if not isinstance(query, str) or not query.strip():
                return self.record(action, {
                    "error": "query 必须是非空字符串"
                })

            if self.state["search_count"] >= self.state["max_searches"]:
                return self.record(action, {
                    "error": "搜索预算已耗尽，请提交答案或说明证据不足",
                    "remaining_searches": 0,
                })

            # 合法搜索一旦发起就计入预算，包括空结果或工具失败。
            self.state["search_count"] += 1

            try:
                results = self.search_tool.search(query.strip(), top_k=3)
                self.seen_document_ids.update(
                    doc["id"] for doc in results
                )
                observation = {"results": results}
            except Exception as error:
                observation = {
                    "error": "search_failed",
                    "detail": str(error),
                }

            observation["remaining_searches"] = (
                self.state["max_searches"]
                - self.state["search_count"]
            )
            return self.record(action, observation)

        if action_type == "finish":
            answer = action.get("answer")
            evidence_ids = action.get("evidence_ids", [])

            if not isinstance(answer, str) or not answer.strip():
                return self.record(action, {
                    "error": "answer 必须是非空字符串"
                })

            if (
                not isinstance(evidence_ids, list)
                or not all(isinstance(item, str) for item in evidence_ids)
            ):
                return self.record(action, {
                    "error": "evidence_ids 必须是字符串列表"
                })

            unseen_ids = set(evidence_ids) - self.seen_document_ids
            if unseen_ids:
                return self.record(action, {
                    "error": "不能引用本次搜索未返回的文档",
                    "unseen_ids": sorted(unseen_ids),
                })

            self.state["status"] = "finished"
            self.state["final_answer"] = answer.strip()
            self.state["evidence_ids"] = list(dict.fromkeys(evidence_ids))

            return self.record(action, {
                "accepted": True,
                "note": "已接收回答；尚未验证答案与证据是否正确",
            })

        return self.record(action, {
            "error": "未知动作，只支持 search 和 finish"
        })


def save_trace(controller):
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"manual_{timestamp}_{uuid4().hex[:8]}.json"

    # 标明这些动作来自人工，不能误认为模型生成轨迹。
    payload = {
        "actor": "human",
        **controller.state,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=8)
    args = parser.parse_args()

    if args.max_actions < 1:
        parser.error("--max-actions 必须大于等于 1")

    tool = SearchTool(ROOT / "data" / "documents.jsonl")
    controller = AgentController(
        args.question, tool, args.max_searches
    )

    print(f"问题：{args.question}")
    print("每次输入一行 JSON 动作。输入 exit 可中止并保存。")

    try:
        for _ in range(args.max_actions):
            raw = input("\n动作> ").strip()

            if raw.lower() == "exit":
                controller.state["status"] = "aborted"
                break

            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                observation = controller.record(
                    {"raw": raw},
                    {"error": "JSON 格式错误"},
                )
            else:
                observation = controller.step(action)

            print(json.dumps(
                observation, ensure_ascii=False, indent=2
            ))

            if controller.state["status"] != "running":
                break

        if controller.state["status"] == "running":
            controller.state["status"] = "action_limit"

    except (KeyboardInterrupt, EOFError):
        controller.state["status"] = "aborted"

    finally:
        path = save_trace(controller)
        print(f"\n结束状态：{controller.state['status']}")
        print(f"搜索次数：{controller.state['search_count']}")
        print(f"轨迹已保存：{path}")


if __name__ == "__main__":
    main()