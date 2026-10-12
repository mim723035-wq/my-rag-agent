import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kzq_agent import AgentController


class FakeSearchTool:
    """模拟正常返回、空结果和工具故障。"""

    def __init__(self):
        self.calls = []

    def search(self, query, top_k=3):
        self.calls.append(query)

        if query == "empty":
            return []

        if query == "fail":
            raise RuntimeError("simulated tool failure")

        return [{
            "id": "doc_demo",
            "title": "Demo document",
            "text": "Demo evidence.",
            "score": 1.0,
        }]


def new_controller(budget=2):
    tool = FakeSearchTool()
    controller = AgentController(
        "Demo question", tool, max_searches=budget
    )
    return controller, tool


def check_invalid_actions():
    controller, tool = new_controller()

    for action in (
        None,
        {"type": "unknown"},
        {"type": "search", "query": ""},
        {"type": "search", "query": 123}
    ):
        result = controller.step(action)
        assert "error" in result

    assert controller.state["search_count"] == 0
    assert controller.state["status"] == "running"
    assert len(controller.state["events"]) == 4
    assert tool.calls == []


def check_empty_result():
    controller, tool = new_controller()
    result = controller.step({"type": "search", "query": "empty"})

    assert result["results"] == []
    assert controller.state["search_count"] == 1
    assert controller.state["status"] == "running"
    assert tool.calls == ["empty"]


def check_tool_failure():
    controller, tool = new_controller()
    result = controller.step({"type": "search", "query": "fail"})

    assert result["error"] == "search_failed"
    assert controller.state["search_count"] == 1
    assert controller.state["status"] == "running"
    assert len(controller.state["events"]) == 1
    assert tool.calls == ["fail"]


def check_budget_and_finish():
    controller, tool = new_controller(budget=1)

    controller.step({"type": "search", "query": "normal"})
    denied = controller.step({"type": "search", "query": "another"})

    assert "error" in denied
    assert denied["remaining_searches"] == 0
    assert controller.state["search_count"] == 1
    assert tool.calls == ["normal"]

    # 搜索预算耗尽后，仍然允许提交回答。
    accepted = controller.step({
        "type": "finish",
        "answer": "Demo answer",
        "evidence_ids": ["doc_demo"],
    })

    assert accepted["accepted"] is True
    assert controller.state["status"] == "finished"


def check_unseen_citation_and_recovery():
    controller, _ = new_controller()

    rejected = controller.step({
        "type": "finish",
        "answer": "Demo answer",
        "evidence_ids": ["unseen_doc"],
    })

    assert "error" in rejected
    assert controller.state["status"] == "running"
    assert controller.state["final_answer"] is None

    # 无效动作后仍允许继续搜索，再提交合法引用。
    controller.step({"type": "search", "query": "normal"})
    accepted = controller.step({
        "type": "finish",
        "answer": "Demo answer",
        "evidence_ids": ["doc_demo"],
    })

    assert accepted["accepted"] is True
    assert len(controller.state["events"]) == 3


def check_invalid_finish_fields():
    controller, _ = new_controller()

    for action in (
        {"type": "finish", "answer": "", "evidence_ids": []},
        {"type": "finish", "answer": "A", "evidence_ids": "doc_demo"},
        {"type": "finish", "answer": "A", "evidence_ids": [123]},
    ):
        assert "error" in controller.step(action)

    assert controller.state["status"] == "running"


def check_zero_budget_abstention():
    controller, tool = new_controller(budget=0)

    assert "error" in controller.step({
        "type": "search",
        "query": "normal",
    })

    result = controller.step({
        "type": "finish",
        "answer": "Cannot determine",
        "evidence_ids": [],
    })

    assert result["accepted"] is True
    assert controller.state["search_count"] == 0
    assert tool.calls == []


def check_finished_task_rejects_actions():
    controller, _ = new_controller()
    controller.step({
        "type": "finish",
        "answer": "Cannot determine",
        "evidence_ids": [],
    })
    event_count = len(controller.state["events"])

    try:
        controller.step({"type": "search", "query": "normal"})
    except ValueError:
        pass
    else:
        raise AssertionError("Finished task accepted another action")

    assert len(controller.state["events"]) == event_count


def main():
    checks = [
        check_invalid_actions,
        check_empty_result,
        check_tool_failure,
        check_budget_and_finish,
        check_unseen_citation_and_recovery,
        check_invalid_finish_fields,
        check_zero_budget_abstention,
        check_finished_task_rejects_actions,
    ]

    for check in checks:
        check()
        print(f"PASS: {check.__name__}")

    print(f"\nAll {len(checks)} controller checks passed.")


if __name__ == "__main__":
    main()