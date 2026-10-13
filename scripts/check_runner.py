import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_baselines as runner


class FakeSearchTool:
    def search(self, query, top_k=3):
        return []


def make_args(max_actions=3, max_total_tokens=100):
    return SimpleNamespace(
        max_searches=2,
        max_actions=max_actions,
        max_total_tokens=max_total_tokens,
        context_limit=4096,
        language="en",
    )


def fake_call(raw, output_tokens=10):
    return {
        "raw_output": raw,
        "input_tokens": 20,
        "output_tokens": output_tokens,
        "max_new_tokens_this_call": output_tokens,
        "generation_seconds": 0.0,
    }


def run_with_outputs(outputs, args):
    with patch.object(runner, "generate", side_effect=outputs) as mocked:
        result = runner.run_one(
            mode="agent",
            question={"id": "check_q", "question": "Demo question"},
            model=None,
            tokenizer=None,
            tool=FakeSearchTool(),
            config={"model_id": "fake", "revision": "fake"},
            args=args,
        )
    return result, mocked


def check_invalid_json_recovery():
    result, mocked = run_with_outputs([
        fake_call("not valid json"),
        fake_call(
            '{"type":"finish","answer":"Cannot determine",'
            '"evidence_ids":[]}'
        ),
    ], make_args())

    assert result["status"] == "finished"
    assert len(result["events"]) == 2
    assert "error" in result["events"][0]["observation"]
    assert mocked.call_count == 2


def check_action_limit():
    # 模拟模型连续输出无法执行的动作。
    result, mocked = run_with_outputs([
        fake_call('{"type":"unknown"}'),
        fake_call('{"type":"unknown"}'),
        fake_call('{"type":"unknown"}'),
    ], make_args(max_actions=3))

    assert result["status"] == "action_limit"
    assert len(result["model_calls"]) == 3
    assert mocked.call_count == 3
    assert result["final_answer"] is None


def check_total_token_limit():
    result, mocked = run_with_outputs([
        fake_call('{"type":"unknown"}', output_tokens=10),
        fake_call('{"type":"unknown"}', output_tokens=10),
    ], make_args(max_actions=5, max_total_tokens=20))

    assert result["status"] == "token_limit"
    assert result["total_output_tokens"] == 20
    assert mocked.call_count == 2

    # 确认每次调用获得的是剩余预算。
    remaining_budgets = [
        call.args[3] for call in mocked.call_args_list
    ]
    assert remaining_budgets == [20, 10]


def check_context_limit():
    # generate 返回 None 表示没有剩余上下文空间。
    result, mocked = run_with_outputs(
        [None], make_args()
    )

    assert result["status"] == "context_limit"
    assert result["model_calls"] == []
    assert result["total_output_tokens"] == 0
    assert mocked.call_count == 1


def check_runtime_failure():
    result, _ = run_with_outputs(
        [RuntimeError("simulated generation failure")],
        make_args(),
    )

    assert result["status"] == "runtime_error"
    assert "simulated generation failure" in result["error"]


def main():
    checks = [
        check_invalid_json_recovery,
        check_action_limit,
        check_total_token_limit,
        check_context_limit,
        check_runtime_failure,
    ]

    for check in checks:
        check()
        print(f"PASS: {check.__name__}")

    print(f"\nAll {len(checks)} runner checks passed.")


if __name__ == "__main__":
    main()