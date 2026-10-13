import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/sft_pilot_v1"
MAX_LENGTH = 4096


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    output_path = data_dir / "tokenized.jsonl"
    if output_path.exists():
        raise FileExistsError(
            "tokenized.jsonl已经存在，请先检查，不覆盖"
        )

    config = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8-sig")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("需要fast tokenizer提供字符偏移")

    with (data_dir / "train.jsonl").open(encoding="utf-8") as file:
        trajectories = [
            json.loads(line) for line in file if line.strip()
        ]

    examples = []

    for trajectory in trajectories:
        messages = trajectory["messages"]
        question_id = trajectory["metadata"]["question_id"]
        assistant_turn = 0

        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue

            assistant_turn += 1
            action = json.loads(message["content"])
            if action.get("type") not in {"search", "finish"}:
                raise ValueError("助手目标不是合法搜索或结束动作")

            # 与推理使用相同的前缀和关闭thinking设置。
            prompt = tokenizer.apply_chat_template(
                messages[:index],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # 结束标记也参与监督，教模型结束本次动作。
            completion = message["content"] + tokenizer.eos_token
            encoded = tokenizer(
                prompt + completion,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            boundary = len(prompt)
            labels = []

            for token_id, (start, end) in zip(input_ids, offsets):
                if end <= start:
                    raise ValueError("发现无法定位的token，停止检查")

                if end <= boundary:
                    labels.append(-100)
                elif start >= boundary:
                    labels.append(token_id)
                else:
                    raise ValueError(
                        "token跨越输入与目标边界，停止而不猜测掩码"
                    )

            if len(input_ids) > MAX_LENGTH:
                raise ValueError(
                    f"{question_id} 第{assistant_turn}个动作"
                    f"长度{len(input_ids)}超过{MAX_LENGTH}，不截断"
                )

            supervised_ids = [
                token_id
                for token_id, label in zip(input_ids, labels)
                if label != -100
            ]
            decoded_target = tokenizer.decode(
                supervised_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded_target != completion:
                raise ValueError(
                    f"{question_id}：实际监督内容与目标不一致"
                )
            if supervised_ids[-1] != tokenizer.eos_token_id:
                raise ValueError("目标未包含正确结束标记")

            examples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "metadata": {
                    "question_id": question_id,
                    "assistant_turn": assistant_turn,
                    "action_type": action["type"],
                    "sequence_tokens": len(input_ids),
                    "supervised_tokens": len(supervised_ids),
                    "model": config,
                    "enable_thinking": False,
                    "max_length": MAX_LENGTH,
                },
            })

            print(
                f"{question_id} | turn={assistant_turn} "
                f"| {action['type']} "
                f"| sequence={len(input_ids)} "
                f"| supervised={len(supervised_ids)}"
            )

    if not examples:
        raise ValueError("没有可监督的助手动作")

    # 所有检查通过之后才写文件。
    with output_path.open("x", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("\nTrajectories:", len(trajectories))
    print("Action examples:", len(examples))
    print("Longest sequence:", max(
        e["metadata"]["sequence_tokens"] for e in examples
    ))
    print("Total supervised tokens:", sum(
        e["metadata"]["supervised_tokens"] for e in examples
    ))
    print("Mask check: PASS")
    print("Saved:", output_path)
    print("\nExample supervised target:")
    print(decoded_target)


if __name__ == "__main__":
    main()