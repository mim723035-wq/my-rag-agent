import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def collate(examples, pad_token_id):
    """右侧补齐；输入、注意力、标签使用不同的补充值。"""
    max_length = max(len(row["input_ids"]) for row in examples)

    input_ids = []
    attention_masks = []
    labels = []

    for row in examples:
        padding = max_length - len(row["input_ids"])

        input_ids.append(
            row["input_ids"] + [pad_token_id] * padding
        )
        attention_masks.append(
            row["attention_mask"] + [0] * padding
        )
        labels.append(
            row["labels"] + [-100] * padding
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(
            attention_masks, dtype=torch.long
        ),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/sft_pilot_v1",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    with (data_dir / "tokenized.jsonl").open(
        encoding="utf-8"
    ) as file:
        rows = [json.loads(line) for line in file if line.strip()]

    config = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8-sig")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
    )

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer没有pad token")

    # 用最短与最长样本，确保实际覆盖补齐行为。
    selected = [
        min(rows, key=lambda row: len(row["input_ids"])),
        max(rows, key=lambda row: len(row["input_ids"])),
    ]
    batch = collate(selected, tokenizer.pad_token_id)

    for index, original in enumerate(selected):
        length = len(original["input_ids"])

        assert batch["input_ids"][index, :length].tolist() == (
            original["input_ids"]
        )
        assert batch["labels"][index, :length].tolist() == (
            original["labels"]
        )
        assert batch["attention_mask"][index, :length].tolist() == (
            original["attention_mask"]
        )

        assert torch.all(
            batch["input_ids"][index, length:] == tokenizer.pad_token_id
        ).item()
        assert torch.all(
            batch["attention_mask"][index, length:] == 0
        ).item()
        assert torch.all(
            batch["labels"][index, length:] == -100
        ).item()

        supervised = batch["labels"][index] != -100
        assert torch.all(
            batch["attention_mask"][index][supervised] == 1
        ).item()
        assert torch.equal(
            batch["labels"][index][supervised],
            batch["input_ids"][index][supervised],
        )

        expected_count = sum(
            label != -100 for label in original["labels"]
        )
        assert supervised.sum().item() == expected_count

        # 自回归损失通常将标签左移一位；
        # 第一位置不应包含目标，移位后监督数应保持一致。
        assert batch["labels"][index, 0].item() == -100
        assert (
            batch["labels"][index, 1:] != -100
        ).sum().item() == expected_count

        print(
            f"PASS sample {index}: "
            f"original_length={length}, "
            f"supervised_tokens={expected_count}"
        )

    print("Batch shape:", tuple(batch["input_ids"].shape))
    print("Padding mask: PASS")
    print("Original labels preserved: PASS")
    print("Shifted target count: PASS")


if __name__ == "__main__":
    main()