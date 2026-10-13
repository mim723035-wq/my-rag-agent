import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def weighted_loss(rows, field):
    token_count = sum(row["supervised_tokens"] for row in rows)
    return sum(
        row[field] * row["supervised_tokens"] for row in rows
    ) / token_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    args = parser.parse_args()
    adapter_dir = args.adapter_dir.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU不可用")

    training_config = json.loads(
        (adapter_dir.parent / "training_config.json").read_text(
            encoding="utf-8"
        )
    )
    config = training_config["model"]

    data_path = Path(training_config["data_file"])
    data_bytes = data_path.read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != (
        training_config["data_sha256"]
    ):
        raise ValueError("训练数据与保存的哈希不一致")

    samples = [
        json.loads(line)
        for line in data_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError("训练数据为空")

    print("Loading model and adapter...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).to("cuda")

    model = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=False
    )
    model.eval()
    model.config.use_cache = False

    results = []

    for sample in samples:
        batch = {
            key: torch.tensor(
                [sample[key]],
                dtype=torch.long,
                device="cuda",
            )
            for key in ("input_ids", "attention_mask", "labels")
        }

        # 因果语言模型会移位标签，按实际目标数加权。
        supervised_tokens = (
            batch["labels"][:, 1:] != -100
        ).sum().item()
        if supervised_tokens == 0:
            raise ValueError("样本没有有效监督目标")

        with torch.inference_mode():
            with model.disable_adapter():
                output = model(**batch)
                base_loss = output.loss.item()
                del output

            output = model(**batch)
            adapter_loss = output.loss.item()
            del output

        if not (
            torch.isfinite(torch.tensor(base_loss)).item()
            and torch.isfinite(torch.tensor(adapter_loss)).item()
        ):
            raise RuntimeError("损失含NaN或Inf")

        metadata = sample["metadata"]
        row = {
            "question_id": metadata["question_id"],
            "assistant_turn": metadata["assistant_turn"],
            "action_type": metadata["action_type"],
            "supervised_tokens": supervised_tokens,
            "base_loss": base_loss,
            "adapter_loss": adapter_loss,
            # 正数表示适配器在该目标上的损失更低。
            "loss_reduction": base_loss - adapter_loss,
        }
        results.append(row)

        print(
            f"{row['question_id']} "
            f"turn={row['assistant_turn']} "
            f"{row['action_type']} | "
            f"base={base_loss:.6f} "
            f"adapter={adapter_loss:.6f}",
            flush=True,
        )
        del batch

    groups = defaultdict(list)
    for row in results:
        groups[row["action_type"]].append(row)

    summary = {
        "experiment": "training_target_loss_comparison",
        "adapter_dir": str(adapter_dir),
        "model": config,
        "sample_count": len(results),
        "total_supervised_tokens": sum(
            row["supervised_tokens"] for row in results
        ),
        "base_token_weighted_loss": weighted_loss(results, "base_loss"),
        "adapter_token_weighted_loss": weighted_loss(
            results, "adapter_loss"
        ),
        "samples_with_lower_loss": sum(
            row["adapter_loss"] < row["base_loss"]
            for row in results
        ),
        "by_action_type": {
            action_type: {
                "sample_count": len(rows),
                "supervised_tokens": sum(
                    row["supervised_tokens"] for row in rows
                ),
                "base_loss": weighted_loss(rows, "base_loss"),
                "adapter_loss": weighted_loss(rows, "adapter_loss"),
            }
            for action_type, rows in groups.items()
        },
        "limitations": [
            "Uses seen training targets, not independent evaluation.",
            "Loss includes citation IDs and EOS, not only answer words.",
            "Lower target loss does not guarantee a changed greedy output.",
        ],
    }

    report_path = adapter_dir.parent / "loss_comparison.json"
    report_path.write_text(
        json.dumps({
            "summary": summary,
            "details": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:", report_path)


if __name__ == "__main__":
    main()