import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/sft_pilot_v1/tokenized.jsonl"


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("epochs必须至少为1")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU不可用")

    torch.manual_seed(42)
    rng = random.Random(42)

    config = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8-sig")
    )
    data_bytes = DATA_PATH.read_bytes()
    rows = [
        json.loads(line)
        for line in data_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("训练数据为空")

    for row in rows:
        if row["metadata"]["model"] != config:
            raise ValueError("数据与模型配置不一致")
        if len(row["input_ids"]) > 1721:
            raise ValueError("存在超过已验证训练长度的样本")
        if not (
            len(row["input_ids"])
            == len(row["labels"])
            == len(row["attention_mask"])
        ):
            raise ValueError("输入与标签长度不一致")
        if not any(label != -100 for label in row["labels"][1:]):
            raise ValueError("样本没有有效监督目标")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "outputs" / f"sft_pilot_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    adapter_dir = run_dir / "adapter"

    training_config = {
        "experiment": "sft_pipeline_pilot",
        "model": config,
        "data_file": str(DATA_PATH),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "sample_count": len(rows),
        "unique_question_count": len({
            row["metadata"]["question_id"] for row in rows
        }),
        "epochs": args.epochs,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "gradient_checkpointing": True,
        "precision": "BF16 base; PEFT adapter dtype recorded at runtime",
        "seed": 42,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "purpose": "Pipeline validation, not evidence of capability gains",
    }
    write_json(run_dir / "training_config.json", training_config)

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    model.print_trainable_parameters()

    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if any(
        "lora_" not in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise ValueError("存在意外解冻的非LoRA参数")

    training_config["trainable_dtypes"] = sorted({
        str(parameter.dtype) for parameter in trainable
    })
    write_json(run_dir / "training_config.json", training_config)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        weight_decay=0.0,
        foreach=False,
    )

    torch.cuda.reset_peak_memory_stats()
    step = 0

    try:
        with (run_dir / "steps.jsonl").open(
            "w", encoding="utf-8"
        ) as log:
            for epoch in range(1, args.epochs + 1):
                order = list(range(len(rows)))
                rng.shuffle(order)

                for index in order:
                    row = rows[index]
                    batch = {
                        key: torch.tensor(
                            [row[key]],
                            dtype=torch.long,
                            device="cuda",
                        )
                        for key in (
                            "input_ids", "attention_mask", "labels"
                        )
                    }

                    optimizer.zero_grad(set_to_none=True)
                    output = model(**batch)
                    loss = output.loss

                    if not torch.isfinite(loss).item():
                        raise RuntimeError("loss出现NaN或Inf")

                    loss_value = loss.detach().item()
                    loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable,
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    step += 1

                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "question_id": row["metadata"]["question_id"],
                        "action_type": row["metadata"]["action_type"],
                        "sequence_tokens": len(row["input_ids"]),
                        "supervised_tokens": row["metadata"][
                            "supervised_tokens"
                        ],
                        "loss_before_update": loss_value,
                        "gradient_norm_before_clipping": float(grad_norm),
                    }
                    log.write(json.dumps(entry) + "\n")
                    log.flush()

                    print(
                        f"step={step} epoch={epoch} "
                        f"action={entry['action_type']} "
                        f"loss={loss_value:.4f}",
                        flush=True,
                    )

                    del output, loss, batch

        model.save_pretrained(
            adapter_dir, safe_serialization=True
        )
        write_json(run_dir / "result.json", {
            "status": "completed",
            "optimizer_steps": step,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
            "adapter_dir": str(adapter_dir),
            "limitation": (
                "Only 5 training questions; "
                "no independent capability evaluation yet."
            ),
        })

    except Exception as error:
        write_json(run_dir / "result.json", {
            "status": "failed",
            "completed_optimizer_steps": step,
            "error": f"{type(error).__name__}: {error}",
        })
        print("Failure log:", run_dir, flush=True)
        raise

    print("\nTraining steps:", step)
    print("Adapter:", adapter_dir)
    print("Logs:", run_dir)


if __name__ == "__main__":
    main()