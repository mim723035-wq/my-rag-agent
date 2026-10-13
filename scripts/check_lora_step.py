import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU不可用")

    torch.manual_seed(42)

    config = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8-sig")
    )
    with (ROOT / "data/sft_pilot_v1/tokenized.jsonl").open(
        encoding="utf-8"
    ) as file:
        rows = [json.loads(line) for line in file if line.strip()]

    # 检查最长样本，包括修订后的最终引用。
    sample = max(rows, key=lambda row: len(row["input_ids"]))

    if sample["metadata"]["model"] != config:
        raise ValueError("训练数据与当前模型配置不一致")

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    model.print_trainable_parameters()

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise ValueError("没有可训练参数")
    if any("lora_" not in name for name in trainable):
        raise ValueError("存在意外解冻的非LoRA参数")

    # 在CPU保留训练前副本，用于检查是否真的更新。
    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in trainable.items()
    }

    batch = {
        key: torch.tensor(
            [sample[key]], dtype=torch.long, device="cuda"
        )
        for key in ("input_ids", "attention_mask", "labels")
    }
    supervised_tokens = (
        batch["labels"][:, 1:] != -100
    ).sum().item()
    if supervised_tokens == 0:
        raise ValueError("没有有效监督token")

    optimizer = torch.optim.AdamW(
        list(trainable.values()),
        lr=1e-4,
        weight_decay=0.0,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    output = model(**batch)
    loss = output.loss

    if not torch.isfinite(loss).item():
        raise RuntimeError("loss不是有限值")

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in trainable.values()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("没有LoRA梯度")
    if not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("梯度存在NaN或Inf")
    if not any(torch.any(g != 0).item() for g in gradients):
        raise RuntimeError("LoRA梯度全部为零")

    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(trainable.values()), max_norm=1.0
    )
    optimizer.step()
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - started

    changes = {
        name: (
            parameter.detach().float().cpu() - before[name]
        ).abs().max().item()
        for name, parameter in trainable.items()
    }
    changed_count = sum(value > 0 for value in changes.values())

    if changed_count == 0:
        raise RuntimeError("optimizer.step后LoRA参数没有变化")

    report = {
        "experiment": "one_step_lora_smoke_check",
        "model": config,
        "question_id": sample["metadata"]["question_id"],
        "action_type": sample["metadata"]["action_type"],
        "sequence_tokens": len(sample["input_ids"]),
        "supervised_tokens": supervised_tokens,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        },
        "learning_rate": 1e-4,
        "gradient_checkpointing": True,
        "loss_before_update": loss.item(),
        "gradient_norm_before_clipping": float(grad_norm),
        "trainable_parameter_count": sum(
            p.numel() for p in trainable.values()
        ),
        "changed_parameter_tensors": changed_count,
        "maximum_parameter_change": max(changes.values()),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
        "step_seconds": elapsed,
        "limitation": (
            "One short-sample update only; "
            "no capability evaluation or checkpoint produced."
        ),
    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"lora_step_{stamp}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nLoRA step: PASS")
    print("Saved:", path)


if __name__ == "__main__":
    main()