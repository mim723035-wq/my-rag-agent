import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    args = parser.parse_args()
    adapter_dir = args.adapter_dir.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU不可用")

    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / filename).is_file():
            raise FileNotFoundError(adapter_dir / filename)

    training_config = json.loads(
        (adapter_dir.parent / "training_config.json").read_text(
            encoding="utf-8"
        )
    )
    config = training_config["model"]

    data_path = Path(training_config["data_file"])
    data_bytes = data_path.read_bytes()

    import hashlib
    if hashlib.sha256(data_bytes).hexdigest() != (
        training_config["data_sha256"]
    ):
        raise ValueError("训练数据已变化，请使用训练时的数据版本")

    rows = [
        json.loads(line)
        for line in data_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    # 检查最长的最终回答样本，覆盖修订后的完整引用。
    sample = max(
        (row for row in rows
         if row["metadata"]["action_type"] == "finish"),
        key=lambda row: len(row["input_ids"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"]
    )
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
    model.config.use_cache = True

    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    if not lora_parameters:
        raise ValueError("没有加载到LoRA参数")
    if not all(
        torch.isfinite(parameter).all().item()
        for parameter in lora_parameters
    ):
        raise ValueError("适配器参数含NaN或Inf")

    # 监督区之前是推理前缀，不包含目标答案。
    boundary = next(
        index for index, label in enumerate(sample["labels"])
        if label != -100
    )
    prompt_ids = sample["input_ids"][:boundary]
    target_ids = sample["input_ids"][boundary:]

    inputs = torch.tensor(
        [prompt_ids], dtype=torch.long, device="cuda"
    )
    attention_mask = torch.ones_like(inputs)

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        output[0, inputs.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    target = tokenizer.decode(
        target_ids, skip_special_tokens=True
    ).strip()

    parsed = None
    parse_error = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        parse_error = str(error)

    report = {
        "experiment": "adapter_reload_inference_check",
        "adapter_dir": str(adapter_dir),
        "model": config,
        "question_id": sample["metadata"]["question_id"],
        "input_tokens": len(prompt_ids),
        "generated_tokens": output.shape[1] - inputs.shape[1],
        "raw_output": raw,
        "training_target": target,
        "json_parse_error": parse_error,
        "parsed_output": parsed,
        "limitation": (
            "Seen training context; validates reload and inference only, "
            "not autonomous search or generalization."
        ),
    }

    path = adapter_dir.parent / "reload_check.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Adapter reload: PASS")
    print("\nGenerated output:\n", raw)
    print("\nTraining target:\n", target)
    print("\nJSON parse error:", parse_error)
    print("Saved:", path)


if __name__ == "__main__":
    main()