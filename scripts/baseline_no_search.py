import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    config = json.loads(
        (ROOT / "configs" / "model.json").read_text(encoding="utf-8")
    )
    questions = read_jsonl(ROOT / "data" / "questions.jsonl")
    documents = read_jsonl(ROOT / "data" / "documents.jsonl")

    question = next(q for q in questions if q["id"] == "q_001")
    document_map = {doc["id"]: doc for doc in documents}

    # 仅诊断程序使用标准证据选择文档，不把标准答案传给模型。
        # 保留正确证据的相对顺序，再加入之前轨迹中的干扰文档。
    evidence_ids = [
        "doc_001",
        "doc_002",
        "doc_006",
        "doc_004",
        "doc_005",
    ]
    evidence = [document_map[doc_id] for doc_id in evidence_ids]

    messages = [
        {
            "role": "system",
            "content": (
                "请回答用户问题。本次没有搜索工具，也没有提供文档。"
                "不知道时请回答“无法确定”。"
                "只输出一个JSON对象："
                '{"answer":"简短答案","evidence_ids":[]}。'
                "不要输出Markdown或额外解释。"
            ),
        },
        {
            "role": "user",
            "content": question["question"],
        },
    ]

    print("加载模型……", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    model.eval()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()

    print("\n模型原始输出：", flush=True)
    print(raw, flush=True)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"no_search_{question['id']}.json"
    path.write_text(
        json.dumps({
            "experiment": "no_search_baseline",
            "model": config,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 192,
            "question_id": question["id"],
            "messages": messages,
            "raw_output": raw,
            "input_tokens": inputs["input_ids"].shape[1],
            "output_tokens": len(new_tokens),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n诊断结果已保存：{path}", flush=True)


if __name__ == "__main__":
    main()