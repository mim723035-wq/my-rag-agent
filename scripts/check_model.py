import os

# 必须在导入 Hugging Face 相关库之前设置。
os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import json
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-1.7B"
CONFIG_PATH = ROOT / "configs" / "model.json"

if not torch.cuda.is_available():
    raise RuntimeError("GPU 不可用，请先运行 check_gpu.py")

# 首次解析并保存模型版本；后续运行使用同一个版本。
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

if CONFIG_PATH.exists():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["model_id"] != MODEL_ID:
        raise ValueError("已有模型配置与当前模型不一致")
else:
    revision = HfApi().model_info(MODEL_ID).sha
    config = {
        "model_id": MODEL_ID,
        "revision": revision,
        "enable_thinking": False,
    }
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

print("模型：", MODEL_ID)
print("版本：", config["revision"])
print("缓存目录：", os.environ["HF_HOME"])
print("首次运行需要下载模型，请等待。")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=config["revision"],
)

# 先在 CPU 加载，再明确移到 GPU，避免自动卸载掩盖显存问题。
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=config["revision"],
    dtype=torch.bfloat16,
    use_safetensors=True,
).to("cuda")
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "你是文档搜索助手。根据问题和已有搜索结果，"
            "生成一次用于补齐缺失信息的下一步查询。"
            "只输出一个JSON对象："
            '{"type":"search","query":"查询词"}。'
            "不要回答问题，不要输出额外解释。"
        ),
    },
    {
        "role": "user",
        "content": (
            "问题：林遥执导的2021年上映的电影，"
            "在2022年晨光电影节获得了什么奖？\n\n"
            "已有搜索结果：\n"
            "[doc_001] 林遥是一名导演。"
            "她执导的电影《星港》于2021年上映。\n\n"
            "请生成下一次搜索查询。"
        ),
    },
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=config["enable_thinking"],
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to("cuda")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# 只解码新生成部分，排除输入提示词。
new_tokens = output[0, inputs["input_ids"].shape[1]:]
raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print("\n模型原始输出：")
print(raw)
print(f"\n输入 token：{inputs['input_ids'].shape[1]}")
print(f"生成 token：{len(new_tokens)}")
print(f"生成耗时：{elapsed:.2f} 秒")
print(
    "PyTorch 峰值已分配显存："
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
)

try:
    action = json.loads(raw)
except json.JSONDecodeError:
    print("\n动作检查失败：输出不是合法 JSON。保留原始输出用于排查。")
else:
    valid = (
        isinstance(action, dict)
        and action.get("type") == "search"
        and isinstance(action.get("query"), str)
        and bool(action["query"].strip())
    )
    if valid:
        print("\n动作检查通过：模型生成了合法搜索动作。")
    else:
        print("\n动作检查失败：JSON 合法，但不符合搜索动作格式。")