import os

# 避免模型和数据缓存继续占用C盘。
os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/raw/hotpotqa"
REPO_ID = "hotpotqa/hotpot_qa"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revision = api.dataset_info(REPO_ID).sha
    files = api.list_repo_files(
        REPO_ID,
        repo_type="dataset",
        revision=revision,
    )

    # 只下载开发数据，不下载训练集。
    validation_files = sorted(
        filename for filename in files
        if filename.startswith("distractor/validation-")
        and filename.endswith(".parquet")
    )
    if not validation_files:
        raise ValueError("未找到distractor validation文件")

    records = []

    for filename in validation_files:
        print("Downloading:", filename, flush=True)
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=revision,
        )

        for row in pq.read_table(local_path).to_pylist():
            context = row["context"]
            facts = row["supporting_facts"]

            # 将HF字段结构转为原始HotpotQA JSON结构。
            records.append({
                "_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "context": [
                    [title, sentences]
                    for title, sentences in zip(
                        context["title"],
                        context["sentences"],
                        strict=True,
                    )
                ],
                "supporting_facts": [
                    [title, sentence_id]
                    for title, sentence_id in zip(
                        facts["title"],
                        facts["sent_id"],
                        strict=True,
                    )
                ],
            })

    if len({row["_id"] for row in records}) != len(records):
        raise ValueError("发现重复问题ID")

    # 使用新文件名，避免覆盖之前失败下载留下的文件。
    output_path = OUTPUT_DIR / "hotpot_dev_distractor_hf.json"
    content = json.dumps(records, ensure_ascii=False).encode("utf-8")
    output_path.write_bytes(content)

    provenance = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "revision": revision,
        "subset": "distractor",
        "split": "validation",
        "downloaded_files": validation_files,
        "record_count": len(records),
        "output_sha256": hashlib.sha256(content).hexdigest(),
        "conversion": "HF Parquet fields converted to original JSON layout",
        "license": "CC BY-SA 4.0",
    }
    (OUTPUT_DIR / "download_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Questions:", len(records))
    print("Saved:", output_path)
    print("Provenance saved.")


if __name__ == "__main__":
    main()