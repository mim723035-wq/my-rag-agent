import os

os.environ["HF_HOME"] = r"D:\search-agent-cache\huggingface"

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/hotpotqa"
REPO_ID = "hotpotqa/hotpot_qa"

OUTPUT = RAW_DIR / "hotpot_train_distractor_hf.json"
PARTIAL = RAW_DIR / "hotpot_train_distractor_hf.json.partial"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        raise FileExistsError(
            "训练文件已存在，停止下载，避免覆盖："
            + str(OUTPUT)
        )

    # 与开发数据采用同一个数据仓库revision。
    provenance_path = RAW_DIR / "download_provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(
            "缺少开发数据的download_provenance.json，请先核对来源记录"
        )

    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8-sig")
    )
    expected_source = f"https://huggingface.co/datasets/{REPO_ID}"

    if provenance.get("source") != expected_source:
        raise ValueError("开发数据来源与预期仓库不同")

    revision = provenance["revision"]
    api = HfApi()
    filenames = sorted(
        name
        for name in api.list_repo_files(
            REPO_ID,
            repo_type="dataset",
            revision=revision,
        )
        if name.startswith("distractor/train-")
        and name.endswith(".parquet")
    )

    if not filenames:
        raise ValueError("未找到训练文件")

    seen_ids = set()
    count = 0

    # 按批次转换，不把全部训练数据一次性载入内存。
    # 未完成文件使用.partial后缀，不作为正式数据使用。
    with PARTIAL.open("w", encoding="utf-8") as output:
        output.write("[\n")
        first = True

        for filename in filenames:
            print("Downloading:", filename, flush=True)
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                revision=revision,
            )

            parquet = pq.ParquetFile(local_path)

            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    question_id = row["id"]
                    if question_id in seen_ids:
                        raise ValueError(
                            f"Duplicate question ID: {question_id}"
                        )
                    seen_ids.add(question_id)

                    context = row["context"]
                    facts = row["supporting_facts"]

                    record = {
                        "_id": question_id,
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
                    }

                    if not first:
                        output.write(",\n")
                    output.write(json.dumps(record, ensure_ascii=False))
                    first = False
                    count += 1

            print("Converted questions:", count, flush=True)

        output.write("\n]\n")

    PARTIAL.replace(OUTPUT)

    source_record = {
        "source": expected_source,
        "revision": revision,
        "subset": "distractor",
        "split": "train",
        "downloaded_files": filenames,
        "record_count": count,
        "output_sha256": file_sha256(OUTPUT),
        "conversion": "HF Parquet fields converted to original JSON layout",
        "license": "CC BY-SA 4.0",
    }

    (RAW_DIR / "train_download_provenance.json").write_text(
        json.dumps(source_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTraining questions:", count)
    print("Saved:", OUTPUT)
    print("SHA256:", source_record["output_sha256"])
    print("Next: validate and select training questions; do not train yet.")


if __name__ == "__main__":
    main()