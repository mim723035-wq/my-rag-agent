import json
from pathlib import Path

# 根据脚本位置找到项目根目录，不依赖运行时所在目录。
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# 全部为虚构数据，仅用于学习和流程调试。
DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "林遥的导演作品",
        "text": "林遥是一名导演。她执导的电影《星港》于2021年上映。",
    },
    {
        "id": "doc_002",
        "title": "第十届晨光电影节获奖名单",
        "text": "2022年，第十届晨光电影节将最佳摄影奖授予电影《星港》。",
    },
    {
        "id": "doc_003",
        "title": "同名小说《星港》",
        "text": "小说《星港》由顾川创作，于2018年获得远帆文学奖。它与同名电影无关。",
    },
    {
        "id": "doc_004",
        "title": "周岚的导演作品",
        "text": "周岚执导的电影《雾桥》于2020年上映。",
    },
    {
        "id": "doc_005",
        "title": "第九届晨光电影节获奖名单",
        "text": "2021年，第九届晨光电影节将最佳音乐奖授予电影《雾桥》。",
    },
    {
        "id": "doc_006",
        "title": "林遥的活动记录",
        "text": "2023年，林遥担任晨光电影节评委。担任评委并不表示她的作品获得奖项。",
    },
]

QUESTIONS = [
    {
        "id": "q_001",
        "question": "林遥执导的2021年上映的电影，在2022年晨光电影节获得了什么奖？",
        "answer": "最佳摄影奖",
        "supporting_doc_ids": ["doc_001", "doc_002"],
        "split": "dev",
    },
    {
        "id": "q_002",
        "question": "周岚执导的2020年上映的电影，在2021年晨光电影节获得了什么奖？",
        "answer": "最佳音乐奖",
        "supporting_doc_ids": ["doc_004", "doc_005"],
        "split": "dev",
    },
]


def write_jsonl(path, records):
    """一行保存一个 JSON 对象；已有文件保留，避免覆盖修改。"""
    if path.exists():
        print(f"保留已有文件：{path.name}")
        return

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name} 第 {line_number} 行不是合法 JSON"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name} 第 {line_number} 行必须是 JSON 对象"
                )
            records.append(record)

    return records


def check_records(records, required_fields, label):
    if not records:
        raise ValueError(f"{label}不能为空")

    seen_ids = set()
    for record in records:
        for field in required_fields:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{label}字段 {field} 必须是非空字符串：{record}"
                )

        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"{label}出现重复 ID：{record_id}")
        seen_ids.add(record_id)

    return seen_ids


def validate(documents, questions):
    document_ids = check_records(
        documents, ["id", "title", "text"], "文档"
    )
    check_records(
        questions, ["id", "question", "answer", "split"], "问题"
    )

    seen_questions = set()
    for question in questions:
        # 这里只检查去掉首尾空白后完全相同的问题，不是语义去重。
        question_text = question["question"].strip()
        if question_text in seen_questions:
            raise ValueError(f"重复问题：{question_text}")
        seen_questions.add(question_text)

        if question["split"] not in {"train", "dev", "test"}:
            raise ValueError(f"未知数据划分：{question['split']}")

        evidence_ids = question.get("supporting_doc_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or not all(
                isinstance(item, str) and item.strip()
                for item in evidence_ids
            )
        ):
            raise ValueError(
                f"{question['id']} 的证据必须是非空文档 ID 列表"
            )

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"{question['id']} 存在重复证据 ID")

        missing = set(evidence_ids) - document_ids
        if missing:
            raise ValueError(
                f"{question['id']} 引用了不存在的文档：{sorted(missing)}"
            )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA_DIR / "documents.jsonl", DOCUMENTS)
    write_jsonl(DATA_DIR / "questions.jsonl", QUESTIONS)

    # 重新读取落盘文件，检查真实文件内容。
    documents = read_jsonl(DATA_DIR / "documents.jsonl")
    questions = read_jsonl(DATA_DIR / "questions.jsonl")
    validate(documents, questions)

    print(f"数据检查通过：{len(documents)} 篇文档，{len(questions)} 道问题。")
    print("注意：这里只验证数据结构与引用关系，不验证证据语义。")


if __name__ == "__main__":
    main()