import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic_dev_v1"

# 人物、作品、年份、奖项均为虚构。
CASES = [
    ("single_hop", "沈禾", "云岸", 2019, 2020, "最佳剪辑奖"),
    ("single_hop", "许青", "灯塔", 2020, 2021, "最佳音乐奖"),
    ("single_hop", "顾宁", "远丘", 2021, 2022, "最佳摄影奖"),
    ("two_hop", "陆溪", "夜航", 2019, 2020, "最佳编剧奖"),
    ("two_hop", "温岚", "雪径", 2020, 2021, "最佳美术奖"),
    ("two_hop", "陈岳", "潮声", 2021, 2022, "最佳导演奖"),
    ("distractor_two_hop", "苏棠", "月渡", 2019, 2020, "最佳摄影奖"),
    ("distractor_two_hop", "江澄", "石湾", 2020, 2021, "最佳剪辑奖"),
    ("distractor_two_hop", "叶舟", "秋信", 2021, 2022, "最佳音乐奖"),
    ("insufficient", "唐雁", "风庭", 2019, 2020, None),
    ("insufficient", "林越", "晨岛", 2020, 2021, None),
    ("insufficient", "秦川", "雨城", 2021, 2022, None),
]


def write_jsonl(path, records):
    # 使用x模式，不覆盖已经存在的数据。
    with path.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset():
    documents = []
    questions = []

    for index, case in enumerate(CASES, start=1):
        task_type, person, work, release_year, award_year, award = case
        prefix = f"syn_{index:03d}"
        director_id = f"{prefix}_director"
        award_id = f"{prefix}_award"

        documents.append({
            "id": director_id,
            "title": f"{person}的导演作品",
            "text": (
                f"{person}执导的电影《{work}》于{release_year}年上映。"
            ),
            "provenance": "synthetic",
        })

        if award is not None:
            documents.append({
                "id": award_id,
                "title": f"{award_year}年青穗电影节：《{work}》",
                "text": (
                    f"电影《{work}》在{award_year}年青穗电影节"
                    f"获得{award}。"
                ),
                "provenance": "synthetic",
            })

        if task_type == "single_hop":
            question_text = (
                f"电影《{work}》在{award_year}年青穗电影节"
                "获得了什么奖？"
            )
            gold_ids = [award_id]
        else:
            question_text = (
                f"{person}执导的{release_year}年上映的电影，"
                f"在{award_year}年青穗电影节获得了什么奖？"
            )
            gold_ids = (
                [director_id, award_id] if award is not None else []
            )

        questions.append({
            "id": f"{prefix}_q",
            "question": question_text,
            "answer": award if award is not None else "无法确定",
            "answer_aliases": [],
            "supporting_doc_ids": gold_ids,
            "split": "dev",
            "task_type": task_type,
            "answerable": award is not None,
            "provenance": "synthetic",
        })

        if task_type == "distractor_two_hop":
            # 同名小说、同一电影其他年份的奖项。
            documents.extend([
                {
                    "id": f"{prefix}_novel",
                    "title": f"同名小说《{work}》",
                    "text": (
                        f"小说《{work}》在{award_year}年获得远帆文学奖。"
                        "它与同名电影属于不同作品。"
                    ),
                    "provenance": "synthetic",
                },
                {
                    "id": f"{prefix}_other_year",
                    "title": f"{award_year + 1}年青穗电影节：《{work}》",
                    "text": (
                        f"电影《{work}》在{award_year + 1}年"
                        "青穗电影节获得观众选择奖。"
                    ),
                    "provenance": "synthetic",
                },
            ])

        if task_type == "insufficient":
            # 提供相关信息，但不提供问题要求的获奖事实。
            documents.append({
                "id": f"{prefix}_screening",
                "title": f"《{work}》的电影节活动",
                "text": (
                    f"电影《{work}》在{award_year}年青穗电影节展映。"
                    "本记录未提供该片的获奖信息。"
                ),
                "provenance": "synthetic",
            })

    return documents, questions


def validate(documents, questions):
    document_ids = [doc["id"] for doc in documents]
    question_ids = [question["id"] for question in questions]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("文档ID重复")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("问题ID重复")

    document_id_set = set(document_ids)
    seen_questions = set()

    for doc in documents:
        if not doc["title"].strip() or not doc["text"].strip():
            raise ValueError(f"空文档：{doc['id']}")

    for question in questions:
        text = question["question"].strip()
        if not text or text in seen_questions:
            raise ValueError("问题为空或重复")
        seen_questions.add(text)

        evidence = question["supporting_doc_ids"]
        if not set(evidence) <= document_id_set:
            raise ValueError(f"证据文档不存在：{question['id']}")

        if question["answerable"] and not evidence:
            raise ValueError(f"可回答题缺少标准证据：{question['id']}")

        if not question["answerable"]:
            if evidence or question["answer"] != "无法确定":
                raise ValueError(f"证据不足题标注错误：{question['id']}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document_path = OUTPUT_DIR / "documents.jsonl"
    question_path = OUTPUT_DIR / "questions.jsonl"

    if document_path.exists() or question_path.exists():
        raise FileExistsError(
            "目标数据文件已经存在，停止生成，避免覆盖。"
        )

    documents, questions = build_dataset()
    validate(documents, questions)

    write_jsonl(document_path, documents)
    write_jsonl(question_path, questions)

    print(f"生成并检查通过：{len(documents)}篇文档，{len(questions)}道问题。")
    print(f"保存目录：{OUTPUT_DIR}")
    print("注意：结构检查不替代人工核验，也不证明真实任务效果。")


if __name__ == "__main__":
    main()