import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT / "outputs"
    / "baselines_20261013T092302Z_9a649338"
    / "sft_review"
)
SOURCE = REVIEW_DIR / "batch_006.json"
DESTINATION = REVIEW_DIR / "batch_006.reviewed.json"

# decision, notes
DECISIONS = {
    "5adec9aa5542992fa25da84f": (
        "accept_citation_revision",
        "人物文档支持British及1917年获奖；Characteristic X-ray"
        "文档明确发现者及因该发现获奖。补充发现与获奖关系引用。",
    ),
    "5aded2cd5542993344016ba5": (
        "accept_citation_revision",
        "Kristina文档明确其被比较为Anna Chapman；Anna文档明确"
        "被捕涉及Illegals Program spy ring。补充比较关系引用。"
        "保留原答案，不为了冠词差异改写。",
    ),
    "5ae1050855429920d5234283": (
        "accept_original",
        "Drew Crompton文档直接说明其任职，以及参与Lynn Swann"
        "2006年宾夕法尼亚州长竞选。单篇引用已覆盖问题，"
        "不强行增加Lynn Swann文档。",
    ),
    "5ae1e92a5542997283cd22cf": (
        "accept_citation_revision",
        "运动员文档确定项目为long jump；项目文档明确女子项目"
        "自1948年进入奥运会。补充运动员引用。",
    ),
    "5ae234aa554299495565da3c": (
        "accept_original",
        "Now Is Good文档直接覆盖2012年、英国青少年剧情片、"
        "Ol Parker导演及Jeremy Irvine出演。原引用充分。",
    ),
    "5ae23f755542994d89d5b3a3": (
        "accept_citation_revision",
        "Naresh Goyal文档确定其为Jet Airways创始董事长；"
        "航空公司文档明确所在地Mumbai。补充人物引用。"
        "按固定历史语料审核，不宣称当前公司状态。",
    ),
    "5ae27d175542996483e649bb": (
        "accept_citation_revision",
        "Hoek文档支持其与Fabianski合作，但不支持Swansea City。"
        "必须加入Fabianski文档支持球队答案；保留Hoek文档"
        "用于题目背景关系。球队归属按固定历史语料审核。",
    ),
    "5ae372ce5542992e3233c406": (
        "accept_citation_revision",
        "小说文档确定作者Jean Rabe；人物文档覆盖职业、"
        "相关系列及1957年6月19日出生。补充小说引用。",
    ),
    "5ae3e5905542996836b02be9": (
        "accept_citation_revision",
        "De Borgia文档支持所属Mineral County；县文档支持"
        "2010年人口4223。删除无关Orchard Homes引用，"
        "加入Mineral County文档。4223指县人口，不是De Borgia人口。",
    ),
    "5ae451bf5542995dadf2431e": (
        "accept_original",
        "Catherine Breillat文档同时说明A Real Young Girl是其"
        "首部电影，以及其在European Graduate School任教。"
        "单篇引用充分；任职描述按固定历史语料审核。",
    ),
}


def main():
    packets = json.loads(SOURCE.read_text(encoding="utf-8-sig"))

    if (
        len(packets) != 10
        or {p["question_id"] for p in packets} != set(DECISIONS)
    ):
        raise ValueError("源批次与审核配置不一致。")

    if DESTINATION.exists():
        raise FileExistsError(f"审核文件已存在：{DESTINATION}")

    counts = Counter()

    for packet in packets:
        decision, notes = DECISIONS[packet["question_id"]]

        if decision == "accept_citation_revision":
            corrected_ids = list(packet["gold_document_ids"])
        else:
            corrected_ids = list(packet["original_citation_ids"])

        seen_ids = {
            doc["id"]
            for step in packet["search_steps"]
            for doc in step["documents"]
        }
        if not set(corrected_ids).issubset(seen_ids):
            raise ValueError(
                f"{packet['question_id']}: 引用包含未检索文档。"
            )

        packet["review"] = {
            "decision": decision,
            "answer_supported": True,
            "query_progression_valid": True,
            "stopping_reasonable": True,
            "corrected_citation_ids": corrected_ids,
            "notes": notes,
            "reviewer": "assistant_evidence_review",
            "raw_trace_modified": False,
        }
        counts[decision] += 1

    DESTINATION.write_text(
        json.dumps(packets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    print(f"Saved: {DESTINATION}")
    print("Raw traces unchanged.")


if __name__ == "__main__":
    main()