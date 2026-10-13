import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT
    / "outputs"
    / "baselines_20261013T092302Z_9a649338"
    / "sft_review"
)
SOURCE = REVIEW_DIR / "batch_002.json"
DESTINATION = REVIEW_DIR / "batch_002.reviewed.json"

# decision、查询过程是否合理、停止时机是否合理、审核说明
DECISIONS = {
    "5a76b7d15542993735360186": (
        "needs_search_revision",
        False,
        True,
        "首次检索已确定 Whitehaven；第二次仍查询人物，"
        "未围绕缺失的球场关系推进。第三次找到球场后停止合理。"
        "暂不作为优质原始轨迹纳入训练。",
    ),
    "5a78bbac5542990784727747": (
        "needs_search_revision",
        True,
        False,
        "前两次搜索已取得演员身份、参演电影及导演证据；"
        "第三次重复已有证据，应研究提前停止的修订版本。",
    ),
    "5a7ceb7155429909bec76886": (
        "needs_search_revision",
        False,
        False,
        "首次结果明确说明1991年翻拍仍叫 Land of the Lost；"
        "第二次检索多余，暂不纳入原始轨迹训练。",
    ),
    "5a7fc6a65542992e7d278d71": (
        "accept_citation_revision",
        True,
        True,
        "先根据出生日期识别人名，再检索该人物找到节目。"
        "最终引用补充人物身份文档，覆盖完整关系。",
    ),
    "5a85d6ee5542996432c570fd": (
        "accept_citation_revision",
        True,
        True,
        "第一步取得角色与作品，第二步核对作品年份和类型。"
        "第二次查询较宽泛，但取得必要证据；保留这一质量备注。"
        "补充 Next to Normal 文档引用。",
    ),
    "5a7b82d75542995eb53be942": (
        "accept_citation_revision",
        True,
        True,
        "首次检索已取得纪录剧情片与回忆录作者关系、"
        "作者出生年份。补充作品文档引用。",
    ),
    "5a7dcda25542990b8f503abb": (
        "accept_citation_revision",
        True,
        True,
        "首次检索已取得电影改编自小说的关系和小说出版时间。"
        "补充电影文档引用。",
    ),
    "5a7f87ce55429969796c1a98": (
        "accept_citation_revision",
        True,
        True,
        "骑行道文档说明服务 Devonshire Mall；商场文档"
        "说明它是 Windsor 最大商场。补充商场文档引用。",
    ),
    "5a8012c75542992bc0c4a69f": (
        "reject_question_evidence_mismatch",
        None,
        None,
        "文档表明 Bridget Elizabeth Hitler 与 Bridget Dowling"
        "是同一人，不支持标准答案作为其第一位儿媳。"
        "EM命中不代表答案可信；排除，不擅自改写原问题。",
    ),
    "5a84c2855542997b5ce3ff59": (
        "accept_original",
        True,
        True,
        "Chihiro Noda 文档直接说明二人在 Argentina 一起踢球。"
        "现有单篇引用已足够，不强行补充 Paulo Dybala 文档。"
        "标记为直接证据支持的单次搜索样本。",
    ),
}


def main():
    packets = json.loads(SOURCE.read_text(encoding="utf-8-sig"))

    actual_ids = {packet["question_id"] for packet in packets}
    if len(packets) != 10 or actual_ids != set(DECISIONS):
        raise ValueError("批次题目与审核配置不一致，请检查源文件。")

    if DESTINATION.exists():
        raise FileExistsError(f"审核文件已存在，避免覆盖：{DESTINATION}")

    counts = {}

    for packet in packets:
        question_id = packet["question_id"]
        decision, progression, stopping, notes = DECISIONS[question_id]

        seen_ids = {
            document["id"]
            for step in packet["search_steps"]
            for document in step["documents"]
        }

        if decision == "accept_citation_revision":
            corrected_ids = list(packet["gold_document_ids"])
        elif decision == "accept_original":
            corrected_ids = list(packet["original_citation_ids"])
        else:
            corrected_ids = []

        unseen_ids = set(corrected_ids) - seen_ids
        if unseen_ids:
            raise ValueError(
                f"{question_id}: 修订引用包含未检索文档：{unseen_ids}"
            )

        packet["review"] = {
            "decision": decision,
            "answer_supported": (
                decision != "reject_question_evidence_mismatch"
            ),
            "query_progression_valid": progression,
            "stopping_reasonable": stopping,
            "corrected_citation_ids": corrected_ids,
            "notes": notes,
            "reviewer": "assistant_evidence_review",
            "raw_trace_modified": False,
        }
        counts[decision] = counts.get(decision, 0) + 1

    DESTINATION.write_text(
        json.dumps(packets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Saved: {DESTINATION}")
    print("Raw traces unchanged.")


if __name__ == "__main__":
    main()