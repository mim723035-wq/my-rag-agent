import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT / "outputs"
    / "baselines_20261013T092302Z_9a649338"
    / "sft_review"
)
SOURCE = REVIEW_DIR / "batch_004.json"
DESTINATION = REVIEW_DIR / "batch_004.reviewed.json"

# decision, answer_supported, progression, stopping, notes
DECISIONS = {
    "5abbe62f55429931dba145d4": (
        "needs_search_revision", True, False, True,
        "Tokyo Disneyland文档支持115英亩，但原引用Disney Music Group"
        "完全不支持答案。首次已出现Tokyo Disneyland，后续查询应围绕该实体。"
        "问题称两处，而Splash Mountain文档列出三处，存在措辞歧义；"
        "须同时复核问题，不能只改引用就接受。",
    ),
    "5ae239d35542992decbdcc86": (
        "needs_search_revision", True, False, False,
        "首次已取得建筑由Marcus T. Reynolds设计及其出生地。"
        "第二次搜索没有补充必要关系，应提前停止并引用完整证据。",
    ),
    "5ae4117355429970de88d8dc": (
        "accept_citation_revision", True, True, True,
        "先确定1887年短篇Kidnapped作者Rudyard Kipling，"
        "再查询出生年份1865。补充短篇文档，避免与同名作品混淆。",
    ),
    "5ae4a8935542995ad6573e03": (
        "accept_citation_revision", True, True, True,
        "先根据电影文档确定作曲家Franz Lehar，再查询出生年份1870。"
        "两部同名电影均指向同一作曲家，答案不受版本影响；"
        "补充标准电影文档引用。",
    ),
    "5ae5da3a554299546bf82fa2": (
        "needs_search_revision", True, False, False,
        "首次人物文档和预算方案文档已分别支持Republican Party。"
        "第二次虽取得新文档，但未补充必要信息，应提前停止。",
    ),
    "5ab2af75554299545a2cfa00": (
        "defer_evidence_ambiguity", None, None, None,
        "公司文档说明2002年成立并制作El Chavo相关动画内容，"
        "但未明确写出El Chavo Animado；另一篇未明确制作公司。"
        "暂不把作品版本推断作为可靠监督目标，等待补充核验。"
        "EM命中和标准ID覆盖不能替代语义支持检查。",
    ),
    "5ab3fffe55429976abd1bd21": (
        "accept_original", True, True, True,
        "问题已明确询问Fark创办人，Fark文档直接写出Drew Curtis。"
        "无需为了标准ID覆盖强行引用Panty Raiders文档。"
        "标记为问题直接给出目标实体的单次搜索样本。",
    ),
    "5ab806605542990e739ec7ea": (
        "accept_original", True, True, True,
        "The Heart, She Holler文档直接覆盖剧情、Hurshe及Kristen Schaal。"
        "问题中的Schaai疑为拼写错误，记录但不修改原问题。"
        "单篇引用足够。",
    ),
    "5ab8495a55429919ba4e229f": (
        "accept_citation_revision", True, True, True,
        "人物文档支持印度裔及无法继续表演，乐队文档支持Monsoon成员关系。"
        "补充乐队文档，覆盖所有识别条件。按历史固定语料审核。",
    ),
    "5aba4f875542994dbf0198d2": (
        "accept_citation_revision", True, True, True,
        "Kenny文档确定父亲为Marv Albert；Marv文档支持名人堂及称号。"
        "首次证据充分，停止合理；补充Kenny文档引用。",
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
        decision, supported, progression, stopping, notes = (
            DECISIONS[packet["question_id"]]
        )

        if decision == "accept_citation_revision":
            corrected_ids = list(packet["gold_document_ids"])
        elif decision == "accept_original":
            corrected_ids = list(packet["original_citation_ids"])
        else:
            corrected_ids = []

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
            "answer_supported": supported,
            "query_progression_valid": progression,
            "stopping_reasonable": stopping,
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