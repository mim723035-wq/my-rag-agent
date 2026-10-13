import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT / "outputs"
    / "baselines_20261013T092302Z_9a649338"
    / "sft_review"
)
SOURCE = REVIEW_DIR / "batch_005.json"
DESTINATION = REVIEW_DIR / "batch_005.reviewed.json"

# decision, answer_supported, progression, stopping, notes
DECISIONS = {
    "5ae69b2955429908198fa655": (
        "accept_citation_revision", True, True, True,
        "首次确定Euroinvestor的控股银行为Saxo Bank，"
        "第二次取得1992年成立、2001年获得银行牌照的信息。"
        "答案指公司成立年份，补充Euroinvestor引用。"
        "currently按固定历史语料理解，不宣称当前所有权。",
    ),
    "5ae7d6e85542993210983f7f": (
        "needs_search_revision", True, False, True,
        "首次三篇同名河流文档均指向Romania，"
        "第二次仍查询所在国家，没有新增文档。"
        "应直接查询Romania首都；第三次找到答案后停止合理。"
        "原引用缺少河流到国家的关系。",
    ),
    "5aba5c3a55429939ce03dc9f": (
        "accept_citation_revision", True, True, True,
        "人物文档支持Irish actor and writer及父子关系，"
        "电影文档支持Peter Rabbit真人出演关系。补充电影引用。"
        "电影上映状态按固定历史语料理解。",
    ),
    "5abc073055429931dba14617": (
        "accept_citation_revision", True, True, True,
        "专辑文档支持No Mean City为Nazareth第十张录音室专辑；"
        "乐队文档支持苏格兰及1968年成立。补充乐队引用。",
    ),
    "5ac19e625542991316484b55": (
        "defer_evidence_ambiguity", None, None, None,
        "The Cherry Orchard文档支持契诃夫最后一部剧。"
        "Sakura no Sono文档说明漫画讲述剧团演出该剧，"
        "不支持该剧本身也是漫画这一表述。"
        "需复核并记录问题修订，不能只补标准引用后接受。",
    ),
    "5ac251565542992f1f2b3852": (
        "accept_citation_revision", True, True, True,
        "Carrington House文档确定在那里写作的作品为"
        "Breakfast at Tiffany's，小说文档明确1958年出版。"
        "首次证据充分，补充房屋文档引用。",
    ),
    "5ac4bc94554299076e296deb": (
        "defer_evidence_ambiguity", None, None, None,
        "人物与电影文档共同支持Melissa McCarthy，"
        "但原问题写205年，电影文档写2015年。"
        "疑为年份笔误，先暂缓；题目修订须单独记录，"
        "不在本审核脚本中静默修改。",
    ),
    "5ac4d31c554299076e296e03": (
        "accept_citation_revision", True, True, True,
        "球队文档支持由Sahara India Pariwar拥有，"
        "公司文档支持March 2011的25.94 billion。"
        "补充球队引用。原文未保留币种，仅接受语料中的数值表述，"
        "不自行补美元、卢比或单位换算。",
    ),
    "5add16ad5542990d50227dd1": (
        "reject_question_evidence_mismatch", False, None, None,
        "原问题称专辑2008年发行，而Michael Johns文档明确写2009年。"
        "专辑文档中的2008修饰American Idol contestant，"
        "不能当作专辑发行年份。Australian虽匹配标准答案，"
        "仍不接受含错误年份前提的原始问题。",
    ),
    "5adec5625542995534e8c708": (
        "accept_original", True, True, True,
        "问题已明确作品Live和作曲者Igor Matvienko。"
        "歌曲文档直接写28位艺人参与，并包含Grigory Leps。"
        "其背景是题中已给出的描述，无需为标准ID覆盖额外引用人物文档。"
        "标记为直接证据支持的单次搜索样本。",
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