import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT / "outputs"
    / "baselines_20261013T092302Z_9a649338"
    / "sft_review"
)
SOURCE = REVIEW_DIR / "batch_003.json"
DESTINATION = REVIEW_DIR / "batch_003.reviewed.json"

# decision, query_progression_valid, stopping_reasonable, notes
DECISIONS = {
    "5a88662255429938390d3f4a": (
        "needs_search_revision", False, False,
        "首次已取得Jody Allen任职关系和Paul Allen创办公司关系；"
        "第二次搜索没有新增文档。应在首次结果后结束并提供完整引用。",
    ),
    "5a887d075542993e715ac03e": (
        "needs_search_revision", False, False,
        "首次Fright Night文档已明确导演、年份及两位制片人；"
        "第二次虽新增文档，但未补充必要关系。暂不接受原始轨迹。"
        "题目关于翻拍年份的措辞不够准确，文档明确指向2011年版本。",
    ),
    "5a8a5f1d55429970aeb702cb": (
        "needs_search_revision", False, False,
        "首次已取得公司生产V2500及该发动机为双轴、用于A320的证据；"
        "第二次只返回已有文档，应提前停止。",
    ),
    "5a8b523055429950cd6afca0": (
        "accept_citation_revision", True, True,
        "首次确定Chris Godwin在Penn State打大学橄榄球；"
        "第二次围绕Penn State查询并取得Big Ten Conference关系。"
        "补充人物文档引用。",
    ),
    "5a8c61fc5542996e8ac88a96": (
        "accept_citation_revision", True, True,
        "首次确定教堂建筑师E. G. Paley；第二次检索该人，"
        "文档明确别名及1823年出生。补充教堂文档引用。",
    ),
    "5a87bf615542996e4f308906": (
        "accept_citation_revision", True, True,
        "一次检索已取得人物背景和Thésée音乐由其创作的证据。"
        "补充歌剧文档，避免只引用人物背景。",
    ),
    "5a88f3c855429938390d3ffc": (
        "accept_citation_revision", True, True,
        "答案由Strand Road Crossing及Ribble Steam Railway文档支持。"
        "原引用Skelmersdale branch不支持答案，应替换，不能保留错误引用。",
    ),
    "5a89a166554299669944a554": (
        "accept_citation_revision", True, True,
        "首次取得球员最后效力球队及球队创办者、创办年份的证据。"
        "补充Connor Hallisey文档。按固定语料中的历史记录审核。",
    ),
    "5a8d79195542994ba4e3dc93": (
        "accept_citation_revision", True, True,
        "Run to the Hills文档确定专辑The Number of the Beast；"
        "专辑文档说明这是最后一张由Clive Burr担任鼓手的专辑。"
        "补充歌曲文档。",
    ),
    "5a8f23ba55429918e830d19d": (
        "accept_citation_revision", True, True,
        "人物文档关联Liu Menggeng与Manchukuo；"
        "Manchukuo文档明确政府于1945年被撤销。补充人物文档。",
    ),
}


def main():
    packets = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    actual_ids = {p["question_id"] for p in packets}

    if len(packets) != 10 or actual_ids != set(DECISIONS):
        raise ValueError("源批次与审核配置不一致。")

    if DESTINATION.exists():
        raise FileExistsError(f"审核文件已存在，避免覆盖：{DESTINATION}")

    counts = Counter()

    for packet in packets:
        decision, progression, stopping, notes = DECISIONS[
            packet["question_id"]
        ]

        corrected_ids = (
            list(packet["gold_document_ids"])
            if decision == "accept_citation_revision"
            else []
        )
        seen_ids = {
            doc["id"]
            for step in packet["search_steps"]
            for doc in step["documents"]
        }

        if not set(corrected_ids).issubset(seen_ids):
            raise ValueError(
                f"{packet['question_id']}: 修订引用包含未检索文档。"
            )

        packet["review"] = {
            "decision": decision,
            "answer_supported": True,
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