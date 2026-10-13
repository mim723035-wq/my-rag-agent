import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT / "outputs/baselines_20261013T092302Z_9a649338/sft_review"
)
SOURCE = REVIEW_DIR / "batch_001.json"
OUTPUT = REVIEW_DIR / "batch_001.reviewed.json"

DECISIONS = {
    "5a739c75554299623ed4ac0a": (
        "accept_citation_revision",
        "Lake document identifies Rumney; second search supplies "
        "the 2010 census population. Add the lake citation.",
    ),
    "5a740e9855429974ef308ba8": (
        "accept_citation_revision",
        "School document identifies the donor; second search confirms "
        "the gambling-tycoon description and 1997 death. Add school citation.",
    ),
    "5a74ecff5542996c70cfae3d": (
        "needs_search_revision",
        "First document already states the required name, relationship "
        "and alias. Do not imitate the unnecessary second search.",
    ),
    "5a771a0a5542993569682cbd": (
        "accept_citation_revision",
        "Returned documents support the boxer description and "
        "April 2017 coverage arrangement. Add Andy Shepherd citation.",
    ),
    "5a79f9125542996c55b2dcbc": (
        "accept_citation_revision",
        "Returned documents support Hughes's identity and appointment "
        "of Huffcut. Add Huffcut citation.",
    ),
}


def main():
    if OUTPUT.exists():
        raise FileExistsError("核验文件已存在，停止以避免覆盖")

    packets = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    if {p["question_id"] for p in packets} != set(DECISIONS):
        raise ValueError("核验包与预期题目不一致")

    for packet in packets:
        decision, notes = DECISIONS[packet["question_id"]]
        accepted = decision == "accept_citation_revision"

        packet["review"] = {
            "decision": decision,
            "reviewer": "assistant_evidence_review",
            "answer_supported": True,
            "query_progression_valid": accepted,
            "stopping_reasonable": True,
            "corrected_citation_ids": (
                packet["gold_document_ids"] if accepted else []
            ),
            "notes": notes,
            "raw_trace_modified": False,
        }

    OUTPUT.write_text(
        json.dumps(packets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Accepted after citation revision: 4")
    print("Needs search revision: 1")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()