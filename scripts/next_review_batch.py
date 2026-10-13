import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = (
    ROOT / "outputs/baselines_20261013T092302Z_9a649338/sft_review"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10)
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size必须至少为1")

    # 尚未核验的已生成批次先处理，避免重复发放。
    for path in sorted(REVIEW_DIR.glob("batch_[0-9][0-9][0-9].json")):
        reviewed_path = path.with_name(path.stem + ".reviewed.json")
        if not reviewed_path.exists():
            print("Existing pending batch:", path)
            return

    reviewed_ids = set()
    for path in REVIEW_DIR.glob("batch_*.reviewed.json"):
        packets = json.loads(path.read_text(encoding="utf-8-sig"))
        for packet in packets:
            if packet["review"]["decision"] != "pending":
                reviewed_ids.add(packet["question_id"])

    with (REVIEW_DIR / "all_packets.jsonl").open(
        encoding="utf-8-sig"
    ) as file:
        packets = [
            json.loads(line) for line in file if line.strip()
        ]

    remaining = [
        packet for packet in packets
        if packet["question_id"] not in reviewed_ids
    ]
    if not remaining:
        print("No unreviewed candidates remain.")
        return

    multi = [p for p in remaining if p["search_count"] > 1]
    single = [p for p in remaining if p["search_count"] == 1]

    multi_quota = (args.size + 1) // 2
    single_quota = args.size - multi_quota
    selected = multi[:multi_quota] + single[:single_quota]
    selected_ids = {p["question_id"] for p in selected}

    for packet in remaining:
        if len(selected) >= args.size:
            break
        if packet["question_id"] not in selected_ids:
            selected.append(packet)
            selected_ids.add(packet["question_id"])

    existing_numbers = [
        int(path.stem.split("_")[1])
        for path in REVIEW_DIR.glob("batch_[0-9][0-9][0-9].json")
    ]
    number = max(existing_numbers, default=0) + 1
    output = REVIEW_DIR / f"batch_{number:03d}.json"

    with output.open("x", encoding="utf-8") as file:
        json.dump(selected, file, ensure_ascii=False, indent=2)

    print("Already reviewed:", len(reviewed_ids))
    print("Unreviewed before this batch:", len(remaining))
    print("Batch size:", len(selected))
    print("Saved:", output)


if __name__ == "__main__":
    main()