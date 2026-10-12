import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/raw/hotpotqa/hotpot_dev_distractor_hf.json"
QUESTION_ID = "5ae61bfd5542992663a4f261"


def main():
    records = json.loads(PATH.read_text(encoding="utf-8"))
    record = next(r for r in records if r["_id"] == QUESTION_ID)

    contexts = dict(record["context"])

    print("ID:", record["_id"])
    print("Question:", record["question"])
    print("Answer:", record["answer"])

    for title, sentence_id in record["supporting_facts"]:
        sentences = contexts.get(title)

        print("\nTitle:", repr(title))
        print("Sentence ID:", repr(sentence_id))
        print("ID type:", type(sentence_id).__name__)
        print(
            "Sentence count:",
            len(sentences) if sentences is not None else "MISSING",
        )

        valid = (
            sentences is not None
            and isinstance(sentence_id, int)
            and 0 <= sentence_id < len(sentences)
        )
        print("Valid:", valid)

        if sentences is not None:
            for index, sentence in enumerate(sentences):
                print(f"[{index}] {sentence}")


if __name__ == "__main__":
    main()