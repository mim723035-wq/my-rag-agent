"""Apply recorded post-generation reviews to a teacher batch."""
import argparse
import json
import finalize_teacher_batch_001 as exporter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    exporter.OUT = exporter.ROOT / "data" / f"teacher_batch_{args.batch:03d}_v1"
    from pathlib import Path
    exporter.RUN = Path(json.loads((exporter.OUT / "selection.json").read_text(encoding="utf-8"))["source_run"])
    exporter.REVIEWS = json.loads((exporter.OUT / "review_decisions.json").read_text(encoding="utf-8"))
    exporter.main()

if __name__ == "__main__":
    main()
