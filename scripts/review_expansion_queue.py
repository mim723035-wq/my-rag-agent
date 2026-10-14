"""Show completed, unreviewed candidates with actual annotated evidence text."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/sft_expansion_review_v1"
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--batch",type=int,required=True);p.add_argument("--size",type=int,default=20);a=p.parse_args()
 path=OUT/f"review_queue_{a.batch:03d}.json"
 if path.exists():raise FileExistsError(path)
 packets=read(OUT/"review_packets.json");done=read(OUT/"review_decisions.json")
 run=Path(read(OUT/"screening_report.json")["source_run"])
 docs={d["id"]:d for d in [json.loads(s) for s in (run/"documents.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]}
 selected=[r for r in packets if r["question_id"] not in done][:a.size]
 path.write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding="utf-8")
 for i,r in enumerate(selected):
  print(json.dumps({"i":i,"q":r["question"],"answer":r["answer"],"queries":[s["query"] for s in r["searches"]],
    "facts":[f["title"]+": "+docs[f["document_id"]]["sentences"][f["sentence_id"]] for f in r["supporting_fact_locations"]]},ensure_ascii=False))
if __name__=="__main__":
 sys.stdout.reconfigure(encoding="utf-8");main()
