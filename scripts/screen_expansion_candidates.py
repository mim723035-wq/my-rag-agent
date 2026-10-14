"""Post-generation screening and review packets; never feeds gold to actor."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from hotpot_evaluate_v1 import exact_match_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/sft_expansion_review_v1"

def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def jsonl(p):return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args()
 run=a.run_dir.resolve();questions={q["id"]:q for q in jsonl(run/"questions.jsonl")}
 teacher_ids=set(read(ROOT/"data/teacher_batch_007_v1/selection.json")["ids"])
 packets=[];counts=Counter()
 for path in sorted((run/"agent").glob("*.json")):
  trace=read(path);qid=trace["question_id"];q=questions[qid];counts["completed"]+=1
  if qid in teacher_ids:counts["teacher_batch_excluded"]+=1;continue
  if trace["status"]!="finished":counts[trace["status"]]+=1;continue
  answer_matched=bool(exact_match_score(trace["final_answer"],q["answer"]))
  if not answer_matched:counts["answer_not_em"]+=1
  if any("error" in e["observation"] for e in trace["events"]):counts["action_error"]+=1;continue
  seen={};stop=None;queries=set();bad_repeat=False
  for n,event in enumerate(trace["events"]):
   if event["action"]["type"]!="search":continue
   query=" ".join(event["action"]["query"].casefold().split())
   returned=event["observation"]["results"]
   if query in queries or (queries and all(d["id"] in seen for d in returned)):
    bad_repeat=True;break
   queries.add(query);seen.update({d["id"]:d for d in returned})
   if set(q["supporting_doc_ids"])<=seen.keys():stop=n;break
  if stop is None:counts["no_complete_gold_retrieval"]+=1;continue
  if bad_repeat:counts["repeat_before_chain_complete"]+=1;continue
  # Only propose; acceptance requires an explicit assistant review decision.
  facts=[]
  for fact in q["supporting_facts"]:
   doc=seen[fact["document_id"]]
   # Full text is kept so reviewer can notice factual/question mismatch.
   facts.append({"title":doc["title"],"document_id":doc["id"],"sentence_id":fact["sentence_id"]})
  docs=[{"id":i,"title":seen[i]["title"],"text":seen[i]["text"]} for i in q["supporting_doc_ids"]]
  searches=trace["events"][:stop+1]
  packets.append({"question_id":qid,"question":q["question"],"answer":trace["final_answer"] if answer_matched else q["answer"],"original_answer":trace["final_answer"],"gold_answer":q["answer"],
    "level":q["level"],"source_trace":str(path),"actor_model":trace["model"],
    "searches":[{"query":e["action"]["query"],"returned_titles":[d["title"] for d in e["observation"]["results"]]} for e in searches],
    "retained_search_events":stop+1,"original_search_count":trace["search_count"],
    "proposed_citation_ids":q["supporting_doc_ids"],"evidence_documents":docs,"supporting_fact_locations":facts,
    "requires_answer_revision":not answer_matched,"requires_stop_revision":stop+1<trace["search_count"],"requires_citation_revision":set(trace["evidence_ids"])!=set(q["supporting_doc_ids"])})
  counts["review_candidates"]+=1
 OUT.mkdir(exist_ok=True)
 (OUT/"review_packets.json").write_text(json.dumps(packets,ensure_ascii=False,indent=2),encoding="utf-8")
 (OUT/"screening_report.json").write_text(json.dumps({"source_run":str(run),"counts":dict(counts),"candidate_count":len(packets),
    "note":"Programmatic candidates only; not accepted SFT. Original traces unchanged. Complete annotation IDs do not prove semantic support."},indent=2),encoding="utf-8")
 print(json.dumps(dict(counts),indent=2))

if __name__=="__main__":
 sys.stdout.reconfigure(encoding="utf-8");main()
