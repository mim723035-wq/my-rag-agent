"""Compare base, C300, staged D400 and fresh-base joint E400 on matched dev questions."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from hotpot_evaluate_v1 import exact_match_score, f1_score
BASE=ROOT/"outputs/baselines_20261013T042507Z_a43b1522"
OLD=ROOT/"outputs/baselines_20261013T234622Z_601d10b9"

def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(run):
 report=read(run/"evaluation/summary.json")
 summary=next(m for m in report["methods"] if m["method"]=="agent")
 details={r["question_id"]:r for r in read(run/"evaluation/details.json") if r["method"]=="agent"}
 assert report["evaluation_complete"] and summary["missing_or_invalid_count"]==0
 assert len(details)==100==summary["evaluated_count"]
 return summary,details

def main():
 p=argparse.ArgumentParser();p.add_argument("--new-run",type=Path,required=True);args=p.parse_args()
 runs={"base":BASE,"sft_300":OLD,"sft_incremental_400":ROOT/"outputs/baselines_20261014T045233Z_5c5a6ab6","sft_joint_400":args.new_run.resolve()}
 loaded={name:load(run) for name,run in runs.items()}
 ids=set(loaded["base"][1]);assert all(set(details)==ids for _,details in loaded.values())
 configs={name:read(run/"run_config.json") for name,run in runs.items()}
 questions={q["id"]:q for q in [json.loads(s) for s in (BASE/"questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]}
 behavior={name:Counter() for name in runs}
 for filename in ["questions.jsonl","documents.jsonl"]:
  assert len({sha(run/filename) for run in runs.values()})==1,(filename,"mismatch")
 for qid in sorted(ids):
  traces=[read(run/"agent"/(qid+".json")) for run in runs.values()]
  assert all(t["question"]==traces[0]["question"] for t in traces)
  assert all(t["runtime"]==traces[0]["runtime"] for t in traces),(qid,"runtime mismatch")
  assert all(t["max_searches"]==3 for t in traces)
  assert all(t["messages"][:2]==traces[0]["messages"][:2] for t in traces),(qid,"prompt mismatch")
  models=[{k:v for k,v in t["model"].items() if k!="adapter_dir"} for t in traces]
  assert all(m==models[0] for m in models),(qid,"base mismatch")
  assert all(loaded[name][1][qid]["gold_answer"]==loaded["base"][1][qid]["gold_answer"] for name in runs)
  for name,trace in zip(runs,traces):
   row=loaded[name][1][qid];accepted=[questions[qid]["answer"],*questions[qid].get("answer_aliases",[])]
   assert row["answer_exact_match"]==any(exact_match_score(row["predicted_answer"],a) for a in accepted)
   assert abs(row["answer_f1"]-max(f1_score(row["predicted_answer"],a)[0] for a in accepted))<1e-8
   answer=(trace.get("final_answer") or "").casefold()
   if "cannot determine" in answer:behavior[name]["abstention_on_answerable_dev"]+=1
   error_events=[e for e in trace["events"] if "error" in e["observation"]]
   if error_events:behavior[name]["questions_with_action_errors"]+=1
   if error_events and trace["status"]=="finished":behavior[name]["finished_after_error"]+=1
   behavior[name]["unseen_citation_error_events"]+=sum("unseen_ids" in e["observation"] for e in error_events)
 # Verify the actual installed evaluator used to score the historical files.
 # Lexical EM rules stay unchanged; semantic alternatives are reviewed separately.
 pairs={};changes=[];rng=random.Random(42)
 for reference in ["base","sft_300","sft_incremental_400"]:
  counts=Counter();diffs=[]
  for qid in sorted(ids):
   old=loaded[reference][1][qid];new=loaded["sft_joint_400"][1][qid]
   a=bool(old["answer_exact_match"]);b=bool(new["answer_exact_match"])
   category="both_matched" if a and b else "new_only_matched" if b else "reference_only_matched" if a else "both_not_matched"
   counts[category]+=1;diffs.append(int(b)-int(a))
   if a!=b:
    changes.append({"reference":reference,"category":category,"question_id":qid,"question":new["question"],
        "gold":new["gold_answer"],"reference_answer":old["predicted_answer"],"new_answer":new["predicted_answer"],
        "reference_evidence_recall":old["gold_evidence_retrieval_recall"],"new_evidence_recall":new["gold_evidence_retrieval_recall"],
        "new_trace":str(runs["sft_joint_400"]/"agent"/(qid+".json"))})
  distribution=sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(5000))
  pairs[reference]={"counts":dict(counts),"em_difference":sum(diffs)/len(diffs),
      "paired_bootstrap_95_percentile_interval":[distribution[125],distribution[4874]],
      "bootstrap_seed":42,"bootstrap_replicates":5000}
 output={"experiment":"joint400_vs_incremental400_development_comparison","config_check":"PASS",
    "question_count":100,"question_sha256":sha(BASE/"questions.jsonl"),"corpus_sha256":sha(BASE/"documents.jsonl"),
    "runs":{name:str(run) for name,run in runs.items()},
    "results":{name:summary for name,(summary,_) in loaded.items()},"paired_comparisons":pairs,
    "behavior_diagnostics":{name:dict(counts) for name,counts in behavior.items()},
    "limitations":["Development evaluation, not a new blind test.",
      "One training seed; bootstrap covers question sampling, not training-run variability.",
      "C versus E changes data and steps; D versus E uses the same 400 parents but changes training order and optimizer reset.",
      "Same search caps do not equal actual inference cost; inspect actual searches/tokens.",
      "Lexical EM/F1 and gold-document-ID coverage do not prove semantic answer/evidence reliability.",
      "Old baseline runtime timings are historical and may not reflect current system load."]}
 folder=runs["sft_joint_400"]/"evaluation"
 (folder/"comparison_C_D_E.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
 (folder/"paired_answer_changes_C_D_E.json").write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"config_check":"PASS","results":{name:{k:m[k] for k in ["answer_exact_match_rate","mean_answer_f1","complete_gold_citation_rate","finished_rate","mean_search_count","mean_output_tokens","mean_repeated_search_count"]} for name,(m,_) in loaded.items()},"paired_comparisons":pairs},indent=2))

if __name__=="__main__":main()
