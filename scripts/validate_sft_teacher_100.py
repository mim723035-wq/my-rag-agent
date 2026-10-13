"""Replay frozen dataset and audit error masking, stopping and provenance."""
import json
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from kzq_agent import AgentController
from search import SearchTool
from build_sft_teacher_100 import OnceFailingSearch, RUN, OUT

def read(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(path): return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def main():
 samples=jsonl(OUT/"train.jsonl");manifest=read(OUT/"manifest.json")
 assert len(samples)==100==len({s["metadata"]["question_id"] for s in samples})
 assert sha(OUT/"train.jsonl")==manifest["train_sha256"]
 train_questions={q["id"]:q for q in jsonl(RUN/"questions.jsonl")}
 source_tool=SearchTool(RUN/"documents.jsonl")
 tools={str(RUN/"documents.jsonl"):source_tool}
 category_counts=Counter();errors=Counter();unsupervised_errors=0;target_count=0
 natural_reformulation=[];token_examples=[]
 for sample in samples:
  md=sample["metadata"];qid=md["question_id"];category_counts[md["task_category"]]+=1
  assert train_questions[qid]["split"]=="train"
  trace_path=Path(md["dataset_trace"]);assert sha(trace_path)==md["dataset_trace_sha256"]
  trace=read(trace_path);corpus=Path(trace["documents_path"])
  assert sha(corpus)==trace["documents_sha256"]==md["documents_sha256"]
  if str(corpus) not in tools:tools[str(corpus)]=SearchTool(corpus)
  tool=tools[str(corpus)]
  if md["controlled_intervention"]=="transient_tool_failure":tool=OnceFailingSearch(tool)
  ctl=AgentController(trace["question"],tool,max_searches=3)
  messages=sample["messages"];assistant_indices=[i for i,m in enumerate(messages) if m["role"]=="assistant"]
  assert len(assistant_indices)==len(trace["events"])
  supervised=set(md["supervised_message_indices"]);injected=set(md["injected_unsupervised_message_indices"])
  assert supervised<=set(assistant_indices) and not(supervised&injected)
  assert supervised|injected==set(assistant_indices)
  seen=set();successful_queries=set();last_search=0;earliest_complete=None
  final_cites=set(trace["evidence_ids"])
  for n,(idx,event) in enumerate(zip(assistant_indices,trace["events"])):
   action=json.loads(messages[idx]["content"]);assert action==event["action"]
   observation=ctl.step(action);assert observation==event["observation"],(qid,n,"replay mismatch")
   if "error" in observation:
    errors[md["controlled_intervention"]]+=1
    if observation["error"]=="search_failed":
     assert md["controlled_intervention"]=="transient_tool_failure" and idx in supervised
    else:
     assert idx in injected and idx not in supervised
     unsupervised_errors+=1
   else:
    assert idx in supervised
   if action["type"]=="search" and "error" not in observation:
    normalized=" ".join(action["query"].casefold().split())
    assert normalized not in successful_queries,(qid,"redundant successful query")
    successful_queries.add(normalized);last_search=n
    seen.update(d["id"] for d in observation["results"])
    if final_cites and final_cites<=seen and earliest_complete is None:earliest_complete=n
   if n<len(trace["events"])-1:
    expected="Controller result:\n"+json.dumps(observation,ensure_ascii=False)+"\nChoose your next action."
    assert messages[idx+1]=={"role":"user","content":expected}
  assert ctl.state["status"]=="finished" and ctl.state["search_count"]<=3
  assert final_cites<=seen and json.loads(messages[-1]["content"])["type"]=="finish"
  if md["task_category"]!="insufficient_evidence":
   # All final chain documents available before last search would indicate an
   # avoidable search. This catches obvious redundant stopping, not all semantic cases.
   assert earliest_complete==last_search,(qid,"cited chain already available earlier")
   assert final_cites
   note=md["review_notes"].casefold()
   if "recovery" in note or "reformulation" in note:natural_reformulation.append(qid)
  else:
   assert trace["final_answer"]=="Cannot determine" and ctl.state["search_count"]==2
   assert len(tools[str(corpus)].documents)==3
   assert md["missing_relation"] and md["controlled_intervention"]=="three_document_closed_corpus"
  target_count+=len(supervised)
  token_examples.extend((md["sample_id"],i) for i in supervised)
 assert dict(category_counts)==manifest["task_categories"]
 report={"status":"PASS","trajectories":100,"unique_training_questions":100,
    "task_categories":dict(category_counts),"supervised_action_count":target_count,
    "controller_replay":"PASS","source_hashes":"PASS","search_budget":"PASS",
    "unseen_final_citations":0,"redundant_successful_queries":0,
    "obvious_avoidable_final_searches":0,"error_feedback_counts":dict(errors),
    "malformed_action_targets_excluded":unsupervised_errors,
    "natural_retrieval_reformulation_question_ids":natural_reformulation,
    "limitations":["Mechanical replay is not independent semantic review.",
       "Citation-ID availability does not alone prove the cited text supports the conclusion.",
       "Small restricted-corpus abstention examples do not represent open-world unanswerability."]}
 token_path=OUT/"tokenized.jsonl"
 if token_path.exists():
  tokens=jsonl(token_path)
  assert {(t["metadata"]["sample_id"],t["metadata"]["message_index"]) for t in tokens}==set(token_examples)
  assert len(tokens)==target_count
  sample_map={s["metadata"]["sample_id"]:s for s in samples}
  for t in tokens:
   m=t["metadata"];labels=t["labels"]
   assert len(labels)==len(t["input_ids"])==len(t["attention_mask"])
   boundary=next(i for i,label in enumerate(labels) if label!=-100)
   assert boundary>0 and all(v==-100 for v in labels[:boundary])
   assert all(label==tid for label,tid in zip(labels[boundary:],t["input_ids"][boundary:]))
   assert m["message_index"] in sample_map[m["sample_id"]]["metadata"]["supervised_message_indices"]
  report.update({"token_mask":"PASS","tokenized_sha256":sha(token_path),
      "maximum_sequence_tokens":max(len(t["input_ids"]) for t in tokens),
      "total_supervised_tokens":sum(t["metadata"]["supervised_tokens"] for t in tokens)})
 (OUT/"validation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
 sys.stdout.reconfigure(encoding="utf-8")
 main()
