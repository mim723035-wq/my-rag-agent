"""Replay the final targeted100 and check masking and fixed shard binding."""
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from search import SearchTool
from kzq_agent import AgentController
from retrieval_binding import verify_retrieval_binding,sha
from shard_teacher_session import POOL,read,rows,save
from build_sft_expanded_300 import JSON_ERROR
OUT=ROOT/"data/sft_targeted_shard_100_v2"

def main():
 binding=verify_retrieval_binding(POOL);manifest=read(OUT/"manifest.json");samples=rows(OUT/"train.jsonl")
 assert len(samples)==len({s["metadata"]["question_id"] for s in samples})==100
 assert sha(OUT/"train.jsonl")==manifest["train_sha256"]
 questions={q["id"]:q for q in rows(POOL/"questions.jsonl")}
 plan=read(OUT/"selection_plan.json");assert not set(plan["selected_pool_indices"])&{0,1,2}
 old=rows(ROOT/"data/sft_expanded_300_v1/train.jsonl")
 assert not {s["metadata"]["question_id"] for s in samples}&{s["metadata"]["question_id"] for s in old}
 tools={};errors=Counter();target_keys=set();searches=Counter();lengths=[]
 for s in samples:
  md=s["metadata"];qid=md["question_id"];assert questions[qid]["split"]=="train"
  assert sha(md["source_trace"])==md["source_trace_sha256"]
  assert sha(md["dataset_trace"])==md["dataset_trace_sha256"]
  t=read(Path(md["dataset_trace"]));assert t["messages"]==s["messages"]
  corpus=Path(t["documents_path"]);digest=sha(corpus);assert digest==t["documents_sha256"]==md["documents_sha256"]
  if not md["controlled_intervention"]=="three_document_closed_corpus":assert digest==binding["documents_sha256"]
  if digest not in tools:tools[digest]=SearchTool(corpus)
  ctl=AgentController(t["question"],tools[digest],max_searches=3)
  ids=[i for i,m in enumerate(s["messages"]) if m["role"]=="assistant"]
  sup=set(md["supervised_message_indices"]);inj=set(md["injected_unsupervised_message_indices"])
  assert sup|inj==set(ids) and not sup&inj and len(ids)==len(t["events"])
  seen=set();queries=set();complete_at=None;last_search=None;cites=set(t["evidence_ids"])
  for i,e in zip(ids,t["events"]):
   raw=s["messages"][i]["content"]
   try:action=json.loads(raw)
   except json.JSONDecodeError:
    assert i in inj and md["controlled_intervention"]=="multiple_json_objects"
    action={"raw":raw};obs=ctl.record(action,{"error":JSON_ERROR})
   else:obs=ctl.step(action)
   assert action==e["action"] and obs==e["observation"]
   if "error" in obs:
    assert i in inj;errors[md["controlled_intervention"]]+=1
   else:assert i in sup
   if i in sup:
    assert action["type"] in {"search","finish"};target_keys.add((md["sample_id"],i))
   if action.get("type")=="search" and "error" not in obs:
    query=" ".join(action["query"].casefold().split());assert query not in queries;queries.add(query)
    seen.update(d["id"] for d in obs["results"]);last_search=i
    if cites<=seen and complete_at is None:complete_at=i
   if i!=ids[-1]:assert s["messages"][i+1]=={"role":"user","content":"Controller result:\n"+json.dumps(obs,ensure_ascii=False)+"\nChoose your next action."}
  assert ctl.state["status"]=="finished" and ctl.state["search_count"]<=3 and cites<=seen
  if md["missing_relation"]:
   assert len(tools[digest].documents)==3 and t["final_answer"]=="Cannot determine" and ctl.state["search_count"]==2
  else:assert complete_at==last_search
  searches[ctl.state["search_count"]]+=1
 tokens=rows(OUT/"tokenized.jsonl")
 assert {(r["metadata"]["sample_id"],r["metadata"]["message_index"]) for r in tokens}==target_keys
 assert len(tokens)==len(target_keys)
 for r in tokens:
  ids=r["input_ids"];labels=r["labels"];assert len(ids)==len(labels)==len(r["attention_mask"])<=4096
  boundary=next(i for i,x in enumerate(labels) if x!=-100)
  assert boundary>0 and all(x==-100 for x in labels[:boundary]) and labels[boundary:]==ids[boundary:]
 report={"status":"PASS","trajectories":100,"unique_parent_questions":100,"old_300_parent_overlap":0,"merged_unique_parent_count":400,
  "controller_replay":"PASS","source_hashes":"PASS","retrieval_pool_binding":"PASS","search_budget":"PASS",
  "unseen_final_citations":0,"repeated_successful_queries":0,"obvious_avoidable_final_searches":0,
  "malformed_action_targets_excluded":sum(errors.values()),"error_feedback_counts":dict(errors),
  "supervised_action_count":len(tokens),"total_supervised_tokens":sum(r["metadata"]["supervised_tokens"] for r in tokens),
  "maximum_sequence_tokens":max(len(r["input_ids"]) for r in tokens),"token_mask":"PASS","tokenized_sha256":sha(OUT/"tokenized.jsonl"),
  "search_count_distribution":dict(searches),"annotation_exposed_draft_parent_indices_excluded":[0,1,2],
  "limitations":["Mechanical checks do not replace independent semantic adjudication.","Behavior groups overlap; search count is not logical hop count.","Controlled small-corpus abstention is not natural Hotpot unanswerability."]}
 save(OUT/"validation_report.json",report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
