"""Export explicit semantic approvals; replay and preserve original actor traces."""
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from kzq_agent import AgentController
from search import SearchTool

OUT=ROOT/"data/sft_expansion_review_v1"

def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
 report=read(OUT/"screening_report.json");run=Path(report["source_run"])
 packets={r["question_id"]:r for r in read(OUT/"review_packets.json")}
 decisions=read(OUT/"review_decisions.json");tool=SearchTool(run/"documents.jsonl")
 target=OUT/"accepted_traces";target.mkdir(exist_ok=True)
 samples=[];audit=[];config=read(ROOT/"configs/model.json")
 for qid,review in sorted(decisions.items()):
  if review["decision"]!="accept":continue
  packet=packets[qid];path=Path(packet["source_trace"]);source=read(path)
  stop=review.get("retained_search_events_override",packet["retained_search_events"])
  searches=[e for e in source["events"] if e["action"]["type"]=="search"][:stop]
  question=review.get("question_override",packet["question"])
  controller=AgentController(question,tool,max_searches=3)
  messages=copy.deepcopy(source["messages"][:2]);seen={};indices=[]
  if question!=packet["question"]:messages[1]["content"]=messages[1]["content"].replace(packet["question"],question,1)
  for search_index,event in enumerate(searches):
   action=copy.deepcopy(event["action"])
   query_revision=review.get("query_revisions",{}).get(str(search_index))
   if query_revision:action["query"]=query_revision
   observation=controller.step(action)
   assert "error" not in observation
   if not query_revision:assert observation==event["observation"]
   seen.update({d["id"]:d for d in observation["results"]})
   indices.append(len(messages));messages.append({"role":"assistant","content":json.dumps(action,ensure_ascii=False)})
   messages.append({"role":"user","content":"Controller result:\n"+json.dumps(observation,ensure_ascii=False)+"\nChoose your next action."})
  cites=packet["proposed_citation_ids"]
  if "citation_titles_override" in review:
   cites=[]
   for title in review["citation_titles_override"]:
    matches=[i for i,d in seen.items() if d["title"]==title]
    if len(matches)!=1:raise ValueError((qid,title,"ambiguous override"))
    cites.extend(matches)
  answer=review.get("answer_override",packet["answer"])
  action={"type":"finish","answer":answer,"evidence_ids":cites}
  observation=controller.step(action);assert observation.get("accepted") is True
  indices.append(len(messages));messages.append({"role":"assistant","content":json.dumps(action,ensure_ascii=False)})
  revised=source["final_answer"]!=answer or set(source["evidence_ids"])!=set(cites) or source["search_count"]!=stop or bool(review.get("query_revisions")) or question!=packet["question"]
  tracepath=target/(qid+".json")
  trace={"question_id":qid,"messages":messages,**controller.state,"documents_path":str(run/"documents.jsonl"),
    "documents_sha256":sha(run/"documents.jsonl"),"source_trace":str(path),"source_trace_sha256":sha(path),
    "actor_model":source["model"],"revision_by":"current assistant semantic evidence review"}
  write(tracepath,trace)
  metadata={"question_id":qid,"sample_id":"expansion_reviewed:"+qid,"model":config,
    "model_field_meaning":"tokenizer/base config identity; actor_model identifies frozen sampling adapter",
    "actor_model":source["model"],"teacher":"current assistant review/revision; exact revision unavailable",
    "source_type":"student_candidate_teacher_revised" if revised else "student_candidate_reviewed",
    "model_generated":not revised,"student_generated":True,"source_trace":str(path),"source_trace_sha256":sha(path),
    "dataset_trace":str(tracepath),"dataset_trace_sha256":sha(tracepath),"documents_sha256":trace["documents_sha256"],
    "supervised_message_indices":indices,"injected_unsupervised_message_indices":[],"controlled_intervention":None,
    "task_category":review["task_category"],"review_notes":review["notes"],"reviewer":review["reviewer"],
    "search_count":stop,"original_search_count":source["search_count"],"original_answer":source["final_answer"],"original_question":packet["question"],"question_revision":question if question!=packet["question"] else None,
    "gold_used_during_generation":False,"annotation_used_during_review":True,"query_revisions":review.get("query_revisions",{}),
    "modification":"Post-generation semantic answer/citation/stop revision; raw actor trace unchanged" if revised else "none"}
  samples.append({"messages":messages,"metadata":metadata})
  audit.append({"question_id":qid,"revised":revised,"retained_searches":stop,"category":review["task_category"],"notes":review["notes"]})
 # Review work-in-progress may grow; frozen300 export is separate and never overwritten.
 (OUT/"accepted_train.jsonl").write_text("".join(json.dumps(s,ensure_ascii=False)+"\n" for s in samples),encoding="utf-8")
 write(OUT/"export_report.json",{"reviewed":len(decisions),"accepted":len(samples),"rejected":sum(d["decision"]!="accept" for d in decisions.values()),
    "sources":dict(Counter(s["metadata"]["source_type"] for s in samples)),"audit":audit,
    "limitations":["Assistant evidence review is not independent human adjudication.","Post-generation revisions may use annotations; actor generation never receives gold.","Successful/repairable retrieval selection is biased and not representative of all tasks."]})
 print(json.dumps({"reviewed":len(decisions),"accepted":len(samples),"sources":dict(Counter(s["metadata"]["source_type"] for s in samples))},indent=2))

if __name__=="__main__":
 sys.stdout.reconfigure(encoding="utf-8");main()
