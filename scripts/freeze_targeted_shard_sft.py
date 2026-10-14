"""Freeze 100 reviewed assistant trajectories and disclosed interventions.

Only local training shard 1 is used. No model training or dev-score changes.
"""
import os
os.environ.setdefault("HF_HOME",r"D:\search-agent-cache\huggingface")
import argparse,copy,json,sys
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from search import SearchTool
from kzq_agent import AgentController
from retrieval_binding import verify_retrieval_binding,sha
from shard_teacher_session import POOL,OUT,read,rows,save
SOURCE_DIR=OUT
from build_sft_expanded_300 import JSON_ERROR

NEGATIVE={274:("Wilhelm Molterer","Steyr Austrian federal state"),
9:("Grand Prix (TV programme)","Murray Walker birth year"),
10:("Agot Isidro","Ateneo de Manila University founding year"),
46:("David Wallis Reeves","Charles Ives birth year"),
47:("Number 23 Basketball Player","Niki de Saint Phalle birth name")}
DISTRACTORS=["The Second World War (disambiguation)","Esteban Cambiasso",
"Soap Opera Digest Award for Hottest Male Star","Emmanuel Clottey",
"The Complaint of an Empress","Roland Riese","Heartless (Once Upon a Time)",
"2010–11 Kilmarnock F.C. season","Lindsey Shockley","Stephen Merchant"]

def append(messages,action,obs,raw=None,running=True):
 i=len(messages);messages.append({"role":"assistant","content":raw if raw is not None else json.dumps(action,ensure_ascii=False)})
 if running:messages.append({"role":"user","content":"Controller result:\n"+json.dumps(obs,ensure_ascii=False)+"\nChoose your next action."})
 return i

def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=ROOT/"data/sft_targeted_shard_100_v2");args=parser.parse_args();OUT=args.output_dir.resolve()
 if (OUT/"train.jsonl").exists():raise FileExistsError("Frozen train exists; refusing overwrite")
 binding=verify_retrieval_binding(POOL);questions=rows(POOL/"questions.jsonl")
 plan=read(OUT/"selection_plan.json");groups=plan["primary_groups"]
 assert len(plan["selected_pool_indices"])==len(set(plan["selected_pool_indices"]))==100
 assert not set(plan["selected_pool_indices"])&{0,1,2},"Annotation-exposed draft parents must be excluded"
 tool=SearchTool(POOL/"documents.jsonl");docmap={d["title"]:d for d in tool.documents}
 config=read(ROOT/"configs/model.json");tokenizer=AutoTokenizer.from_pretrained(config["model_id"],revision=config["revision"],use_fast=True)
 for folder in ["traces","controlled_corpora"]:(OUT/folder).mkdir(exist_ok=True)
 samples=[];audits=[]
 for group,indices in groups.items():
  for index in indices:
   q=questions[index];qid=q["id"];source_path=SOURCE_DIR/"teacher_raw"/(qid+".json");source=read(source_path)
   assert source["status"]=="finished" and source["gold_used_during_generation"] is False
   assert source["documents_sha256"]==binding["documents_sha256"] and q["split"]=="train"
   final=copy.deepcopy(source["events"][-1]["action"]);cites=set(final["evidence_ids"])
   searches=[e for e in source["events"] if e["action"]["type"]=="search"]
   # Explicitly remove a search that only reconfirms a known producer/biography.
   # The following query uses names already present in the retained first result.
   remove={43:{1},75:{1},239:set()}.get(index,set())
   kept=[];seen=set()
   for n,e in enumerate(searches):
    if n in remove:continue
    kept.append(copy.deepcopy(e["action"]));seen.update(d["id"] for d in e["observation"].get("results",[]))
    if cites<=seen:break
   assert cites<=seen
   controlled=None;corpus_path=POOL/"documents.jsonl";local_tool=tool
   missing=None
   if group=="insufficient_evidence":
    title,missing=NEGATIVE[index];bridge=docmap[title]
    n=list(NEGATIVE).index(index)
    corpus=[bridge,docmap[DISTRACTORS[2*n]],docmap[DISTRACTORS[2*n+1]]]
    corpus_path=OUT/"controlled_corpora"/(qid+".jsonl")
    corpus_path.write_text("".join(json.dumps(d,ensure_ascii=False)+"\n" for d in corpus),encoding="utf-8")
    local_tool=SearchTool(corpus_path);controlled="three_document_closed_corpus"
    kept=[searches[0]["action"],searches[1]["action"]]
    final={"type":"finish","answer":"Cannot determine","evidence_ids":[bridge["id"]]}
   elif group=="citation_recovery":
    controlled="near_miss_citation" if indices.index(index)<12 else "multiple_json_objects"
   ctl=AgentController(q["question"],local_tool,max_searches=3)
   messages=copy.deepcopy(source["messages"][:2]);supervised=[];injected=[]
   for action in kept:
    obs=ctl.step(action);assert "error" not in obs
    supervised.append(append(messages,action,obs))
   if controlled=="near_miss_citation":
    bad=copy.deepcopy(final);value=bad["evidence_ids"][0]
    bad["evidence_ids"][0]=value[:-1]+("0" if value[-1]!="0" else "1")
    assert bad["evidence_ids"][0] not in ctl.seen_document_ids
    obs=ctl.step(bad);assert "unseen_ids" in obs
    injected.append(append(messages,bad,obs))
   elif controlled=="multiple_json_objects":
    raw=json.dumps(final,ensure_ascii=False)+"\n"+json.dumps(final,ensure_ascii=False)
    try:json.loads(raw)
    except json.JSONDecodeError:pass
    else:raise ValueError("Expected malformed JSON")
    action={"raw":raw};obs=ctl.record(action,{"error":JSON_ERROR})
    injected.append(append(messages,action,obs,raw=raw))
   obs=ctl.step(final);assert obs.get("accepted") is True
   supervised.append(append(messages,final,obs,running=False))
   modified=bool(controlled) or len(kept)!=len(searches)
   trace={"question_id":qid,"messages":messages,**ctl.state,"documents_path":str(corpus_path),"documents_sha256":sha(corpus_path),"controlled_intervention":controlled}
   trace_path=OUT/"traces"/(qid+".json");save(trace_path,trace)
   # The 15 continue and 10 sufficient groups together form the stop category.
   category="stop_decision" if group in {"continue_missing_relation","stop_when_sufficient"} else group
   evidence={d["id"]:d for e in trace["events"] for d in e["observation"].get("results",[])}
   md={"question_id":qid,"sample_id":OUT.name+":"+qid,"model":config,
    "model_field_meaning":"base/tokenizer identity; teacher is current assistant, not this Qwen model",
    "teacher":source["teacher"],"source_type":"controlled_missing_relation" if missing else "controlled_recovery" if controlled else "teacher_stop_revised" if modified else "teacher_generated",
    "model_generated":not modified,"student_generated":False,"construction":"Assistant teacher with actual local tool execution; interventions disclosed",
    "source_trace":str(source_path),"source_trace_sha256":sha(source_path),"dataset_trace":str(trace_path),"dataset_trace_sha256":sha(trace_path),
    "documents_sha256":sha(corpus_path),"original_documents_sha256":binding["documents_sha256"],"retrieval_pool_id":binding["retrieval_pool_id"],
    "supervised_message_indices":supervised,"injected_unsupervised_message_indices":injected,"controlled_intervention":controlled,
    "task_category":category,"target_behavior":group,"search_count":ctl.state["search_count"],"original_search_count":source["search_count"],
    "missing_relation":missing,"level":q["level"],"gold_used_during_generation":False,"reviewer":"current assistant semantic evidence review; not independent human adjudication",
    "review_notes":"Cited full paragraphs inspected for requested entity, relation, answer type and qualifiers. Primary behavior groups overlap semantically; repeated/nonessential searches removed where necessary.",
    "modification":"Controlled corpus or error context; raw generation unchanged" if controlled else "Remove redundant/nonessential search, replay retained actions and preserve supported answer/citations" if modified else "none"}
   sample={"messages":messages,"metadata":md}
   for i in supervised:
    prompt=tokenizer.apply_chat_template(messages[:i],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    length=len(tokenizer(prompt+messages[i]["content"]+tokenizer.eos_token,add_special_tokens=False)["input_ids"])
    if length>4096:raise ValueError((index,"overlength",length))
   samples.append(sample)
   audits.append({"question_id":qid,"pool_index":index,"decision":"accept","target_behavior":group,"original_answer":source["final_answer"],"dataset_answer":final["answer"],
    "reviewed_documents":[{"title":evidence[i]["title"],"text":evidence[i]["text"]} for i in final["evidence_ids"]],
    "missing_relation":missing,"corpus_review":"Entire three-document corpus inspected; target relationship absent" if missing else None,
    "modification":md["modification"],"retained_searches":len(kept),"source_searches":len(searches)})
 assert len(samples)==len({s["metadata"]["question_id"] for s in samples})==100
 old=rows(ROOT/"data/sft_expanded_300_v1/train.jsonl")
 assert not {s["metadata"]["question_id"] for s in samples}&{s["metadata"]["question_id"] for s in old}
 with (OUT/"train.jsonl").open("x",encoding="utf-8") as f:
  for sample in samples:f.write(json.dumps(sample,ensure_ascii=False)+"\n")
 save(OUT/"review_decisions.json",audits)
 all_raw=[read(p) for p in (SOURCE_DIR/"teacher_raw").glob("*.json")]
 save(OUT/"candidate_inventory.json",[{"pool_index":t["pool_index"],"question_id":t["question_id"],"status":t["status"],"exported":t["question_id"] in {s["metadata"]["question_id"] for s in samples}} for t in all_raw])
 manifest={"trajectory_count":100,"unique_parent_questions":100,"pool_id":binding["retrieval_pool_id"],"source_pool_sha256":binding["documents_sha256"],
    "train_sha256":sha(OUT/"train.jsonl"),"primary_categories":dict(Counter(s["metadata"]["task_category"] for s in samples)),
    "target_behaviors":dict(Counter(s["metadata"]["target_behavior"] for s in samples)),"source_types":dict(Counter(s["metadata"]["source_type"] for s in samples)),
    "controlled_interventions":dict(Counter(s["metadata"]["controlled_intervention"] for s in samples if s["metadata"]["controlled_intervention"])),
    "candidate_questions_attempted":len(all_raw),"raw_finished":sum(t["status"]=="finished" for t in all_raw),"unexported_candidates":len(all_raw)-100,
    "old_300_parent_overlap":0,"training_started":False,"annotation_exposed_draft_parents_excluded":[0,1,2],
    "limitations":["Same assistant generation and semantic review; not independent human adjudication.",
      "Behavior-targeted selection changes source/composition; not representative Hotpot sampling.",
      "Query progression includes retrieval reformulation; number of searches is not logical hops.",
      "15 recovery and 5 missing-relation demonstrations are controlled interventions, not naturally occurring failures.",
      "Restricted three-document abstention does not prove original-corpus unanswerability.",
      "No training, capability gain or new blind evaluation established."]}
 save(OUT/"manifest.json",manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
