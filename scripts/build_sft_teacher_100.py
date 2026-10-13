"""Freeze 100 reviewed trajectories, with explicit controlled interventions.

No new benchmark answers are read by this builder. Natural answers/actions come
from reviewed teacher traces. Controlled abstention cases use deliberately
restricted, inspectable per-question corpora, not failed searches as gold labels.
"""
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kzq_agent import AgentController
from search import SearchTool

RUN = ROOT / "outputs/baselines_20261013T092302Z_9a649338"
OUT = ROOT / "data/sft_teacher_100_v1"

# These are semantic difficulty annotations, not Hotpot's bridge/comparison label.
# Each reason identifies the tempting alternative that must be rejected.
DISTRACTORS = {
 (5,17): "Same-name footballer born1985 versus actor born1964.",
 (5,24): "Cannes2008/2009 versus film's2016 event.",
 (5,28): "Other bands' gold albums versus Kansas and exact target album.",
 (5,29): "60m transportable wheel versus64.75m Prater wheel.",
 (5,37): "Drusilla/Juliet Landau versus Darla/Julie Benz.",
 (5,41): "Other costume designer versus exact1993 Shakespeare adaptation.",
 (5,42): "Prescott's Colombian nationality versus defeated Khan's British nationality.",
 (5,44): "College founding1881 versus university charter1903.",
 (5,47): "Morgan Griffin in San Andreas versus Kalinda actress Archie Panjabi.",
 (5,48): "Mean Girls2004 versus2011 sequel and different directors.",
 (6,0): "Karl versus Klaus Donitz and unrelated officers.",
 (6,2): "Same-title music records versus television cast and eyewear.",
 (6,4): "Bob Quinn and other Ron Howard movies versus verified Far and Away.",
 (6,6): "Captain Nabi versus coach and different-handed bowler.",
 (6,8): "Holywood755acres versus other civil parishes296/204acres.",
 (6,9): "Judwaa2 reboot versus song's original-film context.",
 (3,1): "McCampbell versus first naval medal recipient Edward O'Hare.",
 (3,9): "Dancing with the Stars season12 versus other seasons.",
 (4,35): "Dusk Till Dawn song versus film soundtrack/video game.",
 (4,46): "Scottish Cup finals of other years versus exact2002 scorer.",
}

# Selected terminal relationships: inspectable bridge plus two unrelated records.
# The natural version of each parent is excluded from this training set.
INSUFFICIENT = {
 1: ("Jeff Fatt birth date", "Jeff Fatt born", "The Cockroaches"),
 2: ("Bay Shore county", "Bay Shore county", "Jimmy Jerkens"),
 3: ("Ole Miss university", "Ole Miss Rebels university", "George Kinard"),
 10: ("Vince Neil birth year", "Vince Neil birth year", "Live at the Whisky: One Night Only"),
 11: ("Oklahoma State competition division", "Oklahoma State basketball division", "Jawun Evans"),
 16: ("Alice in Chains recruited lead vocalist", "Alice in Chains recruited singer", "No Excuses"),
 19: ("Louise Josephine father", "Louise Josephine father", "Egelund House"),
 36: ("Auto Shankar death year", "Auto Shankar died", "Pulan Visaranai"),
 38: ("Bernardo Bertolucci birth date", "Bernardo Bertolucci born", "Stealing Beauty"),
 40: ("Nyon lake", "Nyon lake", "Jonas Kocher"),
}
EXCLUDE = {"5a7cd9a255429909bec76849", "5a84c4ee5542994c784dda38", "5a8635d25542994775f6071a"}

def read(path):
 return json.loads(path.read_text(encoding="utf-8-sig"))

def rows(path):
 return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
 path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(path, values):
 with path.open("x", encoding="utf-8") as f:
  for value in values: f.write(json.dumps(value, ensure_ascii=False)+"\n")

def append(messages, action, observation, running):
 index = len(messages)
 messages.append({"role":"assistant", "content":json.dumps(action, ensure_ascii=False)})
 if running:
  messages.append({"role":"user", "content":"Controller result:\n"+json.dumps(observation, ensure_ascii=False)+"\nChoose your next action."})
 return index

class OnceFailingSearch:
 """Controlled transient tool failure; always disclosed in provenance."""
 def __init__(self, tool): self.tool, self.failed = tool, False
 def search(self, query, top_k=3):
  if not self.failed:
   self.failed = True
   raise RuntimeError("temporary retrieval backend unavailable")
  return self.tool.search(query, top_k=top_k)

def main():
 if OUT.exists(): raise FileExistsError("Frozen output already exists; do not overwrite")
 pool, selections = {}, {}
 for batch in range(1,7):
  folder = ROOT / "data" / f"teacher_batch_{batch:03d}_v1"
  selections[batch] = read(folder / "selection.json")["ids"]
  for row in rows(folder / "train.jsonl"):
   qid = row["metadata"]["question_id"]
   if qid in pool: raise ValueError("Duplicate teacher question")
   pool[qid] = row
 insufficient_ids = {selections[5][i] for i in INSUFFICIENT}
 distractor_notes = {selections[b][i]:note for (b,i),note in DISTRACTORS.items()}
 if len(distractor_notes)!=20 or not set(distractor_notes)<=pool.keys():
  raise ValueError("Difficulty annotation points to unaccepted question")
 if insufficient_ids & distractor_notes.keys(): raise ValueError("Stratum overlap")
 candidates = [q for q in pool if q not in insufficient_ids|EXCLUDE and q not in distractor_notes]
 rng = random.Random(42)
 single = sorted(q for q in candidates if pool[q]["metadata"]["search_count"]==1)
 multi = sorted(q for q in candidates if pool[q]["metadata"]["search_count"]>1)
 rng.shuffle(single); rng.shuffle(multi)
 if len(single)<25 or len(multi)<45: raise ValueError(("Insufficient clean strata",len(single),len(multi)))
 categories = {q:"single_search_sufficient" for q in single[:25]}
 categories.update({q:"multi_hop" for q in multi[:45]})
 categories.update({q:"distractor" for q in sorted(distractor_notes)})
 # Choose short, two-search-or-less histories so failures fit the fixed budget.
 recovery_ids = sorted((q for q in categories if pool[q]["metadata"]["search_count"]<=2),
                       key=lambda q:(len(json.dumps(pool[q]["messages"])),q))[:10]
 recovery_kinds = ["unseen_citation"]*4+["unknown_action"]*2+["empty_query"]*2+["transient_tool_failure"]*2
 recovery_map = dict(zip(recovery_ids,recovery_kinds))
 OUT.mkdir()
 (OUT / "traces").mkdir(); (OUT / "controlled_corpora").mkdir()
 source_tool = SearchTool(RUN / "documents.jsonl")
 samples, manifest = [], []
 for qid,category in categories.items():
  source = pool[qid]; original = read(Path(source["metadata"]["source_trace"]))
  controller = AgentController(original["question"],
      OnceFailingSearch(source_tool) if recovery_map.get(qid)=="transient_tool_failure" else source_tool, max_searches=3)
  messages = copy.deepcopy(original["messages"][:2]); supervised=[]; injected=[]
  kind = recovery_map.get(qid)
  for event_number,event in enumerate(original["events"]):
   action = copy.deepcopy(event["action"])
   bad = None
   if kind=="unseen_citation" and action["type"]=="finish":
    bad = copy.deepcopy(action); bad["evidence_ids"]=["unseen_doc_id"]
   elif kind=="unknown_action" and event_number==0:
    bad = {"type":"lookup", "query":action["query"]}
   elif kind=="empty_query" and event_number==0:
    bad = {"type":"search", "query":""}
   elif kind=="transient_tool_failure" and event_number==0:
    bad = copy.deepcopy(action)
   if bad is not None:
    observation=controller.step(bad)
    if "error" not in observation: raise ValueError("Fault did not trigger error")
    index=append(messages,bad,observation,True)
    # Valid request that meets a transient failure may be taught; malformed
    # requests or fabricated citations are input context only, never targets.
    if kind=="transient_tool_failure": supervised.append(index)
    else: injected.append(index)
   observation = controller.step(action)
   if "error" in observation: raise ValueError((qid,observation))
   if action["type"]=="search" and observation["results"]!=event["observation"]["results"]:
    raise ValueError("Natural search results changed")
   supervised.append(append(messages,action,observation,controller.state["status"]=="running"))
  metadata=copy.deepcopy(source["metadata"])
  metadata.update({"sample_id":"sft_teacher_100_v1:"+qid,"task_category":category,
    "difficulty_basis":distractor_notes.get(qid,"Reviewed semantic relation; search count describes retrieval availability, not logical hop count."),
    "behavior_tags":["evidence_based_stop","complete_semantic_citations","no_redundant_successful_query"]+(["failure_recovery"] if kind else []),
    "supervised_message_indices":supervised,"injected_unsupervised_message_indices":injected,
    "controlled_intervention":kind,"source_type":"controlled_recovery" if kind else "teacher_generated",
    "search_count":controller.state["search_count"]})
  if kind:
   metadata["modification"]="Controlled error context executed against actual controller; only valid recovery actions supervised."
   metadata["model_generated"]=False
   metadata["construction"]="Scripted intervention on assistant-reviewed teacher trajectory; actual controller execution."
  trace_path=OUT / "traces" / (qid+".json")
  write_json(trace_path,{"question_id":qid,"question":original["question"],"messages":messages,
      **controller.state,"controlled_intervention":kind,"documents_path":str(RUN/"documents.jsonl"),
      "documents_sha256":sha(RUN/"documents.jsonl"),"source_trace":source["metadata"]["source_trace"]})
  metadata.update({"dataset_trace":str(trace_path),"dataset_trace_sha256":sha(trace_path)})
  samples.append({"messages":messages,"metadata":metadata})
  manifest.append({"question_id":qid,"category":category,"recovery":kind,"search_count":metadata["search_count"],"source_trace":metadata["source_trace"]})
 # Explicitly small closed corpora: inspect ALL three records, not just top-k.
 # These are controlled missing-relation experiments, not natural Hotpot labels.
 distractors=[d for d in source_tool.documents if d["title"] in ["Transformers: The Last Knight – Music from the Motion Picture","Respect to Mehmetçik Monument"]]
 if len(distractors)!=2: raise ValueError("Missing controlled distractor documents")
 for index,(missing_relation,query,bridge_title) in INSUFFICIENT.items():
  qid=selections[5][index]; source=pool[qid]; original=read(Path(source["metadata"]["source_trace"]))
  observed={d["id"]:d for e in original["events"] for d in e["observation"].get("results",[])}
  bridges=[d for d in observed.values() if d["title"]==bridge_title]
  if len(bridges)!=1: raise ValueError("Ambiguous bridge")
  bridge={k:bridges[0][k] for k in ["id","title","text"]}
  corpus=[bridge]+[{k:d[k] for k in ["id","title","text"]} for d in distractors]
  corpus_path=OUT/"controlled_corpora"/(qid+".jsonl"); write_jsonl(corpus_path,corpus)
  tool=SearchTool(corpus_path); controller=AgentController(original["question"],tool,max_searches=3)
  messages=copy.deepcopy(original["messages"][:2]); supervised=[]
  actions=[original["events"][0]["action"],{"type":"search","query":query},
           {"type":"finish","answer":"Cannot determine","evidence_ids":[bridge["id"]]}]
  for action in actions:
   observation=controller.step(action)
   if "error" in observation: raise ValueError(observation)
   supervised.append(append(messages,action,observation,controller.state["status"]=="running"))
  metadata=copy.deepcopy(source["metadata"])
  metadata.update({"sample_id":"sft_teacher_100_v1:"+qid,"task_category":"insufficient_evidence",
     "source_type":"controlled_missing_relation","controlled_intervention":"three_document_closed_corpus",
     "model_generated":False,"construction":"Scripted missing-relation intervention on assistant-reviewed teacher question; actual controller execution.",
     "supervised_message_indices":supervised,"injected_unsupervised_message_indices":[],
     "missing_relation":missing_relation,"corpus_path":str(corpus_path),"documents_sha256":sha(corpus_path),
     "original_corpus_sha256":source["metadata"]["documents_sha256"],"search_count":2,
     "behavior_tags":["abstain_without_guessing","stop_after_targeted_check","partial_evidence_citation"],
     "review_notes":"Entire three-document corpus inspected: bridge relation retained, requested terminal relation absent. Not a claim of absence in the original corpus.",
     "modification":"Re-executed searches against explicitly restricted training corpus; answer changed to corpus-relative abstention."})
  trace_path=OUT/"traces"/(qid+".json")
  write_json(trace_path,{"question_id":qid,"question":original["question"],"messages":messages,**controller.state,
     "controlled_intervention":"three_document_closed_corpus","documents_path":str(corpus_path),
     "documents_sha256":sha(corpus_path),"missing_relation":missing_relation})
  metadata.update({"dataset_trace":str(trace_path),"dataset_trace_sha256":sha(trace_path)})
  samples.append({"messages":messages,"metadata":metadata})
  manifest.append({"question_id":qid,"category":"insufficient_evidence","recovery":None,"search_count":2,"corpus_path":str(corpus_path)})
 if len(samples)!=100 or len({s["metadata"]["question_id"] for s in samples})!=100: raise ValueError("Count mismatch")
 # Freeze sampling policy before training; no selection by new EM scores.
 rng.shuffle(samples)
 write_jsonl(OUT/"train.jsonl",samples)
 summary={"trajectory_count":100,"unique_parent_question_count":100,"teacher_pool_accepted":len(pool),
    "task_categories":dict(Counter(s["metadata"]["task_category"] for s in samples)),
    "controlled_recovery_types":dict(Counter(recovery_map.values())),
    "natural_search_count_distribution":dict(Counter(s["metadata"]["search_count"] for s in samples if s["metadata"]["source_type"]=="teacher_generated")),
    "selection_seed":42,"train_sha256":sha(OUT/"train.jsonl"),"manifest":manifest,
    "limitations":["Assistant generation and semantic review are not independent human adjudication.",
      "Selected prior baseline failures; sampling is behavior-oriented and not representative of HotpotQA.",
      "10 abstention records use three-document restricted corpora, not natural unanswerable Hotpot questions.",
      "10 recovery records have controlled errors, not organically observed teacher failures.",
      "Question category and recovery tags overlap; search count is not semantic hop count.",
      "100 versus old41 changes source/quality/composition, training tokens and steps as well as count.",
      "Prior200 held-out questions were already inspected; final blind claims require fresh isolated questions."]}
 write_json(OUT/"manifest.json",summary)
 print(json.dumps({k:v for k,v in summary.items() if k!="manifest"},ensure_ascii=False,indent=2))

if __name__=="__main__":
 sys.stdout.reconfigure(encoding="utf-8")
 main()
