"""Post-generation evidence review; gold data read only at this stage."""
import hashlib
import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/baselines_20261013T092302Z_9a649338"
OUT = ROOT / "data/teacher_batch_001_v1"
sys.path.insert(0, str(ROOT / "scripts"))
from hotpot_evaluate_v1 import exact_match_score, f1_score

REVIEWS = {
 "5a712ad95542994082a3e613":("reject_question_premise", "Messapus evidence describes Virgil's Aeneid, not the Iliad. Correct central name does not repair inconsistent question."),
 "5a7151965542994082a3e7c5":("defer_underspecified_question", "Former Lakers legend has multiple plausible referents; matching the benchmark does not establish a unique target."),
 "5a7273815542997f827839dc":("accept", "Golden Slumber and Brilliant Legacy cast documents identify the same actress. Alternative retrieved document supports relation despite not being gold ID."),
 "5a729b4d5542991f9a20c52a":("accept", "First search identifies Milena Govich; second connects her to the 2011 film A Novel Romance."),
 "5a7317ba5542992359bc3232":("accept", "Company document lists Athens retail branch; second search verifies consolidated city-county in Georgia."),
 "5a734e7d5542992359bc32f3":("defer_question_date_ambiguity", "1964-2014 refers to former magazine name; evidence does not establish that magazine ceased publication then. Abstention not exported."),
 "5a73fe2d5542993a88ae2ee0":("defer_insufficient_explicit_evidence", "Searches did not explicitly establish shared country of playing. Nationality is not country of playing. Gold answer French is not a country name; no forced match."),
 "5a7442b955429979e28828c5":("accept", "Appeal and physicist documents support initiation and marriage. Frederic Joliot-Curie is a supported short form of Jean Frederic Joliot-Curie; official EM retained."),
 "5a745f3f55429979e288292a":("accept", "First search identifies Heartland from exact debut date; second retrieves Jessica Steen with matching occupation and role."),
 "5a74752255429974ef308c01":("accept", "Emily Watson document identifies Bess McNeil film; second search identifies Peter Raeburn's soundtrack contribution. Different supporting document from benchmark is semantically sufficient."),
}

def read(path):
 return json.loads(path.read_text(encoding="utf-8-sig"))
def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if (OUT / "train.jsonl").exists() or (OUT / "review_report.json").exists():
  raise FileExistsError("Final output exists")
 questions = {q["id"]:q for q in [json.loads(x) for x in (RUN / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()]}
 selected = read(OUT / "selection.json")["ids"]
 if set(selected) != set(REVIEWS) or len(selected) != len(REVIEWS):
  raise ValueError("Selection mismatch")
 samples, reports = [], []
 for qid in selected:
  path = OUT / (qid + ".json")
  trace = read(path)
  question = questions[qid]
  if trace["status"] != "finished" or question["split"] != "train":
   raise ValueError("Incomplete or non-training trace")
  seen = {d["id"] for e in trace["events"] for d in e["observation"].get("results", [])}
  if any("error" in e["observation"] for e in trace["events"]):
   raise ValueError("Unexpected action error")
  if not set(trace["evidence_ids"]) <= seen:
   raise ValueError("Unseen citation")
  decision, note = REVIEWS[qid]
  reports.append({"question_id":qid,"decision":decision,"reviewer":"assistant_evidence_review; same assistant as generator, not independent human review", "notes":note,"generated_answer":trace["final_answer"],"gold_answer":question["answer"],"official_answer_em":bool(exact_match_score(trace["final_answer"], question["answer"])),"official_answer_f1":f1_score(trace["final_answer"], question["answer"])[0],"complete_gold_document_citation":set(question["supporting_doc_ids"]) <= set(trace["evidence_ids"]),"source_trace_sha256":digest(path)})
  if decision == "accept":
   samples.append({"messages":trace["messages"],"metadata":{"question_id":qid,"sample_id":OUT.name+":"+qid,"model":trace["base_model"],"model_field_meaning":"student tokenizer/config identity, not teacher identity", "teacher":trace["teacher"],"construction":"assistant teacher with actual local controller execution", "source_type":"teacher_generated", "model_generated":True,"student_generated":False,"source_trace":str(path),"source_trace_sha256":digest(path),"documents_sha256":trace["documents_sha256"],"review_basis":"assistant semantic evidence review after generation", "review_notes":note,"search_count":trace["search_count"],"modification":"none; original teacher actions preserved","gold_used_during_generation":False}})
 with (OUT / "train.jsonl").open("x",encoding="utf-8") as file:
  for sample in samples:
   file.write(json.dumps(sample,ensure_ascii=False)+"\n")
 summary = {"generated_count":len(selected),"accepted_count":len(samples),"deferred_or_rejected_count":len(selected)-len(samples),"accepted_single_search":sum(s["metadata"]["search_count"]==1 for s in samples),"accepted_multi_search":sum(s["metadata"]["search_count"]>1 for s in samples),"supervised_actions":sum(m["role"]=="assistant" for s in samples for m in s["messages"]),"train_sha256":digest(OUT / "train.jsonl"),"limitations":["Same assistant generated and reviewed; no independent adjudication.","Selected baseline failures, not representative evaluation.","Teacher exact version and sampling settings unavailable.","No injected controller-error recovery examples in this batch."],"reviews":reports}
 (OUT / "review_report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({k:v for k,v in summary.items() if k!="reviews"},ensure_ascii=False,indent=2))
if __name__ == "__main__":
 main()
