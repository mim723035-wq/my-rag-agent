"""Select fresh, structurally valid training candidates without displaying gold."""
import json
import random
from collections import Counter
from pathlib import Path
from prepare_hotpot_train import validate_record, text_key, title_key, document_id, encode_jsonl, file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/hotpot_teacher_expansion_v1"

def lines(p):return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def main():
 if OUT.exists():raise FileExistsError(OUT)
 eval_dir=ROOT/"data/hotpot_bridge_v1"
 eval_docs={d["id"]:d for d in lines(eval_dir/"documents.jsonl")}
 eval_q=lines(eval_dir/"questions.jsonl")+lines(eval_dir/"test/questions.jsonl")
 eval_titles={title_key(eval_docs[i]["title"]) for q in eval_q for i in q["supporting_doc_ids"]}
 old=lines(ROOT/"data/hotpot_train_v1/questions.jsonl")
 excluded_ids={q["id"] for q in old+eval_q}
 excluded_text={text_key(q["question"]) for q in old+eval_q}
 raw_path=ROOT/"data/raw/hotpotqa/hotpot_train_distractor_hf.json"
 records=json.loads(raw_path.read_text(encoding="utf-8"));counts=Counter(r.get("_id") for r in records if isinstance(r,dict))
 candidates=[];seen=set();exclusions=Counter()
 for record in sorted(records,key=lambda r:r.get("_id","")):
  reason=validate_record(record)
  key=text_key(record.get("question",""))
  if not reason and counts[record["_id"]]!=1:reason="duplicate_id"
  if not reason and record.get("type")!="bridge":reason="not_bridge"
  if not reason and (record["_id"] in excluded_ids or key in excluded_text):reason="existing_question"
  if not reason and {title_key(t) for t,_ in record["supporting_facts"]}&eval_titles:reason="eval_support_title_overlap"
  if not reason and key in seen:reason="duplicate_question_text"
  if reason:exclusions[reason]+=1;continue
  seen.add(key);candidates.append(record)
 rng=random.Random(314159)
 # Difficulty-stratified sampling; proportions are explicit, not benchmark claims.
 groups={level:[r for r in candidates if r.get("level")==level] for level in ["easy","medium","hard"]}
 selected=[]
 for level,size in [("easy",80),("medium",160),("hard",160)]:
  rng.shuffle(groups[level]);assert len(groups[level])>=size
  selected.extend(groups[level][:size])
 rng.shuffle(selected)
 documents={};questions=[]
 for r in selected:
  contexts=dict(r["context"])
  for title,sentences in r["context"]:
   docid=document_id(title,sentences)
   documents[docid]={"id":docid,"title":title,"text":" ".join(s.strip() for s in sentences),"sentences":sentences,"provenance":"HotpotQA distractor train context"}
  facts=[{"document_id":document_id(t,contexts[t]),"sentence_id":i} for t,i in r["supporting_facts"]]
  questions.append({"id":r["_id"],"question":r["question"],"answer":r["answer"],"answer_aliases":[],
      "supporting_doc_ids":list(dict.fromkeys(f["document_id"] for f in facts)),"supporting_facts":facts,
      "split":"train","task_type":"bridge","level":r["level"],"answerable":True,"provenance":"HotpotQA distractor train"})
 OUT.mkdir();(OUT/"documents.jsonl").write_bytes(encode_jsonl([documents[k] for k in sorted(documents)]));(OUT/"questions.jsonl").write_bytes(encode_jsonl(questions))
 manifest={"candidate_questions":len(questions),"documents":len(documents),"levels":dict(Counter(q["level"] for q in questions)),
    "seed":314159,"old_train_question_overlap":0,"eval_question_overlap":0,"eval_support_title_overlap":0,
    "raw_sha256":file_sha256(raw_path),"documents_sha256":file_sha256(OUT/"documents.jsonl"),"questions_sha256":file_sha256(OUT/"questions.jsonl"),
    "exclusions":dict(exclusions),"limitations":["Structural validation is not semantic review.","Gold-title isolation is not complete entity isolation.","New fixed training corpus differs from previous training corpus; evaluation corpus stays frozen."]}
 (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
