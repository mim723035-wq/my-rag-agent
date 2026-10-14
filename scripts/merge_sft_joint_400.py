"""Byte-preserving merge of the validated300 and final targeted100.

Token examples are concatenated unchanged, not retokenized or resynthesized.
The trainer independently shuffles all actions using the same fixed seed.
"""
import json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[ROOT/"data/sft_expanded_300_v1",ROOT/"data/sft_targeted_shard_100_v2"]
OUT=ROOT/"data/sft_joint_400_v1"
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
def main():
 if OUT.exists():raise FileExistsError("Joint400 already exists; refusing overwrite")
 samples=[];tokens=[];inputs=[]
 for directory in SOURCES:
  manifest=read(directory/"manifest.json");report=read(directory/"validation_report.json")
  assert report["status"]=="PASS" and report["token_mask"]=="PASS"
  assert sha(directory/"train.jsonl")==manifest["train_sha256"]
  assert sha(directory/"tokenized.jsonl")==report["tokenized_sha256"]
  s=rows(directory/"train.jsonl");t=rows(directory/"tokenized.jsonl")
  assert len(s)==manifest["trajectory_count"] and len(t)==report["supervised_action_count"]
  samples.extend(s);tokens.extend(t)
  inputs.append({"directory":str(directory),"trajectories":len(s),"action_examples":len(t),"train_sha256":sha(directory/"train.jsonl"),"tokenized_sha256":sha(directory/"tokenized.jsonl"),"validation_report_sha256":sha(directory/"validation_report.json")})
 assert len(samples)==len({s["metadata"]["question_id"] for s in samples})==400
 assert len({s["metadata"]["sample_id"] for s in samples})==400
 targets={(s["metadata"]["sample_id"],i) for s in samples for i in s["metadata"]["supervised_message_indices"]}
 assert len(tokens)==len(targets)==1070
 assert {(t["metadata"]["sample_id"],t["metadata"]["message_index"]) for t in tokens}==targets
 for t in tokens:
  ids=t["input_ids"];labels=t["labels"];assert len(ids)==len(labels)==len(t["attention_mask"])<=4096
  boundary=next(i for i,x in enumerate(labels) if x!=-100)
  assert boundary>0 and all(x==-100 for x in labels[:boundary]) and labels[boundary:]==ids[boundary:]
 OUT.mkdir()
 for name in ["train.jsonl","tokenized.jsonl"]:
  parts=[(directory/name).read_bytes() for directory in SOURCES]
  assert all(part.endswith(b"\n") for part in parts)
  (OUT/name).write_bytes(b"".join(parts))
  assert (OUT/name).read_bytes()==b"".join(parts)
 manifest={"trajectory_count":400,"unique_parent_questions":400,"sources":inputs,"source_blocks_byte_preserved":True,
  "train_sha256":sha(OUT/"train.jsonl"),"source_types":dict(Counter(s["metadata"]["source_type"] for s in samples)),
  "training_initialization":"fresh original Qwen3-1.7B with new LoRA; do not load C or D",
  "limitations":["Hybrid source/behavior dataset, not 400 independent teacher generations.","Source corpora remain per-trajectory; this is not a merged retrieval library.","Compare joint shuffling with staged training separately; no pure data-quantity causal claim."]}
 report={"status":"PASS","trajectories":400,"unique_parent_questions":400,"supervised_action_count":len(tokens),
  "total_supervised_tokens":sum(t["metadata"]["supervised_tokens"] for t in tokens),"maximum_sequence_tokens":max(len(t["input_ids"]) for t in tokens),
  "token_mask":"PASS","tokenized_sha256":sha(OUT/"tokenized.jsonl"),"source_hashes":"PASS",
  "token_examples_unchanged":"PASS","source_controller_replays":"PASS in both unchanged source validation reports",
  "malformed_action_targets_excluded":sum(len(s["metadata"]["injected_unsupervised_message_indices"]) for s in samples)}
 save(OUT/"manifest.json",manifest);save(OUT/"validation_report.json",report)
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
