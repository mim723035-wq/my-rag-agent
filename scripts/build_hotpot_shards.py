"""Partition eligible training IDs; materialize only one 300-question shard pool."""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from prepare_hotpot_train import (validate_record, text_key, title_key,
    document_id, encode_jsonl, file_sha256, read_jsonl)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/hotpot_shards_v1"
RAW=ROOT/"data/raw/hotpotqa/hotpot_train_distractor_hf.json"
SEED=20261014

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--shards",type=int,default=10)
    args=parser.parse_args()
    if not 2<=args.shards<=100:parser.error("shards must be between 2 and 100")
    if OUT.exists():raise FileExistsError("Partition already exists; refusing overwrite")
    old=[]
    for name in ["hotpot_train_v1","hotpot_teacher_expansion_v1"]:
        old.extend(read_jsonl(ROOT/"data"/name/"questions.jsonl"))
    eval_dir=ROOT/"data/hotpot_bridge_v1"
    evaluation=read_jsonl(eval_dir/"questions.jsonl")+read_jsonl(eval_dir/"test/questions.jsonl")
    eval_docs={d["id"]:d for d in read_jsonl(eval_dir/"documents.jsonl")}
    excluded_ids={q["id"] for q in old+evaluation}
    excluded_texts={text_key(q["question"]) for q in old+evaluation}
    eval_titles={title_key(eval_docs[i]["title"]) for q in evaluation for i in q["supporting_doc_ids"]}
    print("Loading original training pool...",flush=True)
    raw=json.loads(RAW.read_text(encoding="utf-8"))
    id_counts=Counter(r.get("_id") for r in raw if isinstance(r,dict))
    eligible=[];excluded=Counter();seen=set()
    for row,r in sorted(enumerate(raw),key=lambda pair:pair[1].get("_id","") if isinstance(pair[1],dict) else ""):
        reason=validate_record(r) if isinstance(r,dict) else "not_object"
        if not reason and id_counts[r["_id"]]!=1:reason="duplicate_id"
        if not reason and r.get("type")!="bridge":reason="not_bridge"
        if not reason and (r["_id"] in excluded_ids or text_key(r["question"]) in excluded_texts):reason="existing_or_eval_question"
        if not reason and {title_key(t) for t,_ in r["supporting_facts"]}&eval_titles:reason="eval_gold_title_overlap"
        if not reason and text_key(r["question"]) in seen:reason="duplicate_question_text"
        if not reason and r.get("level") not in {"easy","medium","hard"}:reason="unknown_level"
        if reason:excluded[reason]+=1;continue
        seen.add(text_key(r["question"]));eligible.append((row,r))
    rng=random.Random(SEED);groups=defaultdict(list);shards=[[] for _ in range(args.shards)]
    for entry in eligible:groups[entry[1]["level"]].append(entry)
    for level in ["easy","medium","hard"]:
        group=groups[level];rng.shuffle(group)
        for i,entry in enumerate(group):shards[i%args.shards].append(entry)
    # Freeze membership for every shard; sample only first shard, without looking
    # at answers or baseline outcomes. Use explicit 20/40/40 difficulty quotas.
    first=shards[0];selected=[]
    for level,size in [("easy",60),("medium",120),("hard",120)]:
        group=[e for e in first if e[1]["level"]==level]
        rng.shuffle(group)
        if len(group)<size:raise ValueError((level,"insufficient shard candidates"))
        selected.extend(group[:size])
    rng.shuffle(selected)
    pool_id="hotpot_shards_v1:shard_01:sample_300_v1"
    documents={};questions=[]
    for raw_row,r in selected:
        contexts=dict(r["context"])
        for title,sentences in r["context"]:
            docid=document_id(title,sentences)
            documents[docid]={"id":docid,"title":title,"text":" ".join(s.strip() for s in sentences),
                "sentences":sentences,"provenance":"HotpotQA distractor train context"}
        facts=[{"document_id":document_id(t,contexts[t]),"sentence_id":i} for t,i in r["supporting_facts"]]
        questions.append({"id":r["_id"],"question":r["question"],"answer":r["answer"],"answer_aliases":[],
            "supporting_doc_ids":list(dict.fromkeys(f["document_id"] for f in facts)),"supporting_facts":facts,
            "split":"train","task_type":"bridge","level":r["level"],"answerable":True,
            "provenance":"HotpotQA distractor train","retrieval_pool_id":pool_id,"raw_row":raw_row})
    assert len(questions)==len({q["id"] for q in questions})==300
    assert not {q["id"] for q in questions}&excluded_ids
    assert not {title_key(documents[i]["title"]) for q in questions for i in q["supporting_doc_ids"]}&eval_titles
    OUT.mkdir();index_dir=OUT/"partition_indices";index_dir.mkdir()
    counts=[];all_ids=set()
    for i,entries in enumerate(shards,1):
        index=index_dir/f"shard_{i:02d}.jsonl"
        items=[{"id":r["_id"],"raw_row":row,"level":r["level"]} for row,r in sorted(entries,key=lambda e:e[1]["_id"])]
        ids={x["id"] for x in items};assert not all_ids&ids;all_ids.update(ids)
        index.write_bytes(encode_jsonl(items))
        counts.append({"shard":i,"questions":len(items),"levels":dict(Counter(x["level"] for x in items)),"index_sha256":file_sha256(index)})
    pool=OUT/"shard_01/sample_300_v1";pool.mkdir(parents=True)
    (pool/"questions.jsonl").write_bytes(encode_jsonl(questions))
    (pool/"documents.jsonl").write_bytes(encode_jsonl([documents[k] for k in sorted(documents)]))
    binding={"retrieval_pool_id":pool_id,"questions_file":"questions.jsonl","documents_file":"documents.jsonl",
        "questions_sha256":file_sha256(pool/"questions.jsonl"),"documents_sha256":file_sha256(pool/"documents.jsonl"),
        "question_count":300,"document_count":len(documents),"shard":1,"sample_seed":SEED,
        "corpus_rule":"Union of all context paragraphs (gold and distractor) of these 300 questions only; never gold-only.",
        "old_question_overlap":0,"eval_question_overlap":0,"eval_gold_title_overlap":0}
    save(pool/"retrieval_binding.json",binding);save(pool/"manifest.json",binding)
    manifest={"raw_questions":len(raw),"raw_sha256":file_sha256(RAW),"eligible_questions":len(eligible),
        "partition_seed":SEED,"shard_count":args.shards,"partition_rule":"Difficulty-stratified shuffle, then round-robin disjoint question IDs",
        "existing_training_questions_excluded":len(old),"exclusions":dict(excluded),"shards":counts,
        "materialized_shards":[1],"materialized_sample_size":300,"sample_levels":{"easy":60,"medium":120,"hard":120},
        "limitations":["Question-disjoint shards are not entity/document-disjoint shards.",
            "Structural validation is not semantic review or a guarantee of higher-quality SFT.",
            "Changing retrieval corpora changes retrieval difficulty; freeze the corpus per trajectory batch.",
            "Only shard 1 has a sampled question/document library; the other nine currently contain membership indices only."]}
    save(OUT/"manifest.json",manifest)
    print(json.dumps({"eligible":len(eligible),"shard_sizes":[len(s) for s in shards],"sample":binding,"exclusions":dict(excluded)},ensure_ascii=False,indent=2))

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
