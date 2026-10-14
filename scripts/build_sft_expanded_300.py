"""Freeze 300 distinct parents; original 100 bytes are an unchanged prefix.

New examples are a disclosed mixture of assistant teacher demonstrations,
reviewed student candidates, and controlled interventions. No training occurs.
"""
import os
os.environ.setdefault("HF_HOME", r"D:\search-agent-cache\huggingface")
import copy
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kzq_agent import AgentController
from search import SearchTool
from build_sft_teacher_100 import OnceFailingSearch, append, read, rows, sha, write_json

OLD = ROOT / "data/sft_teacher_100_v1"
REVIEW = ROOT / "data/sft_expansion_review_v1"
TEACHER = ROOT / "data/teacher_batch_007_v1"
POOL = ROOT / "data/hotpot_teacher_expansion_v1"
OUT = ROOT / "data/sft_expanded_300_v1"

# The entire retained paragraph was manually inspected, not just its annotated
# sentence. In each case the intermediate entity exists but this relation does not.
MISSING = {
    "5a7b6fd15542997c3ec97179": ("Amigos para La Aventura", "Palito Ortega birth date"),
    "5a7b781e5542997c3ec971a7": ("C-52 (cipher machine)", "Crypto AG headquarters city"),
    "5a7c15195542997c3ec972df": ("Peace Conference of 1861", "Willard Hotel street address"),
    "5a7cab2b554299683c1c6331": ("Cyril Havermans", "Focus band founding year"),
    "5a825f5755429940e5e1a88f": ("Ba Ba Ti Ki Di Do", "Merce Cunningham years at forefront of modern dance"),
    "5a862326554299211dda2a7f": ("George Pisi", "Apia island"),
    "5a8a05bb55429970aeb70210": ("Bowling at the Pan American Games", "1991 Pan American Games host city"),
    "5a8a53e455429970aeb7028f": ("Temoer Terry", "McPherson Kansas namesake general"),
    "5ac3c373554299204fd21e3e": ("The Coral Sea (book)", "Patti Smith debut album in 1975"),
    "5ae145e855429920d523436e": ("Legacy of the Ancients", "Electronic Arts founding date"),
}
# Twenty short, fully inspected unrelated training paragraphs; different pairs.
DISTRACTOR_TITLES = [
    "Uragh Stone Circle", "Elisabeth Morrow School", "Zazie dans le Métro",
    "Goshen, Kentucky", "Singer &amp; Sons", "Anthophylax viridis", "Lijin Jose",
    "New Hampshire gubernatorial election, 1994", "Let the Northern Lights Erase Your Name",
    "KCFO", "Baroness de Calabrella", "Now That's What I Call Big Sir", "Offa of Angel",
    "Subdivision (country subdivision)", "Jane Emmet de Glehn", "Sophie Vavasseur",
    "Cool Kids of Death", "Juan Alderete", "Sarafina! (film)", "SMU Mustangs football under Mike Cavan",
]
JSON_ERROR = "Invalid JSON. Output only the JSON object. Remove all text before or after it."

def sequence_lengths(sample, tokenizer):
    messages = sample["messages"]
    lengths = []
    for i in sample["metadata"]["supervised_message_indices"]:
        prompt = tokenizer.apply_chat_template(messages[:i], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        lengths.append(len(tokenizer(prompt + messages[i]["content"] + tokenizer.eos_token,
                                     add_special_tokens=False)["input_ids"]))
    return lengths

def build_variant(source, kind, tool, questions, documents):
    sample = copy.deepcopy(source)
    md = sample["metadata"]
    qid = md["question_id"]
    original_path = Path(md.get("dataset_trace", md["source_trace"]))
    original = read(original_path)
    question = original["question"]
    actions = [copy.deepcopy(e["action"]) for e in original["events"]]
    corpus = POOL / "documents.jsonl"
    corpus_records = None
    if kind == "three_document_closed_corpus":
        title, relation = MISSING[qid]
        bridge = next(d for d in original["events"][0]["observation"]["results"] if d["title"] == title)
        n = list(MISSING).index(qid)
        corpus_records = [{k: bridge[k] for k in ("id", "title", "text")}]
        for name in DISTRACTOR_TITLES[2*n:2*n+2]:
            hits = [d for d in documents if d["title"] == name]
            if len(hits) != 1: raise ValueError((name, "ambiguous distractor"))
            corpus_records.append(hits[0])
        # Write tiny temporary corpora for actual BM25 construction, before freeze.
        staging = REVIEW / "controlled_corpora_draft"
        staging.mkdir(exist_ok=True)
        temporary = staging / (qid + ".jsonl")
        temporary.write_text("".join(json.dumps(d, ensure_ascii=False)+"\n" for d in corpus_records), encoding="utf-8")
        tool = SearchTool(temporary)
        corpus = OUT / "controlled_corpora" / (qid + ".jsonl")
        actions = [actions[0], actions[1], {"type":"finish", "answer":"Cannot determine", "evidence_ids":[bridge["id"]]}]
        md.update(task_category="insufficient_evidence", missing_relation=relation,
                  corpus_review="All three full paragraphs inspected; terminal relation absent. Original full corpus is answerable.")
    if kind == "transient_tool_failure": tool = OnceFailingSearch(tool)
    ctl = AgentController(question, tool, max_searches=3)
    messages = copy.deepcopy(sample["messages"][:2])
    supervised, injected = [], []
    for n, action in enumerate(actions):
        bad = None
        if kind == "near_miss_citation" and action["type"] == "finish":
            bad = copy.deepcopy(action)
            seen = ctl.seen_document_ids
            value = bad["evidence_ids"][0]
            replacement = value[:-1] + ("0" if value[-1] != "0" else "1")
            if replacement in seen: raise ValueError("Injected citation unexpectedly seen")
            bad["evidence_ids"][0] = replacement
        elif kind == "multiple_json_objects" and action["type"] == "finish":
            raw = json.dumps(action, ensure_ascii=False) + "\n" + json.dumps(action, ensure_ascii=False)
            try: json.loads(raw)
            except json.JSONDecodeError: pass
            else: raise ValueError("Expected invalid JSON")
            obs = ctl.record({"raw":raw}, {"error":JSON_ERROR})
            index = len(messages)
            messages.append({"role":"assistant", "content":raw})
            messages.append({"role":"user", "content":"Controller result:\n"+json.dumps(obs,ensure_ascii=False)+"\nChoose your next action."})
            injected.append(index)
        elif kind == "transient_tool_failure" and n == 0:
            bad = copy.deepcopy(action)
        if bad is not None:
            obs = ctl.step(bad)
            if "error" not in obs: raise ValueError("Fault was not rejected")
            index = append(messages, bad, obs, True)
            (supervised if kind == "transient_tool_failure" else injected).append(index)
        obs = ctl.step(action)
        if "error" in obs: raise ValueError((qid, obs))
        supervised.append(append(messages, action, obs, ctl.state["status"] == "running"))
    if ctl.state["status"] != "finished": raise ValueError("Unfinished demonstration")
    md.update(sample_id="sft_expanded_300_v1:"+qid,
        parent_source_type=md["source_type"], source_type=("controlled_missing_relation" if kind=="three_document_closed_corpus" else "controlled_recovery") if kind else md["source_type"],
        controlled_intervention=kind, supervised_message_indices=supervised,
        injected_unsupervised_message_indices=injected, search_count=ctl.state["search_count"],
        level=questions[qid]["level"], documents_sha256=sha(POOL/"documents.jsonl") if not corpus_records else None,
        behavior_tags=["evidence_based_stop", "no_redundant_successful_query"]+(["failure_recovery"] if kind and kind!="three_document_closed_corpus" else [])+(["corpus_relative_abstention"] if corpus_records else ["complete_semantic_citations"]))
    if kind:
        md.update(model_generated=False, construction="Disclosed scripted intervention on semantically reviewed trajectory; actual controller execution",
            modification="Controlled corpus/fault; original actor trace unchanged; invalid actions excluded from loss.")
    trace = {"question_id":qid, "messages":messages, **ctl.state,
             "documents_path":str(corpus), "controlled_intervention":kind}
    sample["messages"] = messages
    return sample, trace, corpus_records

def main():
    if OUT.exists(): raise FileExistsError("Frozen 300 output exists; refusing overwrite")
    config = read(ROOT/"configs/model.json")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], use_fast=True)
    old_bytes = (OLD/"train.jsonl").read_bytes()
    old = rows(OLD/"train.jsonl")
    if len(old)!=100 or sha(OLD/"train.jsonl")!="374c69ac1961bacdfd01f7a3257f6691f50a97963892ec5c8e7b61539085df6f": raise ValueError("Original 100 changed")
    questions = {q["id"]:q for q in rows(POOL/"questions.jsonl")}
    documents = rows(POOL/"documents.jsonl")
    tool = SearchTool(POOL/"documents.jsonl")
    teachers = rows(TEACHER/"train.jsonl")
    students = rows(REVIEW/"accepted_train.jsonl")
    by_id = {s["metadata"]["question_id"]:s for s in students}
    if len(teachers)!=38 or not MISSING.keys() <= by_id.keys(): raise ValueError("Input review changed")
    # Reserve reviewed controlled-corpus parents, then round-robin difficulty and
    # retrieval-availability strata. No ranking by subsequent evaluation results.
    groups = defaultdict(list)
    rng = random.Random(314159)
    for s in students:
        qid = s["metadata"]["question_id"]
        if qid not in MISSING: groups[(questions[qid]["level"], min(s["metadata"]["search_count"],2))].append(s)
    for values in groups.values(): rng.shuffle(values)
    chosen = [by_id[q] for q in MISSING]
    while len(chosen)<162:
        for key in sorted(groups):
            if groups[key] and len(chosen)<162: chosen.append(groups[key].pop())
    selected = teachers + chosen
    selected_ids = {s["metadata"]["question_id"] for s in selected}
    if len(selected_ids)!=200 or selected_ids & {s["metadata"]["question_id"] for s in old}: raise ValueError("Parent duplicate")
    built = {}
    for source in selected:
        qid = source["metadata"]["question_id"]
        if source["metadata"]["source_type"]=="teacher_generated":
            source=copy.deepcopy(source)
            source["metadata"]["task_category"]="multi_hop" if source["metadata"]["search_count"]>1 else "single_search_sufficient"
        built[qid] = build_variant(source, "three_document_closed_corpus" if qid in MISSING else None, tool, questions, documents)
    # Prefer longer positive histories for recovery. Try actual variants and
    # reject overlong candidates rather than truncating evidence or target IDs.
    eligible = sorted((s for s in selected if s["metadata"]["question_id"] not in MISSING),
        key=lambda s:(-max(sequence_lengths(built[s["metadata"]["question_id"]][0],tokenizer)),s["metadata"]["question_id"]))
    recovered = {}; skipped = []
    for kind, count in [("near_miss_citation",10),("multiple_json_objects",5),("transient_tool_failure",5)]:
        for source in eligible:
            qid=source["metadata"]["question_id"]
            if qid in recovered or (kind=="transient_tool_failure" and source["metadata"]["search_count"]>2): continue
            candidate=build_variant(built[qid][0],kind,tool,questions,documents)
            if max(sequence_lengths(candidate[0],tokenizer))>4096:
                skipped.append({"question_id":qid,"kind":kind,"reason":"would exceed 4096; no truncation"});continue
            built[qid]=candidate;recovered[qid]=kind
            if sum(k==kind for k in recovered.values())==count:break
        if sum(k==kind for k in recovered.values())!=count:raise ValueError("Not enough safe recovery parents")
    for qid,(sample,_,_) in built.items():
        if max(sequence_lengths(sample,tokenizer))>4096:raise ValueError((qid,"length overflow"))
    OUT.mkdir();(OUT/"traces").mkdir();(OUT/"controlled_corpora").mkdir();(OUT/"actor_source_snapshots").mkdir()
    additions=[];archive=[]
    for source in selected:
        qid=source["metadata"]["question_id"];sample,trace,corpus_records=built[qid];md=sample["metadata"]
        if corpus_records:
            corpus=OUT/"controlled_corpora"/(qid+".jsonl")
            corpus.write_text("".join(json.dumps(d,ensure_ascii=False)+"\n" for d in corpus_records),encoding="utf-8")
            md["documents_sha256"]=sha(corpus)
        trace["documents_sha256"]=md["documents_sha256"]
        path=OUT/"traces"/(qid+".json");write_json(path,trace)
        md.update(dataset_trace=str(path),dataset_trace_sha256=sha(path))
        original=Path(md["source_trace"])
        archived=OUT/"actor_source_snapshots"/(qid+".json");shutil.copyfile(original,archived)
        if sha(archived)!=md["source_trace_sha256"]:raise ValueError("Raw trace hash changed")
        md["archived_source_trace"]=str(archived)
        archive.append({"question_id":qid,"archived_source_trace":str(archived.relative_to(ROOT)),"sha256":sha(archived)})
        additions.append(sample)
    # Byte-for-byte original 100 prefix, including old metadata and ordering.
    with (OUT/"train.jsonl").open("xb") as file:
        file.write(old_bytes)
        for sample in additions:file.write((json.dumps(sample,ensure_ascii=False)+"\n").encode("utf-8"))
    all_samples=old+additions
    manifest={"trajectory_count":300,"unique_parent_question_count":300,"original_100_sha256":sha(OLD/"train.jsonl"),
        "original_100_prefix_bytes":len(old_bytes),"original_100_unchanged":True,"added_count":200,
        "new_assistant_teacher_parent_count":38,"new_reviewed_student_parent_count":162,"selection_seed":314159,
        "source_types":dict(Counter(s["metadata"]["source_type"] for s in all_samples)),
        "parent_source_types_of_additions":dict(Counter(s["metadata"]["parent_source_type"] for s in additions)),
        "task_categories":dict(Counter(s["metadata"]["task_category"] for s in all_samples)),
        "added_difficulty_distribution":dict(Counter(s["metadata"]["level"] for s in additions)),
        "controlled_interventions":dict(Counter(s["metadata"]["controlled_intervention"] for s in all_samples if s["metadata"]["controlled_intervention"])),
        "train_sha256":sha(OUT/"train.jsonl"),"new_pool_manifest":str(POOL/"manifest.json"),
        "raw_trace_archive":archive,"overlength_recovery_candidates_skipped":skipped,
        "limitations":["Assistant generation and review are not independent human adjudication.",
            "Hybrid source dataset: not 300 independent strong-model-generated demonstrations.",
            "Successful/repairable candidate selection is biased; review can use gold annotations after actor generation.",
            "Controlled recovery and missing-relation examples are scripted, not natural failures or naturally unanswerable Hotpot questions.",
            "Difficulty and semantic hops differ from search count; primary categories do not fully describe all behaviors.",
            "Compared with 100, source mix, behavior mix, supervised tokens and optimization steps also change.",
            "No new 300-sample training or independent capability improvement established."]}
    write_json(OUT/"manifest.json",manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['raw_trace_archive','limitations']},ensure_ascii=False,indent=2))

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
