"""Replay 300 demonstrations and audit provenance, loss masks and isolation."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kzq_agent import AgentController
from search import SearchTool
from build_sft_teacher_100 import OnceFailingSearch, read, rows, sha, write_json
from build_sft_expanded_300 import OLD, OUT, POOL, JSON_ERROR

def resolve(value):
    path = Path(value)
    if path.exists(): return path
    normalized = str(value).replace("\\", "/")
    marker = "/search-agent/"
    if marker in normalized:
        candidate = ROOT / normalized.split(marker, 1)[1]
        if candidate.exists(): return candidate
    raise FileNotFoundError(value)

def corpus_path(trace):
    try: return resolve(trace["documents_path"])
    except FileNotFoundError:
        for candidate in [ROOT/"data/hotpot_train_v1/documents.jsonl", POOL/"documents.jsonl"]:
            if sha(candidate) == trace["documents_sha256"]: return candidate
        raise

def main():
    samples = rows(OUT/"train.jsonl")
    manifest = read(OUT/"manifest.json")
    old_bytes = (OLD/"train.jsonl").read_bytes()
    assert (OUT/"train.jsonl").read_bytes()[:len(old_bytes)] == old_bytes
    assert sha(OLD/"train.jsonl") == manifest["original_100_sha256"]
    assert sha(OUT/"train.jsonl") == manifest["train_sha256"]
    assert len(samples) == len({s["metadata"]["question_id"] for s in samples}) == 300
    assert len({s["metadata"]["sample_id"] for s in samples}) == 300
    questions = {}
    for directory in [ROOT/"data/hotpot_train_v1", POOL]:
        questions.update({q["id"]:q for q in rows(directory/"questions.jsonl")})
    tools = {}; errors = Counter(); target_keys = set(); search_counts = Counter()
    invalid_targets_excluded = 0; recovery_lengths = []; stop_exceptions = []
    for sample in samples:
        md=sample["metadata"];qid=md["question_id"]
        assert questions[qid]["split"]=="train"
        path=resolve(md["dataset_trace"])
        assert sha(path)==md["dataset_trace_sha256"]
        trace=read(path);assert trace["messages"]==sample["messages"]
        original=resolve(md.get("archived_source_trace",md["source_trace"]))
        assert sha(original)==md["source_trace_sha256"]
        corpus=corpus_path(trace);digest=sha(corpus)
        assert digest==trace["documents_sha256"]==md["documents_sha256"]
        if digest not in tools:tools[digest]=SearchTool(corpus)
        kind=md["controlled_intervention"]
        tool=OnceFailingSearch(tools[digest]) if kind=="transient_tool_failure" else tools[digest]
        ctl=AgentController(trace["question"],tool,max_searches=3)
        messages=sample["messages"]
        indices=[i for i,m in enumerate(messages) if m["role"]=="assistant"]
        supervised=set(md["supervised_message_indices"])
        injected=set(md["injected_unsupervised_message_indices"])
        assert len(indices)==len(trace["events"])
        assert supervised|injected==set(indices) and not supervised&injected
        seen=set();queries=set();earliest=None;last_search=None
        cites=set(trace["evidence_ids"])
        for idx,event in zip(indices,trace["events"]):
            try: action=json.loads(messages[idx]["content"])
            except json.JSONDecodeError:
                assert kind=="multiple_json_objects" and idx in injected
                action={"raw":messages[idx]["content"]}
                obs=ctl.record(action,{"error":JSON_ERROR})
            else:
                obs=ctl.step(action)
            assert action==event["action"] and obs==event["observation"],(qid,"replay mismatch",idx)
            if "error" in obs:
                errors[kind]+=1
                if obs["error"]=="search_failed":assert kind=="transient_tool_failure" and idx in supervised
                else:
                    assert idx in injected and idx not in supervised
                    invalid_targets_excluded+=1
            else:assert idx in supervised
            if idx in supervised:
                assert action["type"] in {"search","finish"}
                target_keys.add((md["sample_id"],idx))
            if action.get("type")=="search" and "error" not in obs:
                query=" ".join(action["query"].casefold().split())
                assert query not in queries,(qid,"repeated successful query")
                queries.add(query);last_search=idx
                seen.update(d["id"] for d in obs["results"])
                if cites and cites<=seen and earliest is None:earliest=idx
            if idx!=indices[-1]:
                expected="Controller result:\n"+json.dumps(obs,ensure_ascii=False)+"\nChoose your next action."
                assert messages[idx+1]=={"role":"user","content":expected}
        assert ctl.state["status"]=="finished" and ctl.state["search_count"]<=3
        assert cites<=seen and ctl.state["final_answer"]==trace["final_answer"]
        assert messages[-1]["role"]=="assistant" and json.loads(messages[-1]["content"])["type"]=="finish"
        search_counts[ctl.state["search_count"]]+=1
        if md["task_category"]=="insufficient_evidence":
            assert kind=="three_document_closed_corpus" and trace["final_answer"]=="Cannot determine"
            assert len(tools[digest].documents)==3 and md["missing_relation"]
            assert ctl.state["search_count"]==2
        else:
            assert cites
            if earliest!=last_search:stop_exceptions.append(qid)
    assert not stop_exceptions,("Cited chain already available before last search",stop_exceptions)
    eval_questions=[]
    for name in ["questions.jsonl","test/questions.jsonl"]:
        path=ROOT/"data/hotpot_bridge_v1"/name
        if path.exists():eval_questions.extend(rows(path))
    if not eval_questions:raise ValueError("No frozen evaluation questions found")
    eval_ids={q["id"] for q in eval_questions}
    train_ids={s["metadata"]["question_id"] for s in samples}
    assert not train_ids&eval_ids
    from prepare_hotpot_train import title_key
    all_documents = {}
    for path in [ROOT/"data/hotpot_bridge_v1/documents.jsonl", ROOT/"data/hotpot_train_v1/documents.jsonl", POOL/"documents.jsonl"]:
        all_documents.update({d["id"]:d for d in rows(path)})
    def titles(q):return {title_key(all_documents[i]["title"]) for i in q["supporting_doc_ids"]}
    eval_titles=set().union(*(titles(q) for q in eval_questions))
    train_titles=set().union(*(titles(questions[q]) for q in train_ids))
    assert eval_titles and not train_titles&eval_titles
    report={"status":"PASS","trajectories":300,"unique_training_questions":300,
        "original_100_byte_prefix":"PASS","controller_replay":"PASS","source_hashes":"PASS",
        "search_budget":"PASS","unseen_final_citations":0,"repeated_successful_queries":0,
        "obvious_avoidable_final_searches":0,"supervised_action_count":len(target_keys),
        "error_feedback_counts":dict(errors),"malformed_action_targets_excluded":invalid_targets_excluded,
        "search_count_distribution":dict(search_counts),"eval_question_overlap":0,"eval_gold_title_overlap":0,
        "limitations":["Replay and ID availability do not replace independent semantic review.",
            "Title isolation does not prove complete entity isolation or absence of pretraining contamination.",
            "Controlled tiny-corpus abstention differs from natural open-world unanswerability."]}
    token_path=OUT/"tokenized.jsonl"
    if token_path.exists():
        tokens=rows(token_path)
        assert {(t["metadata"]["sample_id"],t["metadata"]["message_index"]) for t in tokens}==target_keys
        assert len(tokens)==len(target_keys)
        for t in tokens:
            labels=t["labels"];ids=t["input_ids"]
            assert len(labels)==len(ids)==len(t["attention_mask"])<=4096
            boundary=next(i for i,label in enumerate(labels) if label!=-100)
            assert boundary>0 and all(v==-100 for v in labels[:boundary])
            assert labels[boundary:]==ids[boundary:]
            if t["metadata"]["controlled_intervention"] in {"near_miss_citation","multiple_json_objects","transient_tool_failure"}:
                recovery_lengths.append(len(ids))
        report.update(token_mask="PASS",tokenized_sha256=sha(token_path),
            maximum_sequence_tokens=max(len(t["input_ids"]) for t in tokens),
            total_supervised_tokens=sum(t["metadata"]["supervised_tokens"] for t in tokens),
            new_recovery_maximum_sequence_tokens=max(recovery_lengths))
    write_json(OUT/"validation_report.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
