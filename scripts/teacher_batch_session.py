"""Local tool execution for assistant-authored teacher trajectories."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from search import SearchTool
from kzq_agent import AgentController

RUN = ROOT / "outputs/baselines_20261013T092302Z_9a649338"
OUT = ROOT / "data/teacher_batch_001_v1"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["init", "step"])
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("Batch must be positive")
    OUT = ROOT / "data" / f"teacher_batch_{args.batch:03d}_v1"
    if args.mode == "init":
        if OUT.exists():
            raise FileExistsError(OUT)
        excluded = set()
        for previous in (ROOT / "data").glob("teacher_batch_*_v1/selection.json"):
            excluded.update(read(previous)["ids"])
        for path in (RUN / "sft_review").glob("batch_*.reviewed.json"):
            excluded.update(p["question_id"] for p in read(path))
        for path in [ROOT / "data/sft_pilot_v1/train.jsonl", ROOT / "data/sft_reviewed_v1/train.jsonl", ROOT / "data/sft_recovery_v1/search_revision_train.jsonl"]:
            if path.exists():
                excluded.update(r["metadata"]["question_id"] for r in jsonl(path))
        path = ROOT / "data/hotpot_train_v1/review_decisions.jsonl"
        if path.exists():
            excluded.update(r["question_id"] for r in jsonl(path))
        questions = {q["id"]: q for q in jsonl(RUN / "questions.jsonl")}
        # Selection uses failure flags, not gold answers/documents.
        rows = sorted([r for r in read(RUN / "evaluation/details.json") if r["method"] == "agent" and r["question_id"] not in excluded and not r["answer_exact_match"]], key=lambda r:r["question_id"])
        if args.size < 1 or len(rows) < args.size:
            raise ValueError("Insufficient eligible questions or invalid size")
        selected = rows[:args.size]
        OUT.mkdir()
        for row in selected:
            q = questions[row["question_id"]]
            if q["split"] != "train":
                raise ValueError("Not training data")
            source = read(RUN / "agent" / (q["id"] + ".json"))
            payload = {"question_id":q["id"], "question":q["question"], "teacher":"current Codex assistant; exact model revision unavailable", "generation_protocol":"question and actual tool observations only", "base_model":read(ROOT / "configs/model.json"), "documents_sha256":digest(RUN / "documents.jsonl"), "messages":copy.deepcopy(source["messages"][:2]), "events":[], "status":"running", "search_count":0, "max_searches":3}
            save(OUT / (q["id"] + ".json"), payload)
        save(OUT / "selection.json", {"ids":[r["question_id"] for r in selected], "source_run":str(RUN), "selection":"lexicographic unanswered training IDs after exclusions; not representative sampling", "requested_size":args.size,"excluded_count":len(excluded)})
        print(json.dumps([{ "id":r["question_id"], "question":questions[r["question_id"]]["question"]} for r in selected], ensure_ascii=False))
        return
    tool = SearchTool(RUN / "documents.jsonl")
    for request in read(args.actions):
        path = OUT / (request["id"] + ".json")
        payload = read(path)
        if payload["status"] != "running":
            raise ValueError("Already finished")
        if digest(RUN / "documents.jsonl") != payload["documents_sha256"]:
            raise ValueError("Corpus changed")
        controller = AgentController(payload["question"], tool, max_searches=3)
        for event in payload["events"]:
            if controller.step(event["action"]) != event["observation"]:
                raise ValueError("Replay mismatch")
        if len(payload["events"]) >= 8:
            raise ValueError("Action limit")
        action = request["action"]
        observation = controller.step(action)
        payload["messages"].append({"role":"assistant", "content":json.dumps(action, ensure_ascii=False)})
        if controller.state["status"] == "running":
            payload["messages"].append({"role":"user", "content":"Controller result:\n"+json.dumps(observation, ensure_ascii=False)+"\nChoose your next action."})
        payload.update({key:controller.state[key] for key in ["events","status","search_count","final_answer","evidence_ids"]})
        save(path, payload)
        shown = copy.deepcopy(observation)
        if args.compact:
            for doc in shown.get("results", []):
                if len(doc["text"]) > 600:
                    doc["text"] = doc["text"][:600] + " [preview truncated; full text saved in trace]"
        print(json.dumps({"id":request["id"], "action":action, "observation":shown}, ensure_ascii=False))

if __name__ == "__main__":
    main()
