"""Verify an optional frozen question/corpus binding before actor generation."""
import hashlib
import json
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def verify_retrieval_binding(data_dir, documents_path=None):
    data_dir=Path(data_dir)
    binding_path=data_dir/"retrieval_binding.json"
    if not binding_path.exists():return None  # Historical libraries stay usable.
    binding=json.loads(binding_path.read_text(encoding="utf-8-sig"))
    for field,actual in [("questions",data_dir/"questions.jsonl"),("documents",Path(documents_path) if documents_path else data_dir/"documents.jsonl")]:
        if sha(actual)!=binding[field+"_sha256"]:
            raise ValueError(f"{field} does not match frozen retrieval pool {binding['retrieval_pool_id']}; refusing mixed library")
    questions=[json.loads(line) for line in (data_dir/"questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(questions)!=binding["question_count"] or any(q.get("retrieval_pool_id")!=binding["retrieval_pool_id"] for q in questions):
        raise ValueError("Question pool identity/count mismatch")
    return binding
