"""Actual tool execution for new assistant-authored shard demonstrations.

The actor interface exposes questions and observations only, not annotations.
"""
import argparse,ast,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from search import SearchTool
from kzq_agent import AgentController
from retrieval_binding import verify_retrieval_binding,sha
PROMPTS={t.id:n.value for n in ast.parse((ROOT/"src/run_baselines.py").read_text(encoding="utf-8-sig")).body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
AGENT_PROMPT_EN=ast.literal_eval(PROMPTS["COMMON_RULES_EN"].func.value).strip()+ast.literal_eval(PROMPTS["AGENT_PROMPT_EN"].right)
POOL=ROOT/"data/hotpot_shards_v1/shard_01/sample_300_v1"
OUT=ROOT/"data/sft_targeted_shard_100_v1"
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
def main():
 p=argparse.ArgumentParser();p.add_argument("--actions",type=Path,required=True);args=p.parse_args()
 binding=verify_retrieval_binding(POOL);qs=rows(POOL/"questions.jsonl");tool=SearchTool(POOL/"documents.jsonl")
 OUT.mkdir(exist_ok=True);(OUT/"teacher_raw").mkdir(exist_ok=True)
 for request in read(args.actions):
  i=request["index"];q=qs[i];path=OUT/"teacher_raw"/(q["id"]+".json")
  if path.exists():trace=read(path)
  else:trace={"question_id":q["id"],"pool_index":i,"question":q["question"],"messages":[{"role":"system","content":AGENT_PROMPT_EN},{"role":"user","content":f"Question: {q['question']}\nSearch budget: 3 calls."}],"events":[],"teacher":"current Codex assistant; exact revision unavailable","documents_sha256":binding["documents_sha256"],"gold_used_during_generation":False}
  ctl=AgentController(q["question"],tool,max_searches=3)
  for e in trace["events"]:assert ctl.step(e["action"])==e["observation"]
  if "query" in request:action={"type":"search","query":request["query"]}
  elif "finish" in request:
   seen={d["title"]:d["id"] for e in trace["events"] for d in e["observation"].get("results",[])}
   action={"type":"finish","answer":request["finish"],"evidence_ids":[seen[t] for t in request["cite_titles"]]}
  else:action=request["action"]
  obs=ctl.step(action)
  trace["messages"].append({"role":"assistant","content":json.dumps(action,ensure_ascii=False)})
  if ctl.state["status"]=="running":trace["messages"].append({"role":"user","content":"Controller result:\n"+json.dumps(obs,ensure_ascii=False)+"\nChoose your next action."})
  trace.update(ctl.state);save(path,trace)
  print(json.dumps({"index":i,"action":action,"observation":obs},ensure_ascii=False))
if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
