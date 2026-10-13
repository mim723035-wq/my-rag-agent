"""Resolve assistant-chosen citation titles against already returned documents."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 folder=ROOT/'data'/f'teacher_batch_{a.batch:03d}_v1'
 ids=json.loads((folder/'selection.json').read_text(encoding='utf-8'))['ids']
 actions=[]
 for plan in json.loads(a.plan.read_text(encoding='utf-8')):
  qid=ids[plan['index']]
  if 'query' in plan:
   action={'type':'search','query':plan['query']}
  else:
   trace=json.loads((folder/(qid+'.json')).read_text(encoding='utf-8'))
   docs={d['id']:d for e in trace['events'] for d in e['observation'].get('results',[])}
   cited=[]
   for title in plan['titles']:
    matches=[k for k,d in docs.items() if d['title']==title]
    if len(matches)!=1:raise ValueError((qid,title,'missing or ambiguous'))
    cited.append(matches[0])
   action={'type':'finish','answer':plan['answer'],'evidence_ids':cited}
  actions.append({'id':qid,'action':action})
 if a.output.exists():raise FileExistsError(a.output)
 a.output.write_text(json.dumps(actions,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
