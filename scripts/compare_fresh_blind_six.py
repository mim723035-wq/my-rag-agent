"""Score frozen six-model predictions after all runs complete; no tuning."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/blind_six_200_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    state=read(OUT/'progress.json');assert state['status']=='completed'
    prereg=read(OUT/'preregistration.json')
    assert all(sha(ROOT/p)==h for p,h in prereg['frozen_hashes'].items())
    results={};rows={};traces={};behavior={}
    for name,path in state['runs'].items():
        run=Path(path)
        assert sha(run/'questions.jsonl')==prereg['binding']['questions_sha256']
        assert sha(run/'documents.jsonl')==prereg['binding']['documents_sha256']
        report=read(run/'evaluation/summary.json');assert report['evaluation_complete']
        m=next(x for x in report['methods'] if x['method']=='agent')
        assert m['evaluated_count']==200 and m['missing_or_invalid_count']==0
        results[name]=m
        rows[name]={r['question_id']:r for r in read(run/'evaluation/details.json') if r['method']=='agent'}
        traces[name]={q:read(run/'agent'/f'{q}.json') for q in rows[name]}
        c=Counter()
        for t in traces[name].values():
            if 'cannot determine' in (t.get('final_answer') or '').casefold():c['abstentions']+=1
            errors=[e for e in t['events'] if 'error' in e['observation']]
            if errors:c['questions_with_action_errors']+=1
            c['unseen_citation_error_events']+=sum('unseen_ids' in e['observation'] for e in errors)
        behavior[name]=dict(c)
    ids=sorted(rows['X']);assert len(ids)==200
    for name in rows:
        assert sorted(rows[name])==ids
        for q in ids:
            a,b=traces['X'][q],traces[name][q]
            assert a['runtime']==b['runtime'] and a['max_searches']==b['max_searches']==3
            assert a['messages'][:2]==b['messages'][:2]
            assert {k:v for k,v in a['model'].items() if k!='adapter_dir'}=={k:v for k,v in b['model'].items() if k!='adapter_dir'}
    pairs={};changes=[]
    for name in ['X','A','B','C','D']:
        diffs=[];counts=Counter()
        for q in ids:
            a,b=rows[name][q],rows['E'][q]
            old,new=bool(a['answer_exact_match']),bool(b['answer_exact_match'])
            cat='both_matched' if old and new else 'E_only_matched' if new else 'reference_only_matched' if old else 'both_not_matched'
            counts[cat]+=1;diffs.append(int(new)-int(old))
            if old!=new:changes.append({'reference':name,'category':cat,'question_id':q,
                'question':b['question'],'gold':b['gold_answer'],'reference_answer':a['predicted_answer'],
                'E_answer':b['predicted_answer'],'reference_trace':str(Path(state['runs'][name])/'agent'/f'{q}.json'),
                'E_trace':str(Path(state['runs']['E'])/'agent'/f'{q}.json')})
        rng=random.Random(42)
        boots=sorted(sum(rng.choice(diffs) for _ in diffs)/200 for _ in range(5000))
        pairs[name]={'counts':dict(counts),'EM_difference':sum(diffs)/200,'paired_bootstrap_95_interval':[boots[125],boots[4874]]}
    result={'status':'PASS','config_check':'PASS','question_count':200,'results':results,
            'primary_comparison':'E vs X','paired_E_vs_reference':pairs,'behavior':behavior,
            'preregistration':str(OUT/'preregistration.json'),'runs':state['runs'],
            'limitations':prereg['limitations']+['EM/F1 are lexical; document ID coverage is not semantic evidence verification.',
                'Bootstrap covers question sampling, not training-seed variability; secondary intervals are not multiple-comparison corrected.',
                'No comprehensive manual adjudication or sentence-level supporting-fact scoring.']}
    save=lambda p,o:p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
    save(OUT/'comparison.json',result);save(OUT/'paired_changes.json',changes)
    metrics=['answer_exact_match_rate','mean_answer_f1','complete_gold_citation_rate','mean_search_count','finished_rate']
    print(json.dumps({'results':{n:{k:m[k] for k in metrics} for n,m in results.items()},'pairs':pairs},indent=2))


if __name__=='__main__':main()
