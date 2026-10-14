"""Compare supplemental no-search/RAG runs with existing X and D agent runs."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BLIND=ROOT/'outputs/blind_six_200_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-search-run',required=True,type=Path)
    parser.add_argument('--rag-run',required=True,type=Path)
    args=parser.parse_args()
    prior=read(BLIND/'progress.json')
    assert prior['status']=='completed'
    runs={'direct':args.no_search_run.resolve(),'rag':args.rag_run.resolve(),
          'base_agent':Path(prior['runs']['X']),'D_agent':Path(prior['runs']['D'])}
    names={'direct':'no_search','rag':'rag','base_agent':'agent','D_agent':'agent'}
    manifest=read(BLIND/'preregistration.json')
    corpus=manifest['binding']['documents_sha256'];questions=manifest['binding']['questions_sha256']
    summaries={};details={};traces={};configs={}
    for label,run in runs.items():
        assert sha(run/'documents.jsonl')==corpus
        assert sha(run/'questions.jsonl')==questions
        report=read(run/'evaluation/summary.json')
        assert report['evaluation_complete']
        summary=next(m for m in report['methods'] if m['method']==names[label])
        assert summary['expected_count']==summary['evaluated_count']==200
        assert summary['missing_or_invalid_count']==0
        summaries[label]=summary
        details[label]={r['question_id']:r for r in read(run/'evaluation/details.json') if r['method']==names[label]}
        traces[label]={qid:read(run/names[label]/f'{qid}.json') for qid in details[label]}
        configs[label]=read(run/'run_config.json')
    ids=set(details['direct'])
    assert len(ids)==200 and all(set(d)==ids for d in details.values())
    base={k:v for k,v in configs['direct']['model'].items() if k!='adapter_dir'}
    for name,config in configs.items():
        assert {k:v for k,v in config['model'].items() if k!='adapter_dir'}==base
        a=config['arguments']
        assert a['language']=='en' and a['max_searches']==3 and a['max_actions']==8
        assert a['max_total_tokens']==768 and a['context_limit']==4096
        if name!='D_agent':assert not config['model'].get('adapter_dir')
    assert configs['D_agent']['model']['adapter_dir'].endswith('sft_incremental100_20261014T044718Z_7d240a0d\\adapter') or configs['D_agent']['model']['adapter_dir'].replace('\\','/').endswith('sft_incremental100_20261014T044718Z_7d240a0d/adapter')
    for qid in ids:
        assert all(details[n][qid]['gold_answer']==details['direct'][qid]['gold_answer'] for n in runs)
        assert all(traces[n][qid]['question']==traces['direct'][qid]['question'] for n in runs)
        assert traces['direct'][qid]['search_count']==0
        assert traces['rag'][qid]['search_count']==1
        assert len(traces['direct'][qid]['model_calls'])<=1 and len(traces['rag'][qid]['model_calls'])<=1
    paired={}
    for reference in ['direct','rag','base_agent']:
        c=Counter();diff=[]
        for qid in sorted(ids):
            a=bool(details[reference][qid]['answer_exact_match'])
            b=bool(details['D_agent'][qid]['answer_exact_match'])
            c['both' if a and b else 'D_only' if b else 'reference_only' if a else 'neither']+=1
            diff.append(int(b)-int(a))
        rng=random.Random(42)
        boots=sorted(sum(rng.choice(diff) for _ in diff)/len(diff) for _ in range(5000))
        paired[reference]={'counts':dict(c),'D_minus_reference_EM':sum(diff)/len(diff),
                           'paired_bootstrap_95_interval':[boots[125],boots[4874]]}
    out={'status':'PASS','question_count':200,'corpus_sha256':corpus,'questions_sha256':questions,
         'runs':{k:str(v) for k,v in runs.items()},'summaries':summaries,'paired':paired,
         'limitations':['Supplemental comparison on a disclosed, already scored held-out set.',
                        'Fixed RAG uses the question as one BM25 query and top-3 documents.',
                        'No-search has zero citations by design; compare answer quality separately.',
                        'Same search cap does not equal actual search or token cost.',
                        'The bootstrap covers question sampling, not training-seed variability.',
                        'Gold document ID coverage does not prove semantic support.']}
    path=BLIND/'supplementary_baseline_comparison.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    keys=['answer_exact_match_rate','mean_answer_f1','complete_gold_citation_rate',
          'mean_search_count','mean_input_tokens','mean_output_tokens','finished_rate']
    print(json.dumps({'status':'PASS','results':{n:{k:s[k] for k in keys} for n,s in summaries.items()},
                      'paired':paired},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
