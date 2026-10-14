"""Freeze unused validation bridge questions, then predict all six before scoring."""
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_hotpot import convert_question, document_id, encode_jsonl
from prepare_hotpot_train import validate_record, text_key, title_key

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/hotpot_fresh_blind_200_v1'
OUT = ROOT / 'outputs/blind_six_200_v1'
MODELS = {
    'X': None,
    'A': 'sft_reviewed_20261013T110629Z_ff70a414',
    'B': 'sft_teacher100_20261013T162355Z_1ec152e7',
    'C': 'sft_expanded300_20261013T233348Z_306e9948',
    'D': 'sft_incremental100_20261014T044718Z_7d240a0d',
    'E': 'sft_expanded400_20261014T051518Z_59b2bf47',
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    if DATA.exists() or OUT.exists():
        raise FileExistsError('Frozen blind experiment already exists; do not overwrite')
    ids, texts, titles = set(), set(), set()
    exclusion_sources = []
    # Materialized question sets and every historical prediction snapshot.
    # Partition index files contain IDs without question text and are not scanned here.
    for base in [ROOT/'data', ROOT/'outputs']:
        for p in base.rglob('questions.jsonl'):
            rows = [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
            for q in rows:
                ids.add(q['id']); texts.add(text_key(q['question']))
            exclusion_sources.append({'path':str(p.relative_to(ROOT)), 'sha256':sha(p)})
        for p in base.rglob('documents.jsonl'):
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if line.strip(): titles.add(title_key(json.loads(line)['title']))
    source=ROOT/'data/raw/hotpotqa/hotpot_dev_distractor_hf.json'
    raw=json.loads(source.read_text(encoding='utf-8-sig'))
    eligible=[]
    for r in raw:
        if r.get('type')!='bridge' or validate_record(r):continue
        if r['_id'] in ids or text_key(r['question']) in texts:continue
        if any(title_key(t) in titles for t,_ in r['supporting_facts']):continue
        eligible.append(r)
    eligible.sort(key=lambda r:r['_id'])
    random.Random(2026101406).shuffle(eligible)
    if len(eligible)<200:raise ValueError(f'Only {len(eligible)} eligible unused questions')
    selected=eligible[:200]
    pool='hotpot_fresh_blind_200_v1'
    questions=[dict(convert_question(r,'fresh_blind_test'),retrieval_pool_id=pool) for r in selected]
    docs={}
    for r in selected:
        for title,sentences in r['context']:
            did=document_id(title,sentences)
            docs[did]={'id':did,'title':title,'text':' '.join(s.strip() for s in sentences),'sentences':sentences,
                       'provenance':'HotpotQA distractor validation context'}
    assert len({q['id'] for q in questions})==200
    DATA.mkdir();OUT.mkdir()
    (DATA/'questions.jsonl').write_bytes(encode_jsonl(questions))
    (DATA/'documents.jsonl').write_bytes(encode_jsonl([docs[i] for i in sorted(docs)]))
    binding={'retrieval_pool_id':pool,'questions_sha256':sha(DATA/'questions.jsonl'),
             'documents_sha256':sha(DATA/'documents.jsonl'),'question_count':200,'document_count':len(docs)}
    save(DATA/'retrieval_binding.json',binding)
    hashes={}
    for name,folder in MODELS.items():
        if folder:
            for p in (ROOT/'outputs'/folder/'adapter').iterdir():
                if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    sources=['src/run_baselines.py','src/kzq_agent.py','src/search.py','scripts/evaluate_trace.py',
             'scripts/hotpot_evaluate_v1.py','configs/model.json']
    hashes.update({s:sha(ROOT/s) for s in sources})
    prereg={'created_utc':datetime.now(timezone.utc).isoformat(),'models':MODELS,'frozen_hashes':hashes,
            'binding':binding,'source_sha256':sha(source),'sampling_seed':2026101406,
            'eligible_count':len(eligible),'excluded_existing_question_ids':len(ids),
            'excluded_existing_document_titles':len(titles),'exclusion_question_sources':exclusion_sources,
            'selection':'200 structurally valid bridge questions; exclude prior IDs/text and all known corpus titles from gold titles',
            'primary_metric':'English Hotpot lexical answer EM; E minus X',
            'secondary_metrics':['answer F1','gold-document citation coverage','actual searches and tokens','finish rate'],
            'budget':{'max_searches':3,'max_actions':8,'max_total_tokens':768,'context_limit':4096,'language':'en'},
            'scoring_policy':'Generate all six predictions before scoring; no prompt/model/alias changes',
            'limitations':['New held-out subset of public validation, not official hidden test.',
                           'No guarantee of absence of pretraining contamination.',
                           'Gold-title isolation does not isolate all entities or distractors.',
                           'New pooled context corpus; dev scores are not directly comparable.',
                           'One training seed; six-model ranking is secondary exploratory analysis.']}
    save(OUT/'preregistration.json',prereg);save(DATA/'manifest.json',prereg)
    print(f'Frozen: questions=200 documents={len(docs)} eligible={len(eligible)}; answers not displayed',flush=True)
    state={'status':'predicting','runs':{}}
    save(OUT/'progress.json',state)
    for name,folder in MODELS.items():
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
        before=set((ROOT/'outputs').glob('baselines_*'))
        cmd=[sys.executable,str(ROOT/'src/run_baselines.py'),'--mode','agent','--language','en','--data-dir',str(DATA),
             '--max-searches','3','--max-actions','8','--max-total-tokens','768','--context-limit','4096']
        if folder:cmd+=['--adapter-dir',str(ROOT/'outputs'/folder/'adapter')]
        state['active_model']=name;save(OUT/'progress.json',state)
        with (OUT/f'{name}_prediction_console.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        new=set((ROOT/'outputs').glob('baselines_*'))-before
        assert len(new)==1
        run=new.pop();assert (run/'summary.json').exists()
        state['runs'][name]=str(run);save(OUT/'progress.json',state)
        print(f'{name}: predictions complete; not scored yet',flush=True)
    state['status']='scoring';save(OUT/'progress.json',state)
    for name,run in state['runs'].items():
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_trace.py'),'--run-dir',run,
                        '--questions',str(Path(run)/'questions.jsonl'),'--answer-language','en'],cwd=ROOT,check=True,
                       stdout=subprocess.DEVNULL)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    state['status']='completed';state.pop('active_model',None);save(OUT/'progress.json',state)
    print('All six generated and scored; frozen files unchanged.',flush=True)


if __name__=='__main__':main()
