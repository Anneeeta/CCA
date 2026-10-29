"""Pure-local descriptive pilot analysis; refuses to read a running batch."""
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean,median

ROOT=Path(__file__).resolve().parents[2]
OUTROOT=ROOT/'output/OmniThink_DeepSeek_前10题_20261027'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    job=Path(read(OUTROOT/'active.json')['job_dir'])
    state=read(job/'progress.json')
    if state['phase'] not in {'complete','failed'} or state['remaining']:
        raise RuntimeError('Content review is permitted only after the entire ten-task pilot finishes')
    results=read(job/'results.json')
    rows=[]
    for r in results:
        d=Path(r['directory'])
        changed=[name for name,h in r['artifact_sha256'].items() if sha(d/name)!=h]
        if changed:raise ValueError('Sealed task artifacts changed: '+r['task_id'])
        requests=lines(d/'requests.jsonl');responses=lines(d/'responses.jsonl')
        task=read(d/'task.json')
        text=(d/'article.md').read_text(encoding='utf-8') if (d/'article.md').exists() else ''
        refs=read(d/'references.json') if (d/'references.json').exists() else {}
        stages={}
        for response in responses:
            stage=response['stage'];u=response['response'].get('usage') or {}
            v=stages.setdefault(stage,{'calls':0,'prompt_tokens':0,'completion_tokens':0})
            v['calls']+=1
            for k in ('prompt_tokens','completion_tokens'):v[k]+=u.get(k,0) or 0
        out={'index':r['index'],'task_id':r['task_id'],'topic':task['topic'],'status':r['status'],
             'stage':r.get('stage'),'error_type':r.get('error_type'),'elapsed_seconds':r['elapsed_seconds'],
             **r['usage'],**r.get('tree',{}),'article_words':r.get('article_words',0),
             'article_characters':r.get('article_characters',0),'cited_sources':r.get('cited_sources',0),
             'article_sections':r.get('article_sections',0),'expected_sections':r.get('expected_sections',0),
             'warnings':r.get('warnings',[]),'stage_usage':stages,
             'headings':re.findall(r'^#{1,3}\s+(.+)$',text,re.M),
             'citation_ids':sorted(set(map(int,re.findall(r'\[(\d+)\]',text)))),
             'mapped_citation_ids':sorted((refs.get('url_to_unified_index') or {}).values()),
             'nonstop_responses':sum(any(x.get('finish_reason')!='stop' for x in p['response'].get('choices',[])) for p in responses)}
        rows.append(out)
    total={k:sum(r.get(k,0) for r in rows) for k in ('prompt_tokens','completion_tokens','total_tokens',
          'model_requests','model_responses','search_attempts','search_successes','unaccounted_model_requests')}
    good=[r for r in rows if r['status']=='ok']
    summary={'job_dir':str(job),'planned':10,'complete':len(good),'failed':10-len(good),'totals':total,
       'successful_task_means':{k:mean(r[k] for r in good) for k in ('total_tokens','prompt_tokens','completion_tokens',
           'search_attempts','unique_urls','article_words','cited_sources','elapsed_seconds')} if good else {},
       'observed_models':sorted({m for r in rows for m in r['observed_models']}),
       'scope':'Generation-only descriptive pilot, not a judge score or factual-verification result', 'tasks':rows}
    analysis=job/'analysis';analysis.mkdir(exist_ok=True)
    (analysis/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['index','topic','status','nodes','depth','unique_urls','cited_sources','article_words','model_requests',
          'prompt_tokens','completion_tokens','total_tokens','search_attempts','elapsed_seconds','error_type']
    with (analysis/'task_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='tasks'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
