"""Retry only failed/stopped ODS continuation tasks; preserve all old files."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import run

ROOT=run.ROOT
SRC=ROOT/'output/OpenDeepSearch_DeepSeek_续跑90题_20261028'
OUT=ROOT/'output/OpenDeepSearch_DeepSeek_续跑90题_20261028_失败项重试001'

def write(p,v):
    from uuid import uuid4
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+uuid4().hex+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)

def main():
    prior=json.loads((SRC/'progress.json').read_text(encoding='utf8'))
    if prior['phase']!='finished':raise ValueError('Initial continuation not finished')
    old=json.loads((SRC/'results.json').read_text(encoding='utf8'))
    indices=[x['index'] for x in old if x['phase'] in ('failed','stopped')]
    if not indices:print('No failures to retry');return
    if len(indices)!=20:raise ValueError('Unexpected failed set; review status first')
    OUT.mkdir(exist_ok=False,parents=True)
    write(OUT/'manifest.json',{'source':str(SRC),'indices':indices,'workers':len(indices),'retries':1,'reason':'deadline or fetch read timeout; throttle locking corrected, timeouts retained as unavailable pages','runner_sha256':hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest()})
    rows=[]
    def progress():
        counts={k:sum(x['phase']==k for x in rows) for k in ('complete','complete_with_step_limit','stopped','failed')}
        write(OUT/'progress.json',{'phase':'running' if len(rows)<len(indices) else 'finished','planned':len(indices),'finished':len(rows),'remaining':len(indices)-len(rows),'counts':counts,'pid':os.getpid(),'updated_at':datetime.now(timezone.utc).isoformat()})
    def task(i):
        try:
            run.run_one(SimpleNamespace(index=i,output=str(OUT/f'{i:03d}'),offline=False),shared_context=True)
            return {'index':i,**json.loads((OUT/f'{i:03d}'/'status.json').read_text(encoding='utf8'))}
        except BaseException as e:return {'index':i,'phase':'failed','reason':str(e)[:250]}
    progress()
    with ThreadPoolExecutor(max_workers=len(indices)) as pool:
        for f in as_completed([pool.submit(task,i) for i in indices]):
            rows.append(f.result());write(OUT/'results.json',sorted(rows,key=lambda x:x['index']));progress()
    print(json.dumps({'retry_finished':len(rows)}))

if __name__=='__main__':main()
