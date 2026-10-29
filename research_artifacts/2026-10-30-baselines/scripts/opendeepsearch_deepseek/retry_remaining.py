"""Retry remaining failures up to five times, including prior retry001."""
import argparse,json,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import run

ROOT=run.ROOT
SRC=ROOT/'output/OpenDeepSearch_DeepSeek_续跑90题_20261028'
FIRST=ROOT/'output/OpenDeepSearch_DeepSeek_续跑90题_20261028_失败项重试001'
OUT=ROOT/'output/OpenDeepSearch_DeepSeek_续跑失败项_20261028'
def write(p,v):
    from uuid import uuid4
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+uuid4().hex+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    out=ROOT/'.build'/('ods-retry-offline-'+datetime.now().strftime('%H%M%S%f')) if args.offline else OUT
    if args.offline:indices=[11,12];old=[]
    else:
        state=json.loads((FIRST/'progress.json').read_text(encoding='utf8'))
        if state['phase']!='finished':raise ValueError('First retry still running')
        old=json.loads((FIRST/'results.json').read_text(encoding='utf8'))
        if len(old)!=20:raise ValueError('First retry mapping incomplete')
        indices=sorted(r['index'] for r in old if r['phase'] not in ('complete','complete_with_step_limit'))
    out.mkdir(exist_ok=False,parents=True)
    attempts={i:1 for i in indices};accepted={};last={}
    write(out/'manifest.json',{'source':str(SRC),'first_retry':str(FIRST),'indices':indices,'max_retries_total':5,'attempt_count_starts_at':1,'runner_sha256':run.hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest(),'offline':args.offline})
    def publish(phase):write(out/'progress.json',{'phase':phase,'planned':len(indices),'complete':len(accepted),'remaining':len(indices)-len(accepted),'attempts':attempts,'exhausted':sum(attempts[i]>=5 and i not in accepted for i in indices),'worker_pid':os.getpid(),'updated_at':datetime.now(timezone.utc).isoformat()})
    def task(i,n):
        if args.offline:return {'index':i,'phase':'failed' if i==12 and n==2 else 'complete'}
        dest=out/f'attempt_{n:03d}'/f'{i:03d}'
        try:
            run.run_one(SimpleNamespace(index=i,output=str(dest),offline=False),shared_context=True)
            return {'index':i,'directory':str(dest),**json.loads((dest/'status.json').read_text(encoding='utf8'))}
        except BaseException as exc:return {'index':i,'phase':'failed','reason':str(exc)[:250]}
    publish('running')
    for attempt in range(2,6):
        pending=[i for i in indices if i not in accepted and attempts[i]<5]
        if not pending or run._GLOBAL_HALT.is_set():break
        with ThreadPoolExecutor(max_workers=min(90,len(pending))) as pool:
            for future in as_completed([pool.submit(task,i,attempt) for i in pending]):
                row=future.result();i=row['index'];attempts[i]+=1;last[i]=row
                if row['phase'] in ('complete','complete_with_step_limit'):accepted[i]=row
                write(out/'accepted.json',accepted);write(out/'last_attempt.json',last);publish('running')
    publish('complete' if len(accepted)==len(indices) else 'blocked' if run._GLOBAL_HALT.is_set() else 'exhausted')
    if args.offline:assert attempts[11]==2 and attempts[12]==3 and len(accepted)==2
if __name__=='__main__':main()
