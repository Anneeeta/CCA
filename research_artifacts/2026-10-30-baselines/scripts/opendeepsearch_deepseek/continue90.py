"""First ten are preserved; run independent tasks 11-100 with a shared encoder."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import threading
import types
import run

ROOT=run.ROOT
PRIOR=ROOT/'output/OpenDeepSearch_DeepSeek_前10题_20261028'
OUT=ROOT/'output/OpenDeepSearch_DeepSeek_续跑90题_20261028'

def write(p,d):
    from uuid import uuid4
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+uuid4().hex+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');a=parser.parse_args()
    root=ROOT/'.build'/('ods-concurrent-offline-'+datetime.now().strftime('%H%M%S%f')) if a.offline else OUT
    root.mkdir(exist_ok=False,parents=True)
    indices=[1,2] if a.offline else list(range(11,101))
    if not a.offline:
        for i in range(1,11):
            p=PRIOR/f'{i:03d}'/'status.json';status=json.loads(p.read_text(encoding='utf-8'))
            if status['phase'] not in ('complete','complete_with_step_limit'):raise ValueError('Prior task incomplete')
    manifest={'indices':indices,'workers':2 if a.offline else 90,'execution':'independent agents; thread-local transport; shared locked CPU encoder','prior_tasks':str(PRIOR),'offline':a.offline,'model':'deepseek-flash','thinking':'disabled','reranker':'all-MiniLM-L6-v2','search':'DeepSeek','runner_sha256':run.hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest()}
    write(root/'manifest.json',manifest)
    done=[]; lock=threading.Lock()
    def publish():
        counts={k:sum(r['phase']==k for r in done) for k in ('complete','complete_with_step_limit','stopped','failed')}
        write(root/'progress.json',{'phase':'running' if len(done)<len(indices) else 'finished','planned':len(indices),'workers':manifest['workers'],'finished':len(done),'remaining':len(indices)-len(done),'counts':counts,'updated_at':datetime.now(timezone.utc).isoformat(),'pid':run.os.getpid()})
    def task(i):
        try:run.run_one(types.SimpleNamespace(index=i,output=str(root/f'{i:03d}'),offline=a.offline),shared_context=True)
        except BaseException as e:
            return {'index':i,'phase':'failed','reason':str(e)[:300]}
        return {'index':i,**json.loads((root/f'{i:03d}'/'status.json').read_text(encoding='utf-8'))}
    publish()
    with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
        for future in as_completed([pool.submit(task,i) for i in indices]):
            row=future.result();done.append(row);write(root/'results.json',sorted(done,key=lambda x:x['index']));publish()
    if a.offline:
        assert all(x['phase']=='complete' and x['counts']['search_requests']==2 and x['counts']['model_requests']==4 for x in done)
    print(json.dumps({'finished':len(done),'counts':json.loads((root/'progress.json').read_text(encoding='utf-8'))['counts']}),flush=True)

if __name__=='__main__':main()
