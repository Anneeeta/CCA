"""Run exactly the first ten tasks, at most ten independent processes."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'output/OpenDeepSearch_DeepSeek_前10题_20261028'
BASE.mkdir(parents=True, exist_ok=False)
(BASE/'manifest.json').write_text(json.dumps({'task_indices':list(range(1,11)),'workers':10,'evaluation_not_auto_launched':True},indent=2),encoding='utf-8')
procs=[]
for i in range(1,11):
    task = BASE / f'{i:03d}'
    handle = (BASE / f'worker_{i:03d}.log').open('w',encoding='utf-8')
    proc = subprocess.Popen([sys.executable,'-B',str(ROOT/'experiments/opendeepsearch_deepseek/run.py'),'--index',str(i),'--output',str(task)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    procs.append((i,proc,handle))
while True:
    phases={}
    for i,proc,handle in procs:
        path=BASE/f'{i:03d}'/'status.json'
        try: phases[str(i)]=json.loads(path.read_text(encoding='utf-8')).get('phase','unknown') if path.exists() else 'starting'
        except (OSError,ValueError): phases[str(i)]='updating'
        if proc.poll() is not None and phases[str(i)] in ('starting','initializing','running','updating'):
            phases[str(i)]='process_exited'
    live=sum(p.poll() is None for _,p,_ in procs)
    data={'phase':'running' if live else 'finished','live':live,'tasks':phases}
    (BASE/'batch_status.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    if not live: break
    time.sleep(5)
for _,_,handle in procs:handle.close()
print(json.dumps(data))
