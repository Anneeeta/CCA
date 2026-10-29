"""Continue OmniThink on WildSeek 11-100 without modifying the first ten."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from threading import Event,BoundedSemaphore,Lock,Thread
import hashlib
import json
import os
import time
import httpx
from openai import OpenAI
import run

ROOT=run.ROOT
PRIOR=ROOT/'output/OmniThink_DeepSeek_前10题_20261027/20261027_233635'
OUT=ROOT/'output/OmniThink_DeepSeek_续跑90题_20261028'

def main():
    prior=run.readlines(PRIOR/'results.json') if False else json.loads((PRIOR/'results.json').read_text(encoding='utf-8'))
    if len(prior)!=10 or any(x['status']!='ok' for x in prior):raise ValueError('Prior ten incomplete')
    all_tasks=run.load_tasks(run.DATASET)
    if len(all_tasks)!=100:raise ValueError('Expected exactly 100 tasks')
    for i,row in enumerate(prior,1):
        if row['index']!=i or json.loads((Path(row['directory'])/'task.json').read_text(encoding='utf-8'))!=all_tasks[i-1]:raise ValueError('Prior task mismatch')
    OUT.mkdir(exist_ok=False,parents=True)
    manifest=run.manifest();manifest['tasks']=[{'index':i,'task_id':run.task_id(i,x),'input_sha256':hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()} for i,x in enumerate(all_tasks,1) if i>10]
    manifest.update(prior_tasks=str(PRIOR),task_workers=90,global_model_inflight_limit=90,continuation='indices 11-100 only',continuation_runner_sha256=run.sha(__file__))
    run.write(OUT/'manifest.json',manifest)
    base=run.settings()
    if not base.deepseek_official_api_key or not base.deepseek_search_api_key:raise ValueError('Keys unavailable')
    state={'phase':'loading_embeddings','planned':90,'complete':0,'failed':0,'remaining':90,'worker_pid':os.getpid(),'updated_at':run.now()};run.write(OUT/'progress.json',state)
    encoder=run.SharedEncoder()
    transport=httpx.Client(trust_env=False,verify=True,follow_redirects=False,timeout=httpx.Timeout(180,connect=20),limits=httpx.Limits(max_connections=100,max_keepalive_connections=50))
    client=OpenAI(api_key=base.deepseek_official_api_key,base_url='https://api.deepseek.com/v1',max_retries=0,http_client=transport)
    halt=Event();slots=BoundedSemaphore(90);finished=Event();lock=Lock();rows=[]
    def tick():
        while not finished.wait(10):
            with lock:
                state['updated_at']=run.now();run.write(OUT/'progress.json',state)
    state.update(phase='running',updated_at=run.now());run.write(OUT/'progress.json',state)
    heartbeat=Thread(target=tick,daemon=True);heartbeat.start()
    try:
        with ThreadPoolExecutor(max_workers=90) as pool:
            futures=[pool.submit(run.run_task,i,item,OUT,encoder,client,halt,slots,(base.deepseek_official_api_key,base.deepseek_search_api_key)) for i,item in enumerate(all_tasks,1) if i>10]
            for future in as_completed(futures):
                try:row=future.result()
                except Exception as e:row={'status':'failed','error_type':type(e).__name__}
                rows.append(row);run.write(OUT/'results.json',sorted(rows,key=lambda x:x.get('index',999)))
                with lock:
                    state.update(complete=sum(x['status']=='ok' for x in rows),failed=sum(x['status']!='ok' for x in rows),remaining=90-len(rows),provider_halt=halt.is_set(),updated_at=run.now());run.write(OUT/'progress.json',state)
        state.update(phase='complete' if state['complete']==90 else 'failed',finished_at=run.now(),updated_at=run.now());run.write(OUT/'progress.json',state)
    finally:finished.set();heartbeat.join(timeout=2);client.close()

if __name__=='__main__':main()
