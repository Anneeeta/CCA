"""Bounded retry of only failed continuation tasks, preserving attempt provenance."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse, hashlib, json, os, time
from pathlib import Path
from threading import Event,BoundedSemaphore
import httpx
from openai import OpenAI
import run as original

ROOT=original.ROOT
SOURCE=ROOT/'output/OmniThink_DeepSeek_续跑90题_20261028'
OUT=ROOT/'output/OmniThink_DeepSeek_续跑失败项_20261028'

def now():return datetime.now(timezone.utc).isoformat()

def prepare_retry():
    state=json.loads((SOURCE/'progress.json').read_text(encoding='utf8'))
    if state['phase'] not in ('complete','failed'):raise ValueError('Original batch not finished')
    rows=json.loads((SOURCE/'results.json').read_text(encoding='utf8'))
    if len(rows)!=90 or len({r['index'] for r in rows})!=90:raise ValueError('Original task map incomplete')
    return sorted(x['index'] for x in rows if x['status']!='ok')

def execute(offline=False):
    out=ROOT/'.build'/('omni-retry-offline-'+datetime.now().strftime('%H%M%S%f')) if offline else OUT
    indices=[11,12] if offline else prepare_retry()
    out.mkdir(parents=True,exist_ok=False)
    manifest={'source':str(SOURCE),'failed_indices':indices,'max_retries_per_task':5,'workers':min(90,len(indices)),
      'protocol_changes':'Retries only: non-polish model output limit raised from 2000 to 8192; transient search HTTP 500/502/503/504 gets at most one in-call retry. Prompts, model, search parameters, depth and pipeline retained. Earlier successful data preserved.',
      'source_runner_sha256':original.sha(Path(original.__file__)),'retry_runner_sha256':original.sha(__file__),'offline':offline}
    original.write(out/'manifest.json',manifest)
    accepted={};attempts={i:0 for i in indices};failures={};halt=Event()
    def publish(phase):
        original.write(out/'progress.json',{'phase':phase,'planned':len(indices),'complete':len(accepted),'remaining':len(indices)-len(accepted),'attempts':attempts,'exhausted':sum(attempts[i]>=5 and i not in accepted for i in indices),'provider_halt':halt.is_set(),'worker_pid':os.getpid(),'updated_at':now()})
    if not indices:publish('complete');return
    tasks=original.load_tasks(original.DATASET)
    if not offline:
        class RetryLM(original.AuditedLM):
            def __init__(self,client,audit,stage='default',cap=2000):
                super().__init__(client,audit,stage,8192 if cap==2000 else cap)
            def copy(self,**kwargs):return RetryLM(self.client,self.audit,self.stage,self.cap)
        original.AuditedLM=RetryLM
        base_search=original.AuditedSearch
        class RetrySearch(base_search):
            def __call__(self,queries,exclude_urls=None):
                for n in range(2):
                    try:return super().__call__(queries,exclude_urls)
                    except Exception as e:
                        if n or getattr(e,'status_code',None) not in (500,502,503,504):raise
                        time.sleep(1)
        original.AuditedSearch=RetrySearch
        base=original.settings();encoder=original.SharedEncoder()
        transport=httpx.Client(trust_env=False,verify=True,follow_redirects=False,timeout=httpx.Timeout(180,connect=20),limits=httpx.Limits(max_connections=100,max_keepalive_connections=50))
        client=OpenAI(api_key=base.deepseek_official_api_key,base_url='https://api.deepseek.com/v1',max_retries=0,http_client=transport)
        slots=BoundedSemaphore(90);keys=(base.deepseek_official_api_key,base.deepseek_search_api_key)
    def call(i,n):
        if offline:
            return {'index':i,'status':'failed' if i==12 and n==1 else 'ok','directory':str(out/f'attempt_{n:03d}'/'tasks'/str(i))}
        return original.run_task(i,tasks[i-1],out/f'attempt_{n:03d}',encoder,client,halt,slots,keys)
    publish('running')
    try:
        for attempt in range(1,6):
            pending=[i for i in indices if i not in accepted and attempts[i]<5]
            if not pending or halt.is_set():break
            with ThreadPoolExecutor(max_workers=min(90,len(pending))) as pool:
                futures={pool.submit(call,i,attempt):i for i in pending}
                for future in as_completed(futures):
                    i=futures[future]
                    try:row=future.result()
                    except Exception as exc:row={'index':i,'status':'failed','error_type':type(exc).__name__}
                    attempts[i]+=1
                    if row['status']=='ok':accepted[i]=row
                    else:failures[i]=row
                    original.write(out/f'attempt_{attempt:03d}'/'results.json',{'accepted_so_far':list(accepted.values()),'latest_failures':list(failures.values())})
                    original.write(out/'accepted.json',accepted);publish('running')
        publish('complete' if len(accepted)==len(indices) else 'blocked' if halt.is_set() else 'exhausted')
        if offline:assert len(accepted)==2 and attempts[12]==2 and attempts[11]==1
    finally:
        if not offline:client.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');execute(p.parse_args().offline)
