"""Three-way anonymous absolute MiMo rating, first 10 matched tasks."""
import csv
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean
from threading import Event
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'CCA'))
from cca.config import Settings,_load_yaml_settings
from cca.explicit_score_parser import parse
from cca.mimo26_eval import DirectJudge
from cca.storage.persistence import write_json_file
from cca.eval_analysis import holm
from scipy.stats import wilcoxon
import tiktoken

CCA=ROOT/'CCA'; ODS=ROOT/'output/OpenDeepSearch_DeepSeek_前10题_20261028/mimo_reference_evaluation_001/materials'
OMNI=ROOT/'output/OmniThink_DeepSeek_前10题_20261027/20261027_233635/mimo_reference_evaluation_003/materials'
OUT=ROOT/'output/三系统匿名独立评分_前10题_20261028'
SPECS={'general':('user_experience_decimal_v1',('Clarity','Engagement','Depth Appropriateness','Breadth')),
       'exploration':('exploration_decimal_v1',('Inquiry Relevance','Perspective Expansion','Evidence-Based Progress','Incremental Value','Adaptive Coherence'))}
SYSTEMS=('CCA','OmniThink','OpenDeepSearch')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def status(**k):write_json_file(OUT/'progress.json',dict(updated_at=now(),worker_pid=os.getpid(),**k))

def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    enc=tiktoken.get_encoding('cl100k_base'); tasks=sorted(OMNI.glob('*__reference.txt'))
    if len(tasks)!=10:raise ValueError('Expected 10 tasks')
    jobs=[]; allmaps=[]
    for index,p in enumerate(tasks,1):
        task=p.name.removesuffix('__reference.txt')
        if int(task[:3])!=index:raise ValueError('Task ordering mismatch')
        sources={'CCA':p,'OmniThink':OMNI/(task+'__target.txt'),'OpenDeepSearch':ODS/(task+'__target.txt')}
        if sha(p)!=sha(ODS/(task+'__reference.txt')):raise ValueError('CCA material differs')
        metadata=read(ROOT/'output/OpenDeepSearch_DeepSeek_前10题_20261028'/f'{index:03d}'/'task.json')
        omni_metadata=read(ROOT/'output/OmniThink_DeepSeek_前10题_20261027/20261027_233635/tasks'/task/'task.json')
        if any(metadata.get(k)!=omni_metadata.get(k) for k in ('topic','intent')):raise ValueError('Mismatched task')
        shuffled=list(SYSTEMS);random.Random(int(hashlib.sha256(('20261028-three-way:'+task).encode()).hexdigest(),16)).shuffle(shuffled)
        mapping=dict(zip('ABC',shuffled));allmaps.append(mapping)
        sections=[f'=== System {letter} ===\n'+sources[system].read_text(encoding='utf-8') for letter,system in mapping.items()]
        for suite,(protocol,metrics) in SPECS.items():
            prompt_path=CCA/'prompts'/protocol/'prompt_en.txt';prompt=prompt_path.read_text(encoding='utf-8')
            text=f"Topic: {metadata['topic']}\nIntent: {metadata['intent']}\n\n"+'\n\n'.join(sections)+'\n\n'+prompt
            count=len(enc.encode(text,disallowed_special=()))
            if count*1.25+81920>1000000:raise ValueError('Estimated context budget exceeded')
            req=OUT/suite/'requests'/f'{task}.json';write_json_file(req,{'messages':[{'role':'user','content':text}]})
            jobs.append({'index':index,'task':task,'suite':suite,'mapping':mapping,'request':str(req),'request_sha256':sha(req),'estimated_input_tokens_cl100k':count,'material_sha256':{k:sha(v) for k,v in sources.items()},'prompt_sha256':sha(prompt_path)})
    if len({tuple(m.values()) for m in allmaps})<2:raise ValueError('Order was not varied')
    write_json_file(OUT/'manifest.json',{'model':'mimo-v2.6-flash','policy':'Three anonymous candidates together; seeded independent task shuffle; no benchmark scores or fixed reference supplied; four general and five exploration metrics; one judge; two calls per task.','order_seed':'sha256(20261028-three-way:task)','generation_material':'exact previously used reference and targets; backend visibility as labelled','workers':10,'max_passes':3,'jobs':jobs,'runner_sha256':sha(__file__)})
    status(phase='prepared',planned=20,complete=0,errors=0,remaining=20)
    print(json.dumps({'prepared':20,'max_estimated_input':max(j['estimated_input_tokens_cl100k'] for j in jobs),'distinct_orders':len({tuple(m.values()) for m in allmaps})}),flush=True)

def run():
    manifest=read(OUT/'manifest.json');jobs=manifest['jobs'];accepted={};stop=Event();failed={}
    settings=Settings(**_load_yaml_settings(CCA/'configs/deepseek_flash_official.yaml'));settings.max_retries=1;settings.request_timeout=300
    if not settings.mimo_api_key:raise ValueError('Missing MiMo API key')
    def publish(phase,passno):status(phase=phase,planned=20,complete=len(accepted),errors=len(failed),remaining=20-len(accepted)-len(failed),current_pass=passno)
    def call(job,attempt):
        client=None;r={'task':job['task'],'suite':job['suite'],'mapping':job['mapping'],'status':'error','attempt':attempt}
        try:
            if stop.is_set():raise RuntimeError('Provider stop')
            path=Path(job['request']);
            if sha(path)!=job['request_sha256']:raise ValueError('Request changed')
            client=DirectJudge(settings.model_copy(deep=True))
            raw=client.chat_completion(messages=read(path)['messages'],model='mimo-v2.6-flash',temperature=0,top_p=0.9,max_tokens=81920)
            r['raw']=raw;metrics=SPECS[job['suite']][1]
            scores,_=parse(raw,job['mapping'],metrics,require_one_decimal=True)
            if 'Explanation:' not in raw:raise ValueError('Missing explanation')
            explanation=raw.split('Explanation:',1)[1]
            if any(f'[{letter}|{metric}]' not in explanation for letter in 'ABC' for metric in metrics):raise ValueError('Missing evidence entries')
            r.update(status='complete',scores=scores)
        except Exception as e:
            code=getattr(e,'status_code',None)
            r.update(error_type=type(e).__name__,http_status=code,validation_error=str(e)[:300] if isinstance(e,ValueError) else None)
            if code in (400,401,402,403,404,410,413,422):stop.set()
        finally:
            r['usage']=list(client.last_call_usage) if client else []
            out=OUT/f'attempt_{attempt:03d}'/job['suite']/'results'/f"{job['task']}.json";write_json_file(out,r)
        return job,r,out
    publish('running',1)
    for attempt in range(1,4):
        pending=[j for j in jobs if (j['suite'],j['task']) not in accepted]
        if not pending or stop.is_set():break
        failed={};publish('running',attempt)
        with ThreadPoolExecutor(max_workers=10) as pool:
            for f in as_completed([pool.submit(call,j,attempt) for j in pending]):
                j,r,path=f.result();key=(j['suite'],j['task'])
                if r['status']=='complete':accepted[key]={'suite':key[0],'task':key[1],'path':str(path),'sha256':sha(path)}
                else:failed[key]=r
                write_json_file(OUT/'accepted.json',list(accepted.values()));publish('running',attempt)
    publish('complete' if len(accepted)==20 else 'blocked' if stop.is_set() else 'incomplete',attempt)

def summarize():
    accepted_rows=read(OUT/'accepted.json');rows=[]; tests=[]
    if len(accepted_rows) not in (20,200):raise ValueError('Evaluation incomplete')
    expected_tasks=len(accepted_rows)//2
    for suite,(_,metrics) in SPECS.items():
        records=[]
        for x in accepted_rows:
            if x['suite']!=suite:continue
            path=Path(x['path']);r=read(path)
            if sha(path)!=x['sha256'] or parse(r['raw'],r['mapping'],metrics,True)[0]!=r['scores']:raise ValueError('Validation mismatch')
            records.append(r)
        if len(records)!=expected_tasks:raise ValueError('Missing tasks')
        for metric in metrics:
            vals={system:[r['scores'][system][metric] for r in records] for system in SYSTEMS}
            for r in records:rows.append({'task':r['task'],'suite':suite,'metric':metric,**{s:r['scores'][s][metric] for s in SYSTEMS}})
            t={'suite':suite,'metric':metric,'n':len(records),**{s:mean(v) for s,v in vals.items()}}
            for a,b,key in [('CCA','OmniThink','cca_omni'),('CCA','OpenDeepSearch','cca_ods'),('OmniThink','OpenDeepSearch','omni_ods')]:
                diff=[round(x-y,1) for x,y in zip(vals[a],vals[b])]
                t['p_'+key]=float(wilcoxon(diff,zero_method='pratt',method='auto').pvalue) if any(diff) else 1.0
            tests.append(t)
    # All 27 comparisons are reported as one Holm family for this exploratory analysis.
    p=holm([r['p_'+key] for r in tests for key in ('cca_omni','cca_ods','omni_ods')])
    for i,r in enumerate(tests):
        for j,key in enumerate(('cca_omni','cca_ods','omni_ods')):r['holm27_'+key]=p[3*i+j]
    with (OUT/'逐题评分.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    attempts=[read(p) for p in OUT.glob('attempt_*/**/results/*.json')]
    usage={k:sum(u.get(k,0) or 0 for r in attempts for u in r.get('usage',[])) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    write_json_file(OUT/'summary.json',{'n':expected_tasks,'metrics':tests,'usage_including_failures':usage,'judge':'mimo-v2.6-flash','evaluation':'three candidates together, anonymous shuffled labels, no anchors','holm_family':'all 27 two-sided pairwise metric tests'})
    lines=[f'# 三系统匿名独立评分：{expected_tasks}题','','同题三个系统一次同时送评；每题匿名随机排列。CCA 重新独立评分，未提供历史分数。MiMo 单评审，一般4项和探索5项各评一次。','', '| 指标 | CCA | OmniThink | OpenDeepSearch |','|---|---:|---:|---:|']
    for r in tests:lines.append(f"| {r['metric']} | {r['CCA']:.2f} | {r['OmniThink']:.2f} | {r['OpenDeepSearch']:.2f} |")
    lines+=['',f'配对双侧 Wilcoxon 的27项比较一起做 Holm 校正，具体 p 值见 summary.json。共{expected_tasks}题，单评审。','',f'本目录评估 token（含失败重试；复用评分的历史用量另计）：{usage}。']
    (OUT/'三系统匿名评分.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)

if __name__=='__main__':
    command=sys.argv[1] if len(sys.argv)>1 else 'prepare'
    if command=='prepare':prepare()
    elif command=='run':run();summarize() if read(OUT/'progress.json')['phase']=='complete' else None
    elif command=='summarize':summarize()
    else:raise ValueError('Unknown command')
