"""MiMo assessment against task-specific frozen CCA benchmark references."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from threading import Event,Lock,Thread

ROOT=Path(__file__).resolve().parents[2]
CCA=ROOT/'CCA'
sys.path.insert(0,str(CCA))
from cca.config import Settings,_load_yaml_settings
from cca.mimo26_eval import DirectJudge
from cca.explicit_score_parser import parse as parse_absolute
from cca.no_state_pool_batch import safe_error

PILOT=ROOT/'output/OmniThink_DeepSeek_前10题_20261027'
PROTOCOLS=ROOT/'public_release/CCA-update-20261025/research_artifacts/2026-10-25/protocols'
BENCH=CCA/'outputs/evaluation/mimo_decimal_full100_20261022'
OLD_ABLATION=CCA/'outputs/evaluation/mimo_general_ablation500_resume_001_20261023'
METRICS={'general':('Clarity','Engagement','Depth Appropriateness','Breadth'),
 'exploration':('Inquiry Relevance','Perspective Expansion','Evidence-Based Progress','Incremental Value','Adaptive Coherence')}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)


def parse_delta(raw,reference):
    def unique(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:raise ValueError('Duplicate metric')
            obj[key]=value
        return obj
    obj=json.loads(raw,parse_float=Decimal,parse_int=Decimal,object_pairs_hook=unique)
    if not isinstance(obj,dict) or set(obj)!=set(reference):raise ValueError('Wrong metric set')
    result={}
    for metric,value in obj.items():
        if not isinstance(value,Decimal) or not value.is_finite() or not Decimal(0)<=Decimal(str(reference[metric]))+value<=Decimal(10):
            raise ValueError('Invalid relative difference')
        result[metric]=float(value)
    return result


def reference_material(task,accepted):
    item=next(x for x in accepted if x['variant']=='no_ms' and x['task']==task)
    resultfile=Path(item['path']);record=read(resultfile)
    if record['status']!='complete':raise ValueError('Unaccepted archived ablation record')
    requestfile=resultfile.parent.parent/'requests'/resultfile.name
    req=read(requestfile);text=req['messages'][0]['content']
    marker='Reference benchmark scores for this exact task and each metric:\n'
    if text.count(marker)!=1:raise ValueError('Ambiguous reference material boundary')
    start=text.index('\n\n',text.index(marker)+len(marker))+2
    end=text.index('\n\n=== Material 2: TARGET ===',start)
    full=text[start:end]
    if not all('=== Round '+str(i)+' ===' in full for i in range(1,4)):
        raise ValueError('Reference is missing recorded CCA rounds')
    return full,{'request_path':str(requestfile),'request_sha256':sha(requestfile),
                 'result_path':str(resultfile),'result_sha256':sha(resultfile),'material_sha256':hashlib.sha256(full.encode()).hexdigest()}


def target_material(directory):
    status=read(directory/'status.json')
    if status['status']!='ok':raise ValueError('Target generation not complete')
    for name,h in status['artifact_sha256'].items():
        if sha(directory/name)!=h:raise ValueError('Changed sealed target artifact')
    messages=['=== Initial user request ===',(directory/'initial_input.txt').read_text(encoding='utf-8'),
      '=== Recorded workflow and visibility ===',
      'The following is an autonomous research workflow after one initial user input. No later user guidance or dialogue turns occurred. Expansion, concept extraction and outline generation are backend process evidence, not separate explanations presented to the user. The final article is the user-facing result. Timestamps identify actual process events; concurrently generated sections are not invented sequential user turns. Native module labels describe functions, not additional user-facing agents.',
      '=== Backend research outputs, in recorded completion order ===']
    records=[json.loads(x) for x in (directory/'responses.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    for event in records:
        # Final polish is represented by the actual parsed final article below.
        if event['stage']=='article_polish':continue
        for choice in event['response'].get('choices',[]):
            messages.extend([f"--- Backend module: {event['stage']}; completed: {event['timestamp']} ---",
                             choice['message'].get('content') or ''])
    messages+=['=== Final article: user-facing output ===',(directory/'article.md').read_text(encoding='utf-8'),
       '=== Native final citation registry: IDs match the final article ===',
       json.dumps(read(directory/'references.json'),ensure_ascii=False),
       '=== Saved retrieved evidence: supporting material, not user dialogue ===',
       'Each node below was actually retrieved and used during the autonomous research phase. Citations alone do not establish external factual verification.',
       (directory/'information_tree.json').read_text(encoding='utf-8')]
    text='\n\n'.join(messages)
    # Same identity hiding policy as the saved ablation material.
    for identity in ['Co-STORM','CoSTORM','LLM Baseline','RAG robot','CCA','OmniThink']:
        text=re.sub(r'(?<![\w])'+re.escape(identity)+r'(?![\w])','[anonymous system]',text,flags=re.I)
    return text


def prepare():
    generation=Path(read(PILOT/'active.json')['job_dir'])
    if read(generation/'progress.json')['phase']!='complete':raise ValueError('Generation pilot incomplete')
    dest=generation/'mimo_reference_evaluation'
    counter=1
    while dest.exists():
        counter+=1
        dest=generation/f'mimo_reference_evaluation_{counter:03d}'
    dest.mkdir(exist_ok=False)
    write(generation/'active_evaluation.json',{'directory':str(dest)})
    import tiktoken
    tokenizer=tiktoken.get_encoding('cl100k_base')
    oldaccepted=read(OLD_ABLATION/'accepted.json')
    accepted={suite:read(BENCH/suite/'accepted.json') for suite in METRICS}
    jobs=[];origins=[]
    for directory in sorted((generation/'tasks').iterdir()):
        if not directory.is_dir():continue
        task=directory.name;metadata=read(directory/'task.json')
        full,origin=reference_material(task,oldaccepted)
        target=target_material(directory)
        ccameta=read(CCA/'outputs/evaluation/mimo26_flash_20261021/views/CCA'/task/'task.json')
        if metadata['topic']!=ccameta['topic'] or metadata['intent']!=ccameta['intent']:
            raise ValueError('Task topic/intent mismatch')
        (dest/'materials').mkdir(exist_ok=True)
        (dest/'materials'/f'{task}__reference.txt').write_text(full,encoding='utf-8')
        (dest/'materials'/f'{task}__target.txt').write_text(target,encoding='utf-8')
        for suite,metrics in METRICS.items():
            source=accepted[suite][task];path=Path(source['path'])
            if sha(path)!=source['sha256']:raise ValueError('Changed accepted benchmark result')
            record=read(path)
            scores,_=parse_absolute(record['raw'],record['mapping'],metrics,True)
            if scores!=record['scores']:raise ValueError('Original benchmark score validation failed')
            reference=scores['CCA']
            prompt=(PROTOCOLS/f'{suite}_ablation_prompt.txt').read_text(encoding='utf-8')
            (dest/f'{suite}_prompt.txt').write_text(prompt,encoding='utf-8')
            text=f"Topic: {metadata['topic']}\nIntent: {metadata['intent']}\n\n=== Material 1: REFERENCE Full ===\nReference benchmark scores for this exact task and each metric:\n"+json.dumps(reference)+'\n\n'+full+'\n\n=== Material 2: TARGET ===\n'+target+'\n\n'+prompt
            # MiMo's native tokenizer is not the OpenAI tokenizer. Record this as
            # an estimate only, with a 25% margin plus the full output allowance.
            estimate=len(tokenizer.encode(text,disallowed_special=()))
            if estimate*1.25+81920>1000000:
                raise ValueError('Material exceeds estimated context budget; never truncate silently')
            request={'messages':[{'role':'user','content':text}]}
            reqpath=dest/suite/'requests'/f'{task}.json';write(reqpath,request)
            jobs.append({'task':task,'suite':suite,'index':int(task[:3]),'reference_scores':reference,
                'request_path':str(reqpath),'request_sha256':sha(reqpath),'characters':len(text),
                'utf8_bytes':len(text.encode('utf-8')),'cl100k_token_estimate':estimate,
                'benchmark_path':str(path),'benchmark_sha256':sha(path)})
        origins.append({'task':task,'target_directory':str(directory),'target_material_sha256':hashlib.sha256(target.encode()).hexdigest(),**origin})
    if len(jobs)!=20:raise ValueError('Expected ten matched tasks and two rubrics')
    write(dest/'manifest.json',{'model':'mimo-v2.6-flash','endpoint':'https://api.xiaomimimo.com/v1',
       'reference_policy':'Each task and metric uses the original accepted CCA benchmark score and frozen ablation reference text',
       'order':['REFERENCE Full','TARGET'],'workers':10,'planned':20,'max_passes':3,
       'prompt_policy':'Exact published ablation prompts; no added scoring criteria or rationale requirement',
       'prompt_sha256':{s:sha(dest/f'{s}_prompt.txt') for s in METRICS},'runner_sha256':sha(__file__),
       'target_material_version':'native_autonomous_workflow_v1','max_tokens':81920,'temperature':0,'top_p':0.9,
       'context_preflight':'cl100k estimate * 1.25 + full output allowance <= 1M; not an exact MiMo token count; no truncation',
       'comparison_type':'Supplementary anchored baseline assessment, not a component ablation or independent absolute scoring',
       'origins':origins,'jobs':jobs})
    write(dest/'progress.json',{'phase':'prepared','planned':20,'complete':0,'errors':0,'remaining':20})
    print(json.dumps({'prepared':str(dest),'planned':20,'min_chars':min(x['characters'] for x in jobs),
        'max_chars':max(x['characters'] for x in jobs),'max_utf8_bytes':max(x['utf8_bytes'] for x in jobs)},ensure_ascii=False))
    return dest


def run(dest):
    plan=read(dest/'manifest.json')
    if read(dest/'progress.json')['phase']!='prepared':raise ValueError('Use a new directory rather than overwrite an executed evaluation')
    base=Settings(_env_file=str(CCA/'.env'),**_load_yaml_settings(CCA/'configs/deepseek_flash_official.yaml'))
    base.max_retries=1;base.request_timeout=300
    if not base.mimo_api_key:raise ValueError('MiMo key missing')
    stop,done,lock=Event(),Event(),Lock()
    state={'phase':'running','planned':20,'complete':0,'errors':0,'remaining':20,'current_pass':1,
           'worker_pid':os.getpid(),'started_at':now(),'updated_at':now()}
    accepted={};failed={}
    def publish():
        with lock:
            state.update(complete=len(accepted),errors=len(failed),remaining=20-len(accepted)-len(failed),updated_at=now())
            write(dest/'progress.json',state)
    def heartbeat():
        while not done.wait(10):publish()
    def call(job,pass_number):
        result={'task':job['task'],'suite':job['suite'],'reference_scores':job['reference_scores'],
                'status':'error','pass':pass_number,'started_at':now()}
        client=DirectJudge(base.model_copy(deep=True))
        try:
            if stop.is_set():raise RuntimeError('Batch stopped after permanent provider error')
            reqpath=Path(job['request_path'])
            if sha(reqpath)!=job['request_sha256']:raise ValueError('Changed evaluation request')
            raw=client.chat_completion(messages=read(reqpath)['messages'],model='mimo-v2.6-flash',temperature=0,
                 top_p=0.9,max_tokens=81920,response_format={'type':'json_object'})
            result['raw']=raw
            delta=parse_delta(raw,job['reference_scores'])
            result.update(status='complete',deltas=delta,calibrated_target_scores={
              m:float(Decimal(str(job['reference_scores'][m]))+Decimal(str(delta[m]))) for m in delta})
        except Exception as exc:
            if getattr(exc,'status_code',None) in {400,401,402,403,404,410,413,422}:stop.set()
            result.update(**safe_error(exc))
            if isinstance(exc,ValueError):result['validation_error']=str(exc)[:180]
        result.update(usage=client.last_call_usage,finished_at=now())
        path=dest/f'attempt_{pass_number:03d}'/job['suite']/'results'/f"{job['task']}.json"
        write(path,result)
        return (job['suite'],job['task']),result,path
    publish();thread=Thread(target=heartbeat,daemon=True);thread.start()
    try:
        for pass_number in range(1,4):
            pending=[x for x in plan['jobs'] if (x['suite'],x['task']) not in accepted]
            if not pending or stop.is_set():break
            failed={};state['current_pass']=pass_number;publish()
            with ThreadPoolExecutor(max_workers=10) as pool:
                for future in as_completed([pool.submit(call,x,pass_number) for x in pending]):
                    key,r,path=future.result()
                    if r['status']=='complete':accepted[key]={'suite':key[0],'task':key[1],'path':str(path),'sha256':sha(path)}
                    else:failed[key]=r
                    write(dest/'accepted.json',list(accepted.values()));publish()
                    print(json.dumps({'suite':key[0],'task_index':int(key[1][:3]),'status':r['status'],
                        'complete':len(accepted),'errors':len(failed),'pass':pass_number}),flush=True)
        state.update(phase='complete' if len(accepted)==20 else 'blocked' if stop.is_set() else 'incomplete',finished_at=now())
        publish()
    finally:done.set();thread.join(timeout=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:
        if not args.execute:print((args.output/'progress.json').read_text(encoding='utf-8'))
        else:run(args.output)
    else:
        destination=prepare()
        if args.execute:run(destination)
