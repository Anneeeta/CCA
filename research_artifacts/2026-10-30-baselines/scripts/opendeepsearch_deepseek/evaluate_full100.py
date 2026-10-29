"""Independent shuffled three-system assessment; reuse verified first ten only."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,random,re
from pathlib import Path
from threading import Event
import sys
import tiktoken

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('omni_ev',ROOT/'experiments/omnithink_deepseek/evaluate_reference.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
import evaluate_three_blind as blind
OUT=ROOT/'output/三系统匿名独立评分_100题_20261028_v2'
OLD=ROOT/'output/三系统匿名独立评分_前10题_20261028'
ODS0=ROOT/'output/OpenDeepSearch_DeepSeek_前10题_20261028'
ODS1=ROOT/'output/OpenDeepSearch_DeepSeek_续跑90题_20261028'
ODSR1=ROOT/'output/OpenDeepSearch_DeepSeek_续跑90题_20261028_失败项重试001'
ODSR=ROOT/'output/OpenDeepSearch_DeepSeek_续跑失败项_20261028'
OMNI0=ROOT/'output/OmniThink_DeepSeek_前10题_20261027/20261027_233635'
OMNI1=ROOT/'output/OmniThink_DeepSeek_续跑90题_20261028'
OMNIR=ROOT/'output/OmniThink_DeepSeek_续跑失败项_20261028'

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def records(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def ods_material(g):
    s=load(g/'status.json')
    if s['phase'] not in ('complete','complete_with_step_limit'):raise ValueError('Incomplete ODS '+str(g))
    meta=load(g/'task.json'); responses=records(g/'model_responses.jsonl');requests=records(g/'model_requests.jsonl')
    if len(requests)!=len(responses):raise ValueError('ODS model response missing')
    inner=[(req,res) for req,res in zip(requests,responses) if not req.get('tools') and len(req.get('messages',[]))==2]
    answers=[res['choices'][0]['message'].get('content') or '' for req,res in inner]
    contexts=[]
    for req,_ in inner:
        content=req['messages'][-1]['content']
        if not isinstance(content,str) or not content.startswith('Context:\n') or '\n\nQuestion: ' not in content:
            raise ValueError('Unexpected recorded ODS search-answer input')
        contexts.append(content[len('Context:\n'):].rsplit('\n\nQuestion: ',1)[0])
    text='\n\n'.join(['=== Initial user request ===','Topic: '+meta['topic']+'\nIntent: '+meta['intent'],
      '=== Recorded workflow and visibility ===','The following is an autonomous research workflow following one initial user input. Search queries, retrieved snippets and source-processing answers are backend evidence rather than separate user dialogue turns. The final answer is the user-facing result. No later user guidance is recorded.',
      '=== Search contexts: retrieved supporting evidence ===',*[f'--- Search context {n} ---\n{x}' for n,x in enumerate(contexts,1)],
      '=== Search-tool answers: backend process evidence ===',*[f'--- Search answer {n} ---\n{x}' for n,x in enumerate(answers,1)],
      '=== Final answer: user-facing output ===',(g/'answer.md').read_text(encoding='utf8')])
    for identity in ['Co-STORM','CoSTORM','LLM Baseline','RAG robot','CCA','OmniThink','OpenDeepSearch']:
        text=re.sub(r'(?<![\w])'+re.escape(identity)+r'(?![\w])','[anonymous system]',text,flags=re.I)
    return text

def gen_maps():
    if load(OMNIR/'progress.json')['phase']!='complete':raise ValueError('OmniThink retries not complete')
    omni={r['index']:Path(r['directory']) for root in (OMNI0,OMNI1) for r in load(root/'results.json') if r['status']=='ok'}
    omni.update({int(k):Path(v['directory']) for k,v in load(OMNIR/'accepted.json').items()})
    ods={i:ODS0/f'{i:03d}' for i in range(1,11)}
    for r in load(ODS1/'results.json'):
        if r['phase'] in ('complete','complete_with_step_limit'):ods[r['index']]=ODS1/f"{r['index']:03d}"
    for r in load(ODSR1/'results.json'):
        if r['phase'] in ('complete','complete_with_step_limit'):ods[r['index']]=ODSR1/f"{r['index']:03d}"
    if (ODSR/'accepted.json').exists():
        ods.update({int(k):Path(v['directory']) for k,v in load(ODSR/'accepted.json').items()})
    if set(omni)!=set(range(1,101)) or set(ods)!=set(range(1,101)):raise ValueError('Generation incomplete or duplicate task map')
    return omni,ods

def prepare():
    omni,ods=gen_maps(); old=load(OLD/'manifest.json');old_accepted=load(OLD/'accepted.json')
    if len(old_accepted)!=20:raise ValueError('First 10 evaluations incomplete')
    datasets=original_tasks=load(ROOT/'CCA/cca/WildSeek/data.json')
    if len(datasets)!=100:raise ValueError('Unexpected dataset size')
    ablation=load(ev.OLD_ABLATION/'accepted.json');tokenizer=tiktoken.get_encoding('cl100k_base')
    OUT.mkdir(parents=True,exist_ok=False);jobs=[];reuse=[]
    first_materials=ROOT/'output/OmniThink_DeepSeek_前10题_20261027/20261027_233635/mimo_reference_evaluation_003/materials'
    first_ods=ODS0/'mimo_reference_evaluation_001/materials'
    for job in old['jobs']:
        name=job['task'];expected=job['material_sha256']
        paths={'CCA':first_materials/(name+'__reference.txt'),'OmniThink':first_materials/(name+'__target.txt'),'OpenDeepSearch':first_ods/(name+'__target.txt')}
        if any(ev.sha(path)!=expected[key] for key,path in paths.items()):raise ValueError('First-10 materials changed')
    for entry in old_accepted:
        path=Path(entry['path'])
        if ev.sha(path)!=entry['sha256']:raise ValueError('Old rating altered')
        reuse.append({**entry,'origin':'prior_blind10'})
    for index in range(11,101):
        g,h=ods[index],omni[index];meta=load(g/'task.json');ometa=load(h/'task.json')
        if meta!=datasets[index-1] or ometa!=meta:raise ValueError(f'Task mismatch: {index}')
        task=original_tasks[index-1]; task_id=ev.task_id(index,task) if hasattr(ev,'task_id') else h.name
        if int(task_id[:3])!=index:raise ValueError('Task identifier mismatch')
        full,origin=ev.reference_material(task_id,ablation)
        materials={'CCA':full,'OmniThink':ev.target_material(h),'OpenDeepSearch':ods_material(g)}
        shuffled=list(blind.SYSTEMS);random.Random(int(hashlib.sha256(('20261028-three-way:'+task_id).encode()).hexdigest(),16)).shuffle(shuffled)
        mapping=dict(zip('ABC',shuffled))
        sections=[f'=== System {letter} ===\n'+materials[system] for letter,system in mapping.items()]
        for suite,(protocol,metrics) in blind.SPECS.items():
            prompt_path=ROOT/'CCA/prompts'/protocol/'prompt_en.txt';prompt=prompt_path.read_text(encoding='utf8')
            text=f"Topic: {meta['topic']}\nIntent: {meta['intent']}\n\n"+'\n\n'.join(sections)+'\n\n'+prompt
            count=len(tokenizer.encode(text,disallowed_special=()))
            if count*1.25+81920>1000000:raise ValueError(f'Estimated context too large for {index} {suite}; no truncation')
            req=OUT/suite/'requests'/f'{task_id}.json';ev.write(req,{'messages':[{'role':'user','content':text}]})
            jobs.append({'index':index,'task':task_id,'suite':suite,'mapping':mapping,'request':str(req),'request_sha256':ev.sha(req),'estimated_input_tokens_cl100k':count,'source_directories':{'OmniThink':str(h),'OpenDeepSearch':str(g)},'cca_origin':origin,'prompt_sha256':ev.sha(prompt_path)})
    ev.write(OUT/'manifest.json',{'policy':'Same prompts and material rules as blind first10; no fixed reference; three shuffled candidates in same request, 4 general and 5 exploratory metrics','model':'mimo-v2.6-flash','jobs':jobs,'reused':reuse,'max_retries_per_task_per_suite':5,'workers':35,'order_seed':'sha256(20261028-three-way:task)','max_input_estimate':max(j['estimated_input_tokens_cl100k'] for j in jobs)})
    ev.write(OUT/'accepted.json',reuse)
    ev.write(OUT/'progress.json',{'phase':'prepared','planned':200,'complete':20,'remaining':180,'failed':0})
    print(json.dumps({'prepared_new':len(jobs),'reuse':len(reuse),'max_input':max(j['estimated_input_tokens_cl100k'] for j in jobs)}),flush=True)

def execute():
    from cca.config import Settings,_load_yaml_settings
    from cca.mimo26_eval import DirectJudge
    manifest=load(OUT/'manifest.json');accepted={(x['suite'],x['task']):x for x in load(OUT/'accepted.json')}
    settings=Settings(**_load_yaml_settings(ROOT/'CCA/configs/deepseek_flash_official.yaml'));settings.max_retries=1;settings.request_timeout=300
    if not settings.mimo_api_key:raise ValueError('MiMo key missing')
    halt=Event();attempts={};last={}
    def publish(phase):ev.write(OUT/'progress.json',{'phase':phase,'planned':200,'complete':len(accepted),'remaining':200-len(accepted),'exhausted':sum(v>=6 and k not in accepted for k,v in attempts.items()),'provider_halt':halt.is_set(),'worker_pid':os.getpid(),'updated_at':datetime.now(timezone.utc).isoformat()})
    def call(job,n):
        client=None;r={'task':job['task'],'suite':job['suite'],'mapping':job['mapping'],'status':'error','attempt':n}
        try:
            if halt.is_set():raise RuntimeError('Provider halt')
            path=Path(job['request'])
            if ev.sha(path)!=job['request_sha256']:raise ValueError('Changed request')
            client=DirectJudge(settings.model_copy(deep=True))
            raw=client.chat_completion(messages=load(path)['messages'],model='mimo-v2.6-flash',temperature=0,top_p=.9,max_tokens=81920)
            r['raw']=raw;metrics=blind.SPECS[job['suite']][1];scores,_=blind.parse(raw,job['mapping'],metrics,True)
            if 'Explanation:' not in raw:raise ValueError('Missing explanation')
            explain=raw.split('Explanation:',1)[1]
            if any(f'[{l}|{m}]' not in explain for l in 'ABC' for m in metrics):raise ValueError('Missing evidence entries')
            r.update(status='complete',scores=scores)
        except Exception as e:
            code=getattr(e,'status_code',None);r.update(error_type=type(e).__name__,http_status=code,validation_error=str(e)[:250] if isinstance(e,ValueError) else None)
            if code in (400,401,402,403,404,410,413,422):halt.set()
        finally:
            r['usage']=list(client.last_call_usage) if client else []
            p=OUT/f'attempt_{n:03d}'/job['suite']/'results'/f"{job['task']}.json";ev.write(p,r)
        return job,r,p
    publish('running')
    for n in range(1,7):
        pending=[j for j in manifest['jobs'] if (j['suite'],j['task']) not in accepted]
        if not pending or halt.is_set():break
        with ThreadPoolExecutor(max_workers=35) as pool:
            for f in as_completed([pool.submit(call,j,n) for j in pending]):
                j,r,p=f.result();key=(j['suite'],j['task']);attempts[key]=n;last[key]=r['status']
                if r['status']=='complete':accepted[key]={'suite':key[0],'task':key[1],'path':str(p),'sha256':ev.sha(p),'origin':'new_blind90'}
                ev.write(OUT/'accepted.json',list(accepted.values()));publish('running')
    publish('complete' if len(accepted)==200 else 'blocked' if halt.is_set() else 'exhausted')
    if len(accepted)==200:
        blind.OUT=OUT;blind.summarize()

if __name__=='__main__':
    cmd=sys.argv[1] if len(sys.argv)>1 else 'prepare'
    if cmd=='prepare':prepare()
    elif cmd=='run':execute()
    else:raise ValueError('Unknown command')
