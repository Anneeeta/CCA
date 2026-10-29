"""Compare ODS to the existing CCA reference with the saved OmniThink protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('omni_eval',ROOT/'experiments/omnithink_deepseek/evaluate_reference.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
BASE=ROOT/'output/OpenDeepSearch_DeepSeek_前10题_20261028'
OMNI=ROOT/'output/OmniThink_DeepSeek_前10题_20261027/20261027_233635/mimo_reference_evaluation_003'

def records(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def prepare():
    batch=ev.read(BASE/'batch_status.json')
    if batch['phase']!='finished': raise RuntimeError('Generation still running')
    dest=BASE/'mimo_reference_evaluation_001'
    dest.mkdir(exist_ok=False)
    oldaccepted=ev.read(ev.OLD_ABLATION/'accepted.json')
    accepted={suite:ev.read(ev.BENCH/suite/'accepted.json') for suite in ev.METRICS}
    tasks=sorted((ROOT/'output/OmniThink_DeepSeek_前10题_20261027/20261027_233635/tasks').iterdir())
    if len(tasks)!=10: raise ValueError('Expected 10 matched OmniThink tasks')
    jobs=[]; missing=[]; origins=[]
    import tiktoken
    tokenizer=tiktoken.get_encoding('cl100k_base')
    for index,odir in enumerate(tasks,1):
        g=BASE/f'{index:03d}'; status=ev.read(g/'status.json')
        if status['phase'] not in ('complete','complete_with_step_limit'):
            missing.append({'index':index,'phase':status['phase']});continue
        metadata=ev.read(g/'task.json'); ometa=ev.read(odir/'task.json')
        if metadata['topic']!=ometa['topic'] or metadata['intent']!=ometa['intent']:raise ValueError('Task identity mismatch')
        full,origin=ev.reference_material(odir.name,oldaccepted)
        response_records=records(g/'model_responses.jsonl'); requests=records(g/'model_requests.jsonl')
        answers=[]
        for req,res in zip(requests,response_records):
            if not req.get('tools') and len(req.get('messages',[]))==2:
                content=res['choices'][0]['message'].get('content') or ''
                answers.append(content)
        contexts=records(g/'retrieval_contexts.jsonl')
        target='\n\n'.join(['=== Initial user request ===','Topic: '+metadata['topic']+'\nIntent: '+metadata['intent'],
          '=== Recorded workflow and visibility ===',
          'The following is an autonomous research workflow following one initial user input. Search queries, retrieved snippets and source-processing answers are backend evidence rather than separate user dialogue turns. The final answer is the user-facing result. No later user guidance is recorded.',
          '=== Search contexts: retrieved supporting evidence ===',
          *[f'--- Search context {n} ---\n{x["context"]}' for n,x in enumerate(contexts,1)],
          '=== Search-tool answers: backend process evidence ===',
          *[f'--- Search answer {n} ---\n{x}' for n,x in enumerate(answers,1)],
          '=== Final answer: user-facing output ===',(g/'answer.md').read_text(encoding='utf-8')])
        for identity in ['Co-STORM','CoSTORM','LLM Baseline','RAG robot','CCA','OmniThink','OpenDeepSearch']:
            target=re.sub(r'(?<![\w])'+re.escape(identity)+r'(?![\w])','[anonymous system]',target,flags=re.I)
        (dest/'materials').mkdir(exist_ok=True)
        (dest/'materials'/f'{odir.name}__reference.txt').write_text(full,encoding='utf-8')
        (dest/'materials'/f'{odir.name}__target.txt').write_text(target,encoding='utf-8')
        for suite,metrics in ev.METRICS.items():
            source=accepted[suite][odir.name];path=Path(source['path'])
            if ev.sha(path)!=source['sha256']:raise ValueError('Changed benchmark result')
            record=ev.read(path);scores,_=ev.parse_absolute(record['raw'],record['mapping'],metrics,True)
            if scores!=record['scores']:raise ValueError('Benchmark score mismatch')
            reference=scores['CCA']; prompt=(ev.PROTOCOLS/f'{suite}_ablation_prompt.txt').read_text(encoding='utf-8')
            (dest/f'{suite}_prompt.txt').write_text(prompt,encoding='utf-8')
            text=f"Topic: {metadata['topic']}\nIntent: {metadata['intent']}\n\n=== Material 1: REFERENCE Full ===\nReference benchmark scores for this exact task and each metric:\n"+json.dumps(reference)+'\n\n'+full+'\n\n=== Material 2: TARGET ===\n'+target+'\n\n'+prompt
            estimate=len(tokenizer.encode(text,disallowed_special=()))
            if estimate*1.25+81920>1000000:raise ValueError('Estimated context too large')
            reqpath=dest/suite/'requests'/f'{odir.name}.json';ev.write(reqpath,{'messages':[{'role':'user','content':text}]})
            jobs.append({'task':odir.name,'suite':suite,'index':index,'reference_scores':reference,'request_path':str(reqpath),'request_sha256':ev.sha(reqpath),'characters':len(text),'cl100k_token_estimate':estimate,'benchmark_path':str(path),'benchmark_sha256':ev.sha(path)})
        origins.append({'index':index,'target_dir':str(g),'target_sha256':hashlib.sha256(target.encode()).hexdigest(),**origin})
    if missing: raise ValueError('Incomplete generation: '+json.dumps(missing))
    ev.write(dest/'manifest.json',{'model':'mimo-v2.6-flash','protocol':'Same fixed task-specific CCA benchmark anchor and exact general/exploration ablation prompts as OmniThink evaluation 003','order':['CCA reference','target'],'workers':10,'planned':20,'max_passes':3,'jobs':jobs,'origins':origins,'omni_comparison':str(OMNI),'prompt_sha256':{s:ev.sha(dest/f'{s}_prompt.txt') for s in ev.METRICS},'limitations':'Two anchored pairwise assessments, not a simultaneous three-way blind comparison; autonomous backend evidence is identified as backend.'})
    ev.write(dest/'progress.json',{'phase':'prepared','planned':20,'complete':0,'errors':0,'remaining':20})
    return dest

if __name__=='__main__':
    dest=prepare()
    print('Prepared:',dest,flush=True)
    ev.run(dest)
