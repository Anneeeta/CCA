"""Isolated, audited OmniThink pilot. No paid calls unless --execute is explicit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock, BoundedSemaphore, Thread
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'external/OmniThink'
CCA = ROOT / 'CCA'
DATASET = CCA / 'cca/WildSeek/data.json'
OUTROOT = ROOT / 'output/OmniThink_DeepSeek_前10题_20261027'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(1, str(CCA))

import dspy
import httpx
import numpy as np
from openai import OpenAI, OpenAIError
from cca.config import Settings, _load_yaml_settings
from cca.deepseek_search import DeepSeekRetriever
from cca.costorm_network import DirectSearchPost
from cca.no_state_pool_batch import load_tasks, safe_error, task_id
from src.tools.mindmap import MindMap
from src.actions.outline_generation import OutlineGenerationModule
from src.actions.article_generation import ArticleGenerationModule
from src.actions.article_polish import ArticlePolishingModule
from src.dataclass.Article import Article


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def append(path, data, lock):
    with lock:
        with Path(path).open('a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def readlines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()] if Path(path).exists() else []


def settings():
    return Settings(_env_file=str(CCA / '.env'), **_load_yaml_settings(CCA / 'configs/deepseek_flash_official.yaml'))


def classify(prompt):
    for text, name in [('You are an analytical robot', 'expand'), ('Please analyze, summarize, and evaluate', 'reflect'),
                       ('Improve an outline', 'outline_refine'), ('Write an outline', 'outline_draft'),
                       ('faithful text editor', 'article_polish'), ('Write a Wikipedia section', 'write_section')]:
        if text in prompt:
            return name
    return 'dspy_continuation'


class BatchStopped(RuntimeError):
    pass


class IncompleteGeneration(RuntimeError):
    def __init__(self, reason):
        self.finish_reason = reason
        super().__init__('Provider response is incomplete')


class Audit:
    def __init__(self, directory, halt, api_slots, secrets=()):
        self.directory, self.halt, self.api_slots = directory, halt, api_slots
        self.lock, self.secrets = Lock(), [x for x in secrets if x]
        self.failures = []
        self.last_io = time.time()

    def record(self, filename, data):
        append(self.directory / filename, data, self.lock)
        self.last_io = time.time()

    def error(self, exc, kind):
        data = {'timestamp': now(), 'kind': kind, **safe_error(exc)}
        self.failures.append(data)
        self.record('errors.jsonl', data)
        if getattr(exc, 'halt_batch', False) or getattr(exc, 'status_code', None) in {400, 401, 402, 403, 404, 410, 413, 422}:
            self.halt.set()
        return data


class AuditedLM(dspy.dsp.LM):
    def __init__(self, client, audit, stage='default', cap=2000):
        super().__init__('deepseek-flash')
        self.client, self.audit, self.stage, self.cap = client, audit, stage, cap
        self.kwargs.update(model='deepseek-flash', temperature=1.0, top_p=0.9, max_tokens=cap, n=1)

    def copy(self, **kwargs):
        return AuditedLM(self.client, self.audit, self.stage, self.cap)

    def basic_request(self, prompt, **kwargs):
        return self(prompt, **kwargs)

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        if self.audit.halt.is_set():
            raise BatchStopped('Batch stopped after a permanent provider error')
        # DSPy may request a longer continuation. Record its effective cap.
        cap = min(max(self.cap, int(kwargs.get('max_tokens', self.cap))), 32768)
        stage = classify(prompt)
        call_id = uuid.uuid4().hex
        messages = [{'role': 'user', 'content': prompt}]
        for attempt in range(1, 3):
            started = time.monotonic()
            self.audit.record('requests.jsonl', {'timestamp': now(), 'call_id': call_id, 'attempt': attempt,
                'kind': 'model', 'stage': stage, 'messages': messages, 'model': 'deepseek-flash',
                'max_tokens': cap, 'temperature': 1.0, 'top_p': 0.9, 'thinking': 'disabled'})
            try:
                with self.audit.api_slots:
                    if self.audit.halt.is_set():
                        raise BatchStopped('Batch stopped')
                    response = self.client.chat.completions.create(model='deepseek-flash', messages=messages,
                        max_tokens=cap, temperature=1.0, top_p=0.9,
                        extra_body={'thinking': {'type': 'disabled'}})
                raw = response.model_dump()
                self.audit.record('responses.jsonl', {'timestamp': now(), 'call_id': call_id, 'attempt': attempt,
                    'stage': stage, 'elapsed_seconds': round(time.monotonic()-started, 3), 'response': raw})
                usage = raw.get('usage') or {}
                self.audit.record('usage.jsonl', {'timestamp': now(), 'call_id': call_id, 'attempt': attempt,
                    'stage': stage, 'requested_model': 'deepseek-flash', 'observed_model': raw.get('model'),
                    'elapsed_seconds': round(time.monotonic()-started, 3), **usage})
                if not all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                    raise ValueError('Missing provider token usage')
                if not response.choices:
                    raise ValueError('No model choice')
                choice = response.choices[0]
                if choice.finish_reason != 'stop':
                    raise IncompleteGeneration(choice.finish_reason)
                text = choice.message.content
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('Empty model content')
                self.history.append({'prompt': prompt, 'response': {'choices': [{'text': text}]}, 'kwargs': {}})
                return [text]
            except Exception as exc:
                self.audit.error(exc, 'model')
                transient = isinstance(exc, OpenAIError) and (getattr(exc, 'status_code', None) is None or getattr(exc, 'status_code', 0) in {408, 409, 429, 500, 502, 503, 504})
                if not transient or attempt == 2 or self.audit.halt.is_set():
                    raise
                time.sleep(2)
        raise AssertionError('Unreachable')


class AuditedSearch:
    def __init__(self, key, audit):
        self.audit = audit
        self.backend = DeepSeekRetriever(key, max_results=5, need_content=False, timeout=60,
            record_path=audit.directory / 'search_usage.jsonl', post=DirectSearchPost())

    def __call__(self, queries, exclude_urls=None):
        if self.audit.halt.is_set():
            raise BatchStopped('Batch stopped')
        queries = [queries] if isinstance(queries, str) else list(queries)
        results, seen = [], set(exclude_urls or [])
        for query in queries:
            if self.audit.halt.is_set():
                raise BatchStopped('Batch stopped')
            try:
                items = self.backend.search(query)
                data = [{'url': x.url, 'title': x.title, 'description': x.description,
                         'snippets': [x.snippet], 'metadata': x.metadata} for x in items]
                self.audit.record('search_results.jsonl', {'timestamp': now(), 'query': query, 'results': data})
                for item in data:
                    if item['url'] not in seen:
                        seen.add(item['url'])
                        results.append(item)
            except Exception as exc:
                self.audit.error(exc, 'search')
                raise
        if not results:
            raise ValueError('Search produced no usable evidence for this node')
        return results


class SharedEncoder:
    def __init__(self):
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        self.path = CCA / 'models/all-MiniLM-L6-v2'
        self.model = SentenceTransformer(str(self.path), device='cpu')
        self.lock = Lock()
        self.encode(['offline embedding preflight'])

    def encode(self, texts, **kwargs):
        with self.lock:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class LocalMindMap(MindMap):
    def __init__(self, *args, local_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = local_encoder

    def prepare_table_for_retrieval(self):
        if getattr(self, 'encoded_snippets', None) is not None:
            return
        self.encoder = self.local_encoder
        self.collected_urls, self.collected_snippets = [], []
        for info in self.get_all_infos():
            for snippet in info.get('snippets', []):
                self.collected_urls.append(info['url'])
                self.collected_snippets.append(snippet)
        if not self.collected_snippets:
            raise ValueError('Information tree contains no usable snippets')
        self.encoded_snippets = self.encoder.encode(self.collected_snippets)


def tree_stats(node, depth=1):
    descendants = [tree_stats(child, depth+1) for child in node.children.values()]
    return {'nodes': 1+sum(x['nodes'] for x in descendants),
            'depth': max([depth]+[x['depth'] for x in descendants]),
            'concepts': len(node.concept)+sum(x['concepts'] for x in descendants)}


def summarize_usage(directory):
    calls, searches = readlines(directory/'usage.jsonl'), readlines(directory/'search_usage.jsonl')
    return {**{k: sum(x.get(k, 0) or 0 for x in calls) for k in (
        'prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens')},
        'model_responses': len(calls), 'model_requests': len(readlines(directory/'requests.jsonl')),
        'search_attempts': len(searches), 'search_successes': sum(x.get('status_code') == 200 for x in searches),
        'unique_urls': len({y['url'] for x in readlines(directory/'search_results.jsonl') for y in x['results']}),
        'unaccounted_model_requests': len(readlines(directory/'requests.jsonl'))-len(calls),
        'observed_models': sorted({str(x.get('observed_model')) for x in calls})}


def run_task(index, item, job, encoder, client, halt, api_slots, keys):
    directory = job/'tasks'/task_id(index, item)
    directory.mkdir(parents=True, exist_ok=False)
    audit = Audit(directory, halt, api_slots, keys)
    started = time.monotonic()
    row = {'index': index, 'task_id': directory.name, 'directory': str(directory), 'status': 'running', 'started_at': now()}
    write(directory/'task.json', item)
    write(directory/'status.json', row)
    stage = 'initialize'
    def progress(new):
        nonlocal stage
        stage = new
        write(directory/'status.json', {**row, 'stage': stage, 'updated_at': now()})
    try:
        # OmniThink has one initial input; provide the benchmark objective upfront.
        initial = item['topic'].strip() + '\nUser research objective: ' + item['intent'].strip()
        (directory/'initial_input.txt').write_text(initial, encoding='utf-8')
        lm = AuditedLM(client, audit)
        retriever = AuditedSearch(keys[1], audit)
        mindmap = LocalMindMap(retriever=retriever, gen_concept_lm=lm, depth=2, local_encoder=encoder)
        progress('information_acquisition')
        for number, _ in enumerate(mindmap.build_map(initial), 1):
            mindmap.save_map(mindmap.root, str(directory/f'tree_checkpoint_{number:02d}.json'))
        mindmap.save_map(mindmap.root, str(directory/'information_tree.json'))
        concepts = mindmap.export_categories_and_concepts()
        (directory/'concepts.txt').write_text(concepts, encoding='utf-8')
        if not mindmap.root.concept:
            raise ValueError('No parsed initial concepts')
        progress('outline_generation')
        outline = OutlineGenerationModule(lm).generate_outline(topic=initial, mindmap=mindmap)
        (directory/'outline.md').write_text(outline, encoding='utf-8')
        base_article = Article.from_outline_str(topic=initial, outline_str=outline)
        expected_sections = base_article.get_first_level_section_names()
        if not expected_sections:
            raise ValueError('No parsed article sections')
        progress('article_generation')
        article = ArticleGenerationModule(retriever=retriever, article_gen_lm=lm, retrieve_top_k=3,
                    max_thread_num=10).generate_article(topic=initial, mindmap=mindmap, article_with_outline=base_article)
        (directory/'draft_article.md').write_text(article.to_string(), encoding='utf-8')
        write(directory/'draft_references.json', article.reference)
        progress('article_polish')
        polish_lm = AuditedLM(client, audit, cap=16384)
        article = ArticlePolishingModule(lm, polish_lm).polish_article(topic=initial, draft_article=article)
        text = article.to_string()
        (directory/'article.md').write_text(text, encoding='utf-8')
        write(directory/'references.json', article.reference)
        references = article.reference['url_to_unified_index']
        cited = sorted(set(map(int,re.findall(r'\[(\d+)\]', text))))
        valid_ids = set(references.values())
        if not text.strip() or not cited or not references or not set(cited).issubset(valid_ids):
            raise ValueError('Empty output or invalid citation mapping')
        sources = '\n'.join(f'[{idx}] {url}' for url, idx in sorted(references.items(),key=lambda x:x[1]))
        (directory/'article_with_sources.md').write_text(text+'\n\n## Sources\n\n'+sources+'\n', encoding='utf-8')
        row.update(status='ok', tree=tree_stats(mindmap.root), expected_sections=len(expected_sections),
            article_sections=len(article.get_first_level_section_names()), article_words=len(text.split()),
            article_characters=len(text), cited_sources=len(cited), warnings=[])
        if row['tree']['nodes'] == 1:
            row['warnings'].append('No parsed expansion beyond root; inspect expansion response before interpreting exploration')
        if len(audit.failures):
            row['warnings'].append('Recovered API transport errors present; retained in errors.jsonl')
    except Exception as exc:
        row.update(status='failed', stage=stage, **safe_error(exc))
        trace = traceback.format_exc()
        for secret in audit.secrets:
            trace = trace.replace(secret, '[REDACTED]')
        (directory/'diagnostic.txt').write_text(trace, encoding='utf-8')
    row.update(finished_at=now(), elapsed_seconds=round(time.monotonic()-started,3), usage=summarize_usage(directory))
    row['artifact_sha256'] = {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'status.json'}
    write(directory/'status.json', row)
    return row


def manifest():
    selected = list(enumerate(load_tasks(DATASET),1))[:10]
    if len(selected) != 10:
        raise ValueError('Expected at least ten benchmark tasks')
    commit = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    return {'upstream_url':'https://github.com/zjunlp/OmniThink', 'upstream_commit':commit,
            'dataset':str(DATASET), 'dataset_sha256':sha(DATASET),
            'tasks':[{'index':i,'task_id':task_id(i,x),'input_sha256':hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()} for i,x in selected],
            'model':'deepseek-flash','model_endpoint':'https://api.deepseek.com/v1',
            'official_mapping':'DeepSeek-V4.1-Flash-1017 per official docs on 2026-10-27',
            'search_endpoint':DeepSeekRetriever.API_URL,'search_k':5,'need_content':False,
            'input_protocol':'Topic plus dataset user intent supplied upfront; no synthetic user rounds',
            'depth':2,'depth_convention':'Upstream includes root in depth; depth2 performs one expansion layer',
            'temperature':1.0,'top_p':0.9,'thinking':False,'cache':False,
            'max_tokens':2000,'polish_max_tokens':16384,'section_retrieval_k':3,
            'task_workers':10,'global_model_inflight_limit':10,'search_qps':100,
            'proxy_mode':'direct','tls_verify':True,'transport_attempts':2,
            'polish_note':'Raised only polish output ceiling to preserve merged article; length finish is an error',
            'local_embedding':str(CCA/'models/all-MiniLM-L6-v2'),
            'patched_source_sha256':{str(p.relative_to(UPSTREAM)):sha(p) for p in sorted((UPSTREAM/'src').rglob('*.py'))},
            'runner_sha256':sha(__file__), 'generation_only':True,
            'packages':{x:importlib.metadata.version(x) for x in ('dspy-ai','openai','sentence-transformers')}}


def run(job):
    job.mkdir(parents=True,exist_ok=False)
    plan=manifest()
    write(job/'manifest.json',plan)
    patch=subprocess.check_output(['git','-C',str(UPSTREAM),'diff','--','src'],text=True,encoding='utf-8')
    (job/'upstream.patch').write_text(patch,encoding='utf-8')
    base=settings()
    if not base.deepseek_official_api_key or not base.deepseek_search_api_key:
        raise ValueError('Official DeepSeek keys missing')
    state={'phase':'loading_embeddings','planned':10,'complete':0,'failed':0,'remaining':10,
           'worker_pid':os.getpid(),'started_at':now(),'updated_at':now()}
    write(job/'progress.json',state)
    encoder=SharedEncoder()
    transport=httpx.Client(trust_env=False,verify=True,follow_redirects=False,
        timeout=httpx.Timeout(180,connect=20),limits=httpx.Limits(max_connections=30,max_keepalive_connections=20))
    client=OpenAI(api_key=base.deepseek_official_api_key,base_url='https://api.deepseek.com/v1',
                  max_retries=0,http_client=transport)
    halt,slots,finished=Event(),BoundedSemaphore(10),Event()
    rows=[]
    state_lock=Lock()
    def heartbeat():
        while not finished.wait(10):
            with state_lock:
                state['updated_at']=now()
                files=list((job/'tasks').glob('*/usage.jsonl'))+list((job/'tasks').glob('*/search_usage.jsonl'))
                state['last_task_io_at']=datetime.fromtimestamp(max([p.stat().st_mtime for p in files] or [time.time()]),timezone.utc).isoformat()
                write(job/'progress.json',state)
    state.update(phase='running',updated_at=now())
    write(job/'progress.json',state)
    thread=Thread(target=heartbeat,daemon=True);thread.start()
    try:
        tasks=list(enumerate(load_tasks(DATASET),1))[:10]
        with ThreadPoolExecutor(max_workers=10) as pool:
            pending=[pool.submit(run_task,i,item,job,encoder,client,halt,slots,
                    (base.deepseek_official_api_key,base.deepseek_search_api_key)) for i,item in tasks]
            for future in as_completed(pending):
                row=future.result();rows.append(row)
                write(job/'results.json',sorted(rows,key=lambda x:x['index']))
                with state_lock:
                    state.update(complete=sum(x['status']=='ok' for x in rows),failed=sum(x['status']=='failed' for x in rows),
                         remaining=10-len(rows),updated_at=now())
                    write(job/'progress.json',state)
                print(json.dumps({'index':row['index'],'status':row['status'],'complete':state['complete'],
                    'failed':state['failed'],'remaining':state['remaining']},ensure_ascii=True),flush=True)
        state.update(phase='complete' if state['complete']==10 else 'failed',finished_at=now(),updated_at=now())
        write(job/'progress.json',state)
    finally:
        finished.set();thread.join(timeout=2);client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.status:
        active=json.loads((OUTROOT/'active.json').read_text(encoding='utf-8'))
        print((Path(active['job_dir'])/'progress.json').read_text(encoding='utf-8'))
    elif args.execute:
        destination=args.output or OUTROOT/datetime.now().strftime('%Y%m%d_%H%M%S')
        OUTROOT.mkdir(parents=True,exist_ok=True)
        write(OUTROOT/'active.json',{'job_dir':str(destination.resolve())})
        run(destination.resolve())
    else:
        print(json.dumps(manifest(),ensure_ascii=False,indent=2))
