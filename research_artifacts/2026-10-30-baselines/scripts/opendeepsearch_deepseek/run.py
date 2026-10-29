"""One authorized task, native ODS/ReAct with audited DeepSeek transports."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'external/OpenDeepSearch/src'))
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.pop('OPENAI_BASE_URL', None)
_THREAD = threading.local()
_ENCODER = None
_ENCODER_LOCK = threading.Lock()
_RATE_LOCK = threading.Lock()
_LAST_REQUEST = {}
_GLOBAL_HALT = threading.Event()
_ORIGINAL_BUILD_CONTEXT = None
_PATCH_LOCK = threading.Lock()

class RuntimeProxy:
    def __getattr__(self, name):
        if name == 'completion':
            return lambda **kwargs: _THREAD.runtime.completion(**kwargs)
        return getattr(_THREAD.runtime, name)

def throttle(kind, url=''):
    keys=[(kind, .012 if kind == 'search' else .05)]
    if kind == 'fetch': keys.append(('domain:'+urlsplit(url).netloc.lower(),1.0))
    while True:
        with _RATE_LOCK:
            delay=max([_LAST_REQUEST.get(k,0)+interval-time.monotonic() for k,interval in keys]+[0])
            if delay<=0:
                tick=time.monotonic()
                for key,_ in keys:_LAST_REQUEST[key]=tick
                return
        time.sleep(delay)


class StopProbe(BaseException):
    """Do not let native agent error recovery conceal an external/budget stop."""


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


class Runtime:
    def __init__(self, output, offline=False):
        import httpx
        import tiktoken
        from dotenv import dotenv_values
        from openai import OpenAI
        self.output, self.offline = output, offline
        self.shared_context = False
        output.mkdir(parents=True, exist_ok=False)
        self.counts, self.usage = Counter(), Counter()
        self.started = time.monotonic()
        self.halt = None
        self.encoding = tiktoken.get_encoding('cl100k_base')
        if not offline:
            vals = dotenv_values(ROOT / 'CCA/.env')
            key = vals.get('deepseek_official_api_key') or vals.get('DEEPSEEK_JUDGE_API_KEY') or vals.get('DEEPSEEK_API_KEY')
            self.search_key = vals.get('deepseek_search_api_key') or vals.get('DEEPSEEK_SEARCH_API_KEY') or vals.get('DEEPSEEK_API_KEY') or key
            if not key or not self.search_key:
                raise StopProbe('missing_credentials')
            self.http = httpx.Client(trust_env=False, verify=True, timeout=httpx.Timeout(180, connect=30))
            self.client = OpenAI(api_key=key, base_url='https://api.deepseek.com/v1', http_client=self.http, max_retries=0)
        self.status('initializing')

    def status(self, phase='running', **extra):
        save(self.output / 'status.json', dict(phase=phase, pid=os.getpid(), elapsed_seconds=round(time.monotonic()-self.started, 2), counts=dict(self.counts), usage=dict(self.usage), **extra))

    def log(self, name, data):
        with (self.output / (name+'.jsonl')).open('a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str)+'\n')

    def stop(self, reason):
        self.halt = reason
        raise StopProbe(reason)

    def check(self):
        if self.shared_context and _GLOBAL_HALT.is_set():
            self.stop('batch_provider_halt')
        if self.halt:
            raise StopProbe(self.halt)
        if time.monotonic()-self.started > 1200:
            self.stop('deadline_1200s')

    def completion(self, **kwargs):
        from openai.types.chat import ChatCompletion
        self.check()
        kwargs.update(model='deepseek-flash', max_tokens=8192, extra_body={'thinking': {'type':'disabled'}})
        estimate = int(len(self.encoding.encode(json.dumps(kwargs, ensure_ascii=False)))*1.2)+8500
        if self.usage['total_tokens']+estimate > 200000 or self.counts['model_requests'] >= 40:
            self.stop('budget_guard')
        self.counts['model_requests'] += 1
        self.log('model_requests', kwargs)
        self.status()
        if self.offline:
            n = self.counts['model_requests']
            message = {'role':'assistant', 'content':None}
            if n == 1:
                message['tool_calls'] = [{'id':'call_1','type':'function','function':{'name':'web_search','arguments':json.dumps({'query':'wind solar forecasting'})}}, {'id':'call_2','type':'function','function':{'name':'web_search','arguments':json.dumps({'query':'solar forecasting assessment'})}}]
            elif n in (2,3):
                assert 'Context:' in str(kwargs['messages']) and 'example.org' in str(kwargs['messages'])
                message['content'] = 'Forecast ensembles support wind and solar planning. Source: https://example.org/a'
            elif n == 4:
                message['tool_calls'] = [{'id':'call_3','type':'function','function':{'name':'final_answer','arguments':json.dumps({'answer':'Offline complete: forecast ensembles support planning. https://example.org/a'})}}]
            else:
                raise AssertionError('Unexpected offline model call')
            response = ChatCompletion.model_validate({'id':f'offline{n}','object':'chat.completion','created':0,'model':'offline','choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop'}], 'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})
        else:
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                self.log('errors', {'stage':'model','type':type(exc).__name__, 'message':str(exc)})
                if self.shared_context and getattr(exc,'status_code',None) in (400,401,402,403,404,410,413,422): _GLOBAL_HALT.set()
                self.stop('model_request_failed')
        self.log('model_responses', response.model_dump())
        if response.usage is None:
            self.stop('missing_usage')
        for key, val in response.usage.model_dump().items():
            if isinstance(val, int): self.usage[key] += val
        self.counts['model_responses'] += 1
        self.status()
        if response.choices[0].finish_reason not in ('stop','tool_calls'):
            self.stop('noncomplete_model_response')
        return response

    def request(self, kind, payload):
        self.check()
        self.counts[kind+'_requests'] += 1
        self.log('tool_requests', {'kind':kind,'payload':payload})
        self.status()
        if self.offline:
            result = {'results':[{'title':'Forecast evidence','url':f'https://example.org/{x}','snippet':'Ensemble forecasting supports wind and solar assessment.'} for x in ('a','b')]} if kind == 'search' else {'content':'Forecast ensembles quantify weather uncertainty. Wind power depends on hub height wind speed. Solar power depends on irradiance.'}
        else:
            try:
                throttle(kind, payload.get('url',''))
                response = self.http.post('https://api.deepseek.com/v1/tools/'+kind, json=payload, headers={'Authorization':'Bearer '+self.search_key})
                self.log('tool_responses', {'kind':kind,'status':response.status_code,'body':response.text})
                if kind == 'fetch' and response.status_code in (403,404,408,500,502,503,504):
                    self.counts['fetch_unavailable'] += 1
                    return {}
                response.raise_for_status()
                result = response.json()
            except Exception as exc:
                self.log('errors', {'stage':kind,'type':type(exc).__name__,'message':str(exc)})
                import httpx
                if kind=='fetch' and isinstance(exc,httpx.TimeoutException):
                    self.counts['fetch_unavailable'] += 1
                    self.status()
                    return {}
                if self.shared_context and getattr(getattr(exc,'response',None),'status_code',None) in (400,401,402,403,410,413,422): _GLOBAL_HALT.set()
                self.stop(kind+'_request_failed')
        if self.offline: self.log('tool_responses', {'kind':kind,'body':result})
        self.counts[kind+'_responses'] += 1
        self.status()
        return result


def build_agent(rt):
    # Concrete transport substitution: no browser is needed with official Fetch.
    class OfficialScraper:
        def __init__(self, **kwargs): pass
        async def scrape_many(self, links):
            result = {}
            for url in links:
                data = rt.request('fetch', {'url':url})
                result[url] = {'no_extraction':types.SimpleNamespace(content=data.get('content',''))}
            return result
    module = types.ModuleType('opendeepsearch.context_scraping.crawl4ai_scraper')
    module.WebScraper = OfficialScraper
    sys.modules[module.__name__] = module
    import opendeepsearch.ods_agent as ods
    import opendeepsearch.context_building.process_sources_pro as processors
    from opendeepsearch.ranking_models.base_reranker import BaseSemanticSearcher
    from opendeepsearch.serp_search.serp_search import SearchResult
    from opendeepsearch.ods_tool import OpenDeepSearchTool
    from opendeepsearch.prompts import REACT_PROMPT
    from smolagents import OpenAIServerModel, ToolCallingAgent
    from smolagents.memory import ToolCall
    from smolagents.agents import AgentGenerationError
    import importlib.resources
    import yaml
    from sentence_transformers import SentenceTransformer
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            import torch
            torch.set_num_threads(1)
            _ENCODER = SentenceTransformer(str(ROOT/'CCA/models/all-MiniLM-L6-v2'), device='cpu', local_files_only=True)
    encoder = _ENCODER

    class LocalSemanticSearcher(BaseSemanticSearcher):
        def _get_embeddings(self, texts):
            with _ENCODER_LOCK:
                return encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

    class OfficialSearch:
        def get_sources(self, query, num_results=8, stored_location=None):
            sent = query if len(query)<=128 else query[:128].rsplit(' ',1)[0]
            data = rt.request('search', {'query':sent,'count':num_results,'need_content':False})
            organic = [{'title':x.get('title',''), 'link':x['url'], 'snippet':x.get('snippet') or x.get('summary') or x.get('content') or ''} for x in data.get('results',[]) if x.get('url')]
            if not organic: rt.stop('empty_search_results')
            return SearchResult({'organic':organic})

    processors.InfinitySemanticSearcher = LocalSemanticSearcher
    ods.create_search_api = lambda **kwargs: OfficialSearch()
    ods.completion = rt.completion
    global _ORIGINAL_BUILD_CONTEXT
    with _PATCH_LOCK:
        if _ORIGINAL_BUILD_CONTEXT is None:
            _ORIGINAL_BUILD_CONTEXT = ods.build_context
    native_build_context = _ORIGINAL_BUILD_CONTEXT
    def checked_context(sources):
        if not isinstance(sources, dict): rt.stop('source_processing_failed')
        context = native_build_context(sources)
        if not context.strip(): rt.stop('empty_context')
        rt.log('retrieval_contexts', {'sources':sources,'context':context})
        return context
    ods.build_context = checked_context
    model = OpenAIServerModel('deepseek-flash',api_base='https://api.deepseek.com/v1',api_key='unused-adapter',temperature=.7)
    model.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=rt.completion)))
    tool = OpenDeepSearchTool(model_name='deepseek-flash',reranker='infinity')
    class MultiCallAgent(ToolCallingAgent):
        def step(self, memory_step):
            messages = self.write_memory_to_messages()
            memory_step.model_input_messages = messages.copy()
            try:
                message = self.model(messages, tools_to_call_from=list(self.tools.values()), stop_sequences=['Observation:'])
                memory_step.model_output_message = message
                if not message.tool_calls: raise ValueError('No tool calls')
            except Exception as exc:
                raise AgentGenerationError(f'Error in generating tool call: {exc}', self.logger) from exc
            memory_step.tool_calls = [ToolCall(name=c.function.name, arguments=c.function.arguments, id=c.id) for c in message.tool_calls]
            observations = []
            final = None
            for call in message.tool_calls:
                name, args = call.function.name, call.function.arguments or {}
                if name == 'final_answer':
                    final = args.get('answer', args) if isinstance(args,dict) else args
                else:
                    observations.append(f'Call id: {call.id}\nTool: {name}\nObservation: {self.execute_tool_call(name, args)}')
            memory_step.observations = '\n\n'.join(observations) if observations else None
            if final is not None:
                memory_step.action_output = final
                return final
            return None
    templates = dict(REACT_PROMPT)
    templates['final_answer'] = yaml.safe_load(importlib.resources.files('smolagents.prompts').joinpath('toolcalling_agent.yaml').read_text())['final_answer']
    agent = MultiCallAgent(tools=[tool], model=model, prompt_templates=templates, verbosity_level=0)
    native_final = agent.provide_final_answer
    def marked_final(*args, **kwargs):
        rt.counts['step_limit_finalizations'] += 1
        return native_final(*args, **kwargs)
    agent.provide_final_answer = marked_final
    return agent


def run_one(args, shared_context=False):
    rt = Runtime(Path(args.output), args.offline)
    if shared_context:
        rt.shared_context = True
        _THREAD.runtime = rt
    save(rt.output/'manifest.json', {'baseline':'OpenDeepSearch + ReAct','upstream_commit':'ec7aa06dc5ead71821a3d92ea56e54a8a9d16ece','task_index':args.index,'model':'deepseek-flash','thinking':'disabled','search':'DeepSeek Search','fetch':'DeepSeek Fetch','embedding':'local all-MiniLM-L6-v2 (replaces Infinity/Jina service)','native_max_steps':6,'pro_mode':True,'pages_per_query':2,'native_chunk_size':150,'native_overlap':50,'native_top_chunks':5,'token_guard':200000,'offline':args.offline,'multiple_tool_calls':'execute all and retain all observations','finalization':'smolagents default template at native step limit','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        agent = build_agent(RuntimeProxy() if shared_context else rt)
        task = json.loads((ROOT/'CCA/cca/WildSeek/data.json').read_text(encoding='utf-8'))[args.index-1]
        save(rt.output/'task.json',task)
        result = agent.run('Topic: '+task['topic']+'\nUser intent: '+task['intent'])
        if not isinstance(result,str) or not result.strip(): rt.stop('empty_answer')
        (rt.output/'answer.md').write_text(result,encoding='utf-8')
        if args.offline:
            assert rt.counts['model_requests']==4 and rt.counts['search_requests']==2 and rt.counts['fetch_requests']==4
            assert rt.usage['total_tokens']==480
            reqs = [json.loads(x) for x in (rt.output/'model_requests.jsonl').read_text(encoding='utf-8').splitlines()]
            assert 'call_1' in str(reqs[3]['messages']) and 'call_2' in str(reqs[3]['messages'])
        rt.status('complete_with_step_limit' if rt.counts['step_limit_finalizations'] else 'complete', answer_chars=len(result))
    except BaseException as exc:
        rt.log('errors', {'type':type(exc).__name__, 'message':str(exc)})
        rt.status('stopped' if isinstance(exc,StopProbe) else 'failed',reason=str(exc))
        if args.offline: raise
    print(json.dumps(json.loads((rt.output/'status.json').read_text(encoding='utf-8')),ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', type=int, default=1)
    run_one(parser.parse_args())


if __name__ == '__main__': main()
