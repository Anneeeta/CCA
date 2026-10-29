"""Replay the saved probe without requests, then make ONE framework finalization call."""
import json
from pathlib import Path
import importlib.resources
import yaml
from openai.types.chat import ChatCompletion
from smolagents.agents import MultiStepAgent
from run import ROOT, Runtime, StopProbe, build_agent, save


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


class ReplayRuntime(Runtime):
    def __init__(self, source, target):
        super().__init__(target)
        self.source = source
        self.model_inputs = lines(source/'model_requests.jsonl')
        self.model_outputs = lines(source/'model_responses.jsonl')
        self.tool_inputs = lines(source/'tool_requests.jsonl')
        self.tool_outputs = lines(source/'tool_responses.jsonl')
        self.mi = self.ti = 0

    def completion(self, **kwargs):
        if self.mi < len(self.model_inputs):
            kwargs.update(model='deepseek-flash',max_tokens=8192,extra_body={'thinking':{'type':'disabled'}})
            if kwargs != self.model_inputs[self.mi]: self.stop('replay_request_mismatch')
            result = ChatCompletion.model_validate(self.model_outputs[self.mi])
            self.mi += 1
            self.counts['replayed_model_calls'] += 1
            self.log('reused_model_responses', {'source':str(self.source),'index':self.mi,'response':result.model_dump()})
            return result
        if self.counts['model_requests'] >= 1: self.stop('one_finalization_call_only')
        return super().completion(**kwargs)

    def request(self, kind, payload):
        if self.ti >= len(self.tool_inputs): self.stop('no_new_search_or_fetch_authorized')
        if {'kind':kind,'payload':payload} != self.tool_inputs[self.ti]: self.stop('replay_tool_mismatch')
        record = self.tool_outputs[self.ti]
        self.ti += 1
        self.counts['replayed_'+kind] += 1
        return json.loads(record['body']) if record['status']==200 else {}


def main():
    parent = ROOT/'output/OpenDeepSearch_DeepSeek_单题探测_20261028'
    rt = ReplayRuntime(parent/'run_001', parent/'finalization_001')
    save(rt.output/'manifest.json', {'purpose':'one finalization of existing six-step probe','source':str(rt.source),'new_search_requests':0,'prompt_change':'missing ODS final_answer template filled from smolagents 1.9.2 official toolcalling_agent.yaml','not_a_clean_benchmark':True})
    try:
        agent = build_agent(rt)
        defaults = yaml.safe_load(importlib.resources.files('smolagents.prompts').joinpath('toolcalling_agent.yaml').read_text())
        agent.prompt_templates = dict(agent.prompt_templates, final_answer=defaults['final_answer'])
        agent.provide_final_answer = MultiStepAgent.provide_final_answer.__get__(agent)
        task = json.loads((rt.source/'task.json').read_text(encoding='utf-8'))
        answer = agent.run('Topic: '+task['topic']+'\nUser intent: '+task['intent'])
        assert rt.mi == 12 and rt.ti == 18 and rt.counts['model_requests']==1
        if not answer or answer.startswith('Error in generating'): rt.stop('finalization_failed')
        (rt.output/'answer.md').write_text(answer,encoding='utf-8')
        rt.status('complete_with_step_limit', answer_chars=len(answer))
    except BaseException as exc:
        rt.status('failed',reason=str(exc))
    print((rt.output/'status.json').read_text(encoding='utf-8'))


if __name__ == '__main__': main()
