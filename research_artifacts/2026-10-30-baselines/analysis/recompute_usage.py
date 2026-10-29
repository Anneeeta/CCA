"""Reproduce all-system generation usage from selected-run records, without model calls."""
from pathlib import Path
from collections import Counter
import hashlib, json, lzma

base = Path(__file__).resolve().parent
gen = base.parent/'generation'
group1 = json.loads((base/'group1_usage_summary.json').read_text(encoding='utf8'))
systems = {}
for source, name in [('CCA','CCA'), ('Co-STORM','Co-STORM'), ('RAG','RAG robot')]:
    item = group1[source]
    assert item['tasks'] == 100 and item['input_tokens'] + item['output_tokens'] == item['total_tokens']
    systems[name] = {'tasks': 100, 'calls': item['model_calls'], 'input': item['input_tokens'],
                     'output': item['output_tokens'], 'total': item['total_tokens']}
for filename, name, key in [('omnithink.jsonl.xz','OmniThink','usage.jsonl'),
                            ('opendeepsearch.jsonl.xz','OpenDeepSearch + ReAct','model_responses.jsonl')]:
    totals = Counter(); seen_tasks = set()
    with lzma.open(gen/filename, 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line); index = row['index']
            assert index not in seen_tasks; seen_tasks.add(index)
            files = row['files']; records = files[key]['content']; totals['calls'] += len(records)
            ids = set(); task_total = 0
            for record in records:
                ident = (record['call_id'],record['attempt']) if key == 'usage.jsonl' else record['id']
                assert ident not in ids; ids.add(ident)
                usage = record if key == 'usage.jsonl' else record['usage']
                i, o, t = (usage[k] for k in ('prompt_tokens','completion_tokens','total_tokens'))
                assert i + o == t
                totals.update({'input':i,'output':o,'total':t}); task_total += t
            status = files['status.json']['content']['usage']
            assert status['total_tokens'] == task_total
    assert seen_tasks == set(range(1,101))
    systems[name] = {'tasks':100, **totals}
for x in systems.values():
    x['mean_input'], x['mean_output'], x['mean_total'] = [x[k]/x['tasks'] for k in ('input','output','total')]
result = {'scope':'provider-reported input and output tokens in the 100 selected successful generation runs per system; excludes evaluator, preflight and superseded runs',
          'inputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'group1_usage_summary.json',gen/'omnithink.jsonl.xz',gen/'opendeepsearch.jsonl.xz']},
          'systems':systems}
(base/'generation_usage_all_systems.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print('PASS: generation usage for five systems on 100 tasks each')
