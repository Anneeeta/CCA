"""Compare saved generation usage on the same first ten task indices, offline."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTROOT=ROOT/'output/OmniThink_DeepSeek_前10题_20261027'
SOURCES={
 'CCA':'CCA/outputs/pilot_gate/20261021_123521_full_5901ad2a/private/recovery_001/combined_results.json',
 'Co-STORM':'CCA/outputs/costorm_gate/20261021_161754_deepseek/direct_recovery_002/combined_results.json',
 'RAG':'CCA/outputs/rag_gate/20261021_180142_deepseek/combined_results.json'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    job=Path(read(OUTROOT/'active.json')['job_dir'])
    if read(job/'progress.json')['phase'] not in {'complete','failed'}:raise RuntimeError('Pilot still running')
    rows=[]
    for system,source in SOURCES.items():
        tasks=[]
        for row in sorted(read(ROOT/source),key=lambda x:x['index']):
            if row['index']>10:continue
            directory=Path(row['directory']);data=read(directory/'usage.json');u=data.get('totals',data)
            stats={k:u.get(k,u.get('total_'+k)) for k in ('prompt_tokens','completion_tokens','total_tokens')}
            if any(v is None for v in stats.values()):raise ValueError('Missing usage, not zero')
            stats['search_attempts']=data.get('deepseek_search_attempts',data.get('search_attempts'))
            if stats['search_attempts'] is None:
                logfile=directory/'search_calls.jsonl'
                if not logfile.exists():raise ValueError('Missing search accounting')
                stats['search_attempts']=len([x for x in logfile.read_text(encoding='utf-8').splitlines() if x.strip()])
            tasks.append({'index':row['index'],'directory':str(directory),**stats})
        totals={k:sum(x[k] for x in tasks) for k in ('prompt_tokens','completion_tokens','total_tokens','search_attempts')}
        rows.append({'system':system,'tasks':len(tasks),'totals':totals,'per_task':tasks,'source':source})
    omni=read(job/'analysis/summary.json')
    rows.append({'system':'OmniThink','tasks':10,'totals':{k:omni['totals'][k] for k in ('prompt_tokens','completion_tokens','total_tokens','search_attempts')}})
    (job/'analysis/historical_cost_comparison.json').write_text(json.dumps({'scope':'Generation totals for matched indices; different run dates, outputs and workflows; no equal-budget or causal claim','systems':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k!='per_task'} for row in rows],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
