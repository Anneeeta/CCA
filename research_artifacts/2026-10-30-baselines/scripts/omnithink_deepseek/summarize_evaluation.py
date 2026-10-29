"""Validate accepted anchored ratings, then summarize paired task-level scores."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean,median

spec=importlib.util.spec_from_file_location('omni_reference',Path(__file__).with_name('evaluate_reference.py'))
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
from scipy.stats import wilcoxon
from cca.eval_analysis import holm,bootstrap_ci


def main():
    generation=Path(ev.read(ev.PILOT/'active.json')['job_dir'])
    root=Path(ev.read(generation/'active_evaluation.json')['directory'])
    progress=ev.read(root/'progress.json')
    if progress['phase'] not in {'complete','incomplete','blocked'}:raise RuntimeError('Evaluation still running')
    accepted=ev.read(root/'accepted.json') if (root/'accepted.json').exists() else []
    flat=[];tests=[];groups={}
    for suite,metrics in ev.METRICS.items():
        rows=[]
        for origin in accepted:
            if origin['suite']!=suite:continue
            path=Path(origin['path'])
            if ev.sha(path)!=origin['sha256']:raise ValueError('Changed accepted rating')
            row=ev.read(path)
            if ev.parse_delta(row['raw'],row['reference_scores'])!=row['deltas']:raise ValueError('Score revalidation failed')
            rows.append(row)
            for metric in metrics:
                delta=row['deltas'][metric];reference=row['reference_scores'][metric]
                flat.append({'task_index':int(row['task'][:3]),'task':row['task'],'suite':suite,'metric':metric,
                    'CCA_reference':reference,'OmniThink_calibrated':row['calibrated_target_scores'][metric],
                    'delta_OmniThink_minus_CCA':delta})
        for metric in metrics:
            values=[r['deltas'][metric] for r in rows]
            if not values:continue
            test={'suite':suite,'metric':metric,'n':len(values),
                'mean_CCA_reference':mean(r['reference_scores'][metric] for r in rows),
                'mean_OmniThink_calibrated':mean(r['calibrated_target_scores'][metric] for r in rows),
                'mean_delta':mean(values),'median_delta':median(values),
                'target_higher':sum(v>0 for v in values),'equal':sum(v==0 for v in values),'target_lower':sum(v<0 for v in values),
                'ci95_mean_delta':bootstrap_ci(values,iterations=10000,seed=17),
                'wilcoxon_two_sided_p':float(wilcoxon(values,zero_method='pratt',method='auto').pvalue) if any(values) else 1.0}
            tests.append(test)
        groups[suite]={'complete':len(rows),'planned':10}
    if len(tests)==9 and all(x['n']==10 for x in tests):
        for row,p in zip(tests,holm([x['wilcoxon_two_sided_p'] for x in tests])):
            row['holm9_p']=p
    elif tests:
        # Preserve the originally planned nine-test family, treating unavailable
        # comparisons as p=1 rather than reducing the multiplicity after failures.
        ps=[x['wilcoxon_two_sided_p'] for x in tests]+[1.0]*(9-len(tests))
        for row,p in zip(tests,holm(ps)):row['holm9_p']=p
    attempts=[ev.read(p) for p in root.glob('attempt_*/**/results/*.json')]
    usage={k:sum(u.get(k,0) or 0 for r in attempts for u in r.get('usage',[])) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    summary={'status':progress['phase'],'accepted':len(accepted),'planned':20,'groups':groups,
       'metrics':tests,'rating_attempts':len(attempts),'usage_including_failed_attempts':usage,
       'interpretation':'Task-specific CCA scores are fixed anchors. Target scores are reference-calibrated, not independently judged absolute benchmark scores. n is tasks, not nine metric scores treated as independent observations.'}
    ev.write(root/'summary.json',summary)
    with (root/'task_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]) if flat else ['task','metric','score'])
        writer.writeheader();writer.writerows(sorted(flat,key=lambda x:(x['task_index'],x['suite'],x['metric'])))
    with (root/'metric_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['suite','metric','n','mean_CCA_reference','mean_OmniThink_calibrated','mean_delta','wilcoxon_two_sided_p','holm9_p']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(tests)
    table=['# OmniThink 前10题：CCA逐题得分参照评估','','评审：官方 MiMo-V2.6-Flash。Reference 为每题原 CCA 输出和原 benchmark 得分，Target 为本次 OmniThink 保存材料。一般4项、探索5项；提示词沿用现有消融协议。','',
      f"有效评分 {len(accepted)}/20；一般 {groups['general']['complete']}/10，探索 {groups['exploration']['complete']}/10。",'',
      '| 指标 | n | CCA参照均分 | OmniThink校准均分 | 差值（OmniThink−CCA） | Holm校正p |',
      '|---|---:|---:|---:|---:|---:|']
    for t in tests:
        table.append(f"| {t['metric']} | {t['n']} | {t['mean_CCA_reference']:.3f} | {t['mean_OmniThink_calibrated']:.3f} | {t['mean_delta']:+.3f} | {t.get('holm9_p',1):.5g} |")
    table+=['','统计单位为任务；对每项的任务差值做双侧 Wilcoxon 检验，九项检验采用 Holm 校正。部分评分缺失时按该项有效任务汇总，并保留原定九项校正范围。','',
     '校准分由 CCA 原得分加评审返回差值计算；这是参照式补充比较，不是独立绝对评分，也不是组件消融。模型未被要求解释原因，本报告不替模型编造评分理由。','',
     '评估用量（含失败响应）：'+json.dumps(usage,ensure_ascii=False)]
    (root/'评分比较.md').write_text('\n'.join(table),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
