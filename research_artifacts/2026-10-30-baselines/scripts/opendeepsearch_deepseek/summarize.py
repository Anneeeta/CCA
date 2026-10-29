"""Summarize paired three-system, fixed-CCA-reference assessments."""
import csv
from pathlib import Path
from statistics import mean
from decimal import Decimal
from scipy.stats import wilcoxon
from evaluate import BASE,OMNI,ev
from cca.eval_analysis import holm

def load(root):
    scores={}
    for item in ev.read(root/'accepted.json'):
        path=Path(item['path'])
        if ev.sha(path)!=item['sha256']:raise ValueError('Changed score record')
        r=ev.read(path)
        if ev.parse_delta(r['raw'],r['reference_scores'])!=r['deltas']:raise ValueError('Score parse mismatch')
        scores[(item['suite'],item['task'])]=r
    return scores

def main():
    odsroot=BASE/'mimo_reference_evaluation_001'; ods=load(odsroot); omni=load(OMNI)
    if set(ods)!=set(omni) or len(ods)!=20:raise ValueError('Ratings or matched tasks missing')
    rows=[]; tests=[]
    for suite,metrics in ev.METRICS.items():
        names=sorted(task for s,task in ods if s==suite)
        if len(names)!=10:raise ValueError('Expected ten tasks per suite')
        for metric in metrics:
            c=[];o=[];d=[]; odelta=[]; ddelta=[]
            for task in names:
                x,y=ods[(suite,task)],omni[(suite,task)]
                if x['reference_scores'][metric]!=y['reference_scores'][metric]:raise ValueError('Different task anchors')
                a=x['reference_scores'][metric];b=y['calibrated_target_scores'][metric];z=x['calibrated_target_scores'][metric]
                c.append(a);o.append(b);d.append(z)
                odelta.append(y['deltas'][metric]); ddelta.append(x['deltas'][metric])
                rows.append({'task':task,'suite':suite,'metric':metric,'CCA_reference':a,'OmniThink_calibrated':b,'OpenDeepSearch_calibrated':z})
            def p(values):return float(wilcoxon(values,zero_method='pratt',method='auto').pvalue) if any(values) else 1.0
            direct=[float(Decimal(str(a))-Decimal(str(b))) for a,b in zip(ddelta,odelta)]
            tests.append({'suite':suite,'metric':metric,'n':len(c),'CCA':mean(c),'OmniThink':mean(o),'OpenDeepSearch':mean(d),'omni_minus_CCA':mean(odelta),'ODS_minus_CCA':mean(ddelta),'p_omni':p(odelta),'p_ODS':p(ddelta),'p_ODS_vs_omni_exploratory':p(direct)})
    adjusted=holm([p for row in tests for p in (row['p_omni'],row['p_ODS'])])
    extra=holm([row['p_ODS_vs_omni_exploratory'] for row in tests])
    for i,row in enumerate(tests):
        row['p_omni_holm18']=adjusted[2*i];row['p_ODS_holm18']=adjusted[2*i+1];row['p_ODS_vs_omni_holm9_exploratory']=extra[i]
    with (odsroot/'task_scores_3systems.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    ev.write(odsroot/'summary_3systems.json',{'metrics':tests,'n_tasks':10,'method':'Fixed CCA reference with two separate target evaluations. Reused OmniThink scores from evaluation_003; ODS new evaluation_001. Not a simultaneous three-way judge.','multiple_testing':'18 target-vs-CCA tests Holm jointly; ODS-vs-Omni 9 exploratory tests Holm separately.'})
    statuses=[ev.read(BASE/f'{i:03d}'/'status.json') for i in range(1,11)]
    toks={key:sum(s.get('usage',{}).get(key,0) for s in statuses) for key in ['prompt_tokens','completion_tokens','total_tokens']}
    counts={key:sum(s.get('counts',{}).get(key,0) for s in statuses) for key in ['model_requests','search_requests','fetch_requests','fetch_unavailable','step_limit_finalizations']}
    eval_attempts=[ev.read(p) for p in odsroot.glob('attempt_*/**/results/*.json')]
    eval_usage={k:sum(u.get(k,0) or 0 for r in eval_attempts for u in r.get('usage',[])) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    table=['# 三系统前10题参照式评估','','CCA 为每题既有 benchmark 得分，OmniThink 复用已完成的参照式评估，OpenDeepSearch 使用相同的指标和提示词新评估。两个目标分别与 CCA 比较；不是三系统同时匿名评分。','', '| 指标 | CCA参照均分 | OmniThink校准均分 | OpenDeepSearch校准均分 | Omni–CCA | ODS–CCA | ODS–CCA Holm p |','|---|---:|---:|---:|---:|---:|---:|']
    for r in tests:table.append(f"| {r['metric']} | {r['CCA']:.2f} | {r['OmniThink']:.2f} | {r['OpenDeepSearch']:.2f} | {r['omni_minus_CCA']:+.2f} | {r['ODS_minus_CCA']:+.2f} | {r['p_ODS_holm18']:.4f} |")
    table+=['','双侧配对 Wilcoxon：两个目标相对 CCA 的18项比较共同进行 Holm 校正。ODS 对 Omni 的9项探索性比较单独校正，详见 JSON。每项 n=10；评分相对于固定参考锚点，并非独立绝对分数。','',f'OpenDeepSearch 生成用量：{toks}；调用统计：{counts}。',f'本次新增评估用量（含失败重试）：{eval_usage}。','', 'CCA 与 OmniThink 使用此前记录；不同工作流、运行日期和提示词，token 不是预算匹配结论。网页读取可能失败，搜索摘要仍被记录；不得视作均成功阅读网页正文。']
    (odsroot/'三系统评分比较.md').write_text('\n'.join(table),encoding='utf-8')
    print('\n'.join(table))

if __name__=='__main__':main()
