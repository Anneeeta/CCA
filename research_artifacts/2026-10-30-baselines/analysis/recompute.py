"""Reproduce the 27 Group 2 comparisons locally; makes no model/search calls."""
from pathlib import Path
from statistics import mean,stdev
import csv,json,math
from scipy.stats import wilcoxon

B=Path(__file__).parent
with (B/'scores.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
original=json.loads((B/'summary_original.json').read_text(encoding='utf8'))
def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]);out=[0.0]*len(ps);current=0.0
    for rank,i in enumerate(order):
        current=max(current,(len(ps)-rank)*ps[i]);out[i]=min(1.0,current)
    return out

result=[]
for m in original['metrics']:
    selected=sorted((r for r in rows if r['suite']==m['suite'] and r['metric']==m['metric']),key=lambda r:r['task'])
    assert len(selected)==100 and len({r['task'] for r in selected})==100
    vals={s:[float(r[s]) for r in selected] for s in ['CCA','OmniThink','OpenDeepSearch']}
    row={'suite':m['suite'],'metric':m['metric'],'n':100}
    for s,v in vals.items():row[s]=mean(v);row[s+'_sd']=stdev(v)
    for a,b,k in [('CCA','OmniThink','cca_omni'),('CCA','OpenDeepSearch','cca_ods'),('OmniThink','OpenDeepSearch','omni_ods')]:
        d=[round(x-y,1) for x,y in zip(vals[a],vals[b])]
        row['p_'+k]=float(wilcoxon(d,zero_method='pratt',method='auto').pvalue) if any(d) else 1.0
    result.append(row)
adjusted=holm([r['p_'+k] for r in result for k in ['cca_omni','cca_ods','omni_ods']])
for i,r in enumerate(result):
    for j,k in enumerate(['cca_omni','cca_ods','omni_ods']):r['holm27_'+k]=adjusted[3*i+j]
    source=next(x for x in original['metrics'] if x['suite']==r['suite'] and x['metric']==r['metric'])
    for k,v in source.items():
        if k in r and isinstance(v,(int,float)):assert math.isclose(v,r[k],rel_tol=1e-10,abs_tol=1e-12),(k,v,r[k])
(B/'summary_reproduced.json').write_text(json.dumps({'n':100,'tests':27,'verified_against_original':True,'metrics':result},indent=2),encoding='utf8')
print('PASS: 100 paired tasks, 9 criteria, 27 Holm-adjusted comparisons match the original summary.')
