"""Parse explicit score sections only; never mine other candidates' explanation prose."""
import re, math
METRICS=['Clarity','Engagement','Depth Appropriateness','Breadth']
def parse(raw,mapping,metrics=None,require_one_decimal=False):
    metrics=METRICS if metrics is None else tuple(metrics)
    if not metrics or len(set(metrics))!=len(metrics):raise ValueError("Invalid metric set")
    lines=[]
    for line in raw.splitlines():
        clean=re.sub(r'^[ \t]*#{1,6}\s*','',line).replace('*','').replace('`','').strip()
        if re.match(r'^(?:Final\s+Ranking|Explanation)\b',clean,re.I):break
        lines.append(clean)
    text='\n'.join(lines)
    pattern=re.compile(r'^System\s+([ABC])(?:\s*\([^\n)]*\))?\s*:?[ \t]*$',re.M|re.I)
    heads=list(pattern.finditer(text));result={};spans=[]
    for i,h in enumerate(heads):
        letter=h[1].upper()
        if letter not in mapping:raise ValueError('Unknown candidate')
        if letter in result:raise ValueError('Duplicate candidate score section')
        block=text[h.end():heads[i+1].start() if i+1<len(heads) else len(text)]
        scores={}
        for metric in metrics:
            p=r'^'+re.escape(metric)+r'\s*(?::|\()?\s*(\d+(?:\.\d+)?)\s*/\s*10'
            found=list(re.finditer(p,block,re.M|re.I))
            if require_one_decimal and any(not re.fullmatch(r'\d+\.\d', x[1]) for x in found):raise ValueError('Score must have exactly one decimal place: '+metric)
            values={float(x[1]) for x in found}
            if len(values)!=1:raise ValueError('Missing or conflicting explicit metric score: '+metric)
            score=next(iter(values))
            if not math.isfinite(score) or not 0<=score<=10:raise ValueError('Invalid score')
            scores[metric]=score
            spans.append({'label':letter,'metric':metric,'score':score,'score_line':found[0][0]})
        result[letter]=scores
    if set(result)!=set(mapping):raise ValueError('Incomplete candidate score sections')
    return {mapping[k]:v for k,v in result.items()},spans
