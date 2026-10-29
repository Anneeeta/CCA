"""Reproduce the post-hoc Group 2 Friedman tests; no model calls."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import scipy
from scipy.stats import friedmanchisquare

base = Path(__file__).resolve().parent
source = base/'scores.csv'
groups = defaultdict(list)
with source.open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        groups[(row['suite'], row['metric'])].append(row)
assert len(groups) == 9
results = []
for (suite, metric), rows in groups.items():
    rows.sort(key=lambda r: r['task'])
    assert len(rows) == 100 and len({r['task'] for r in rows}) == 100
    values = [[float(r[system]) for r in rows] for system in ('CCA', 'OmniThink', 'OpenDeepSearch')]
    statistic, p = friedmanchisquare(*values)
    results.append({'suite': suite, 'metric': metric, 'n': 100,
                    'statistic': float(statistic), 'p': float(p),
                    'kendall_w': float(statistic/200)})
order = sorted(range(len(results)), key=lambda i: results[i]['p'])
previous = 0.0
for rank, index in enumerate(order):
    previous = max(previous, min(1.0, (len(results)-rank)*results[index]['p']))
    results[index]['holm9_p'] = previous
out = {'analysis': 'post-hoc group-2 omnibus tests for harmonized reporting',
       'method': 'Friedman chi-square with tie correction; Holm over nine metrics',
       'n_tasks': 100, 'scipy_version': scipy.__version__,
       'scores_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
       'metrics': results}
(base/'omnibus_holm9.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
print('PASS: nine Group 2 omnibus tests, Holm-adjusted; significant:',
      sum(x['holm9_p'] < .05 for x in results))
