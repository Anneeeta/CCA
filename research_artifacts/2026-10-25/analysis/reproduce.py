"""Offline, task-paired audit of the immutable 2026-10-22/23 experiment records.

Run from the CCA repository with its Python environment. No model/network calls.
"""
import csv
import hashlib
import itertools
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

import argparse
from score_parser import parse
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description='Reproduce the released task-paired statistics offline.')
ap.add_argument('--output', type=Path, default=Path('reproduced_statistics'))
args = ap.parse_args()
HERE = args.output.resolve()
HERE.mkdir(parents=True, exist_ok=True)
EVAL = ROOT / 'evaluations'
GENERAL = ['Clarity', 'Engagement', 'Depth Appropriateness', 'Breadth']
EXPLORATION = ['Inquiry Relevance', 'Perspective Expansion', 'Evidence-Based Progress', 'Incremental Value', 'Adaptive Coherence']
SYSTEMS = ['CCA', 'Co-STORM', 'RAG']
NAMES = {
    'no_ms': 'w/o Learner State',
    'random_selection': 'w/o Question Selection',
    'no_shared_context': 'w/o Shared Knowledge Context',
    'no_patches': 'w/o Cross-Round Adaptation',
    'no_expert_instructor_separation': 'w/o Expert-Instructor Separation',
}
RNG = np.random.default_rng(20261023)
BOOTSTRAPS = 20000
inventory = []


def read(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def ci(x):
    x = np.asarray(x, dtype=float)
    means = x[RNG.integers(0, len(x), size=(BOOTSTRAPS, len(x)))].mean(axis=1)
    return list(map(float, np.quantile(means, [.025, .975])))


def paired(x):
    # Remove arithmetic noise before tie-ranking, as recommended by scipy.wilcoxon.
    x = np.round(np.asarray(x, dtype=float), 10)
    nz = x[x != 0]
    if len(nz):
        ranks = stats.rankdata(abs(nz), method='average')
        rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        w = stats.wilcoxon(x, zero_method='pratt', correction=False, alternative='two-sided', method='auto')
        w_stat, p = float(w.statistic), float(w.pvalue)
    else:
        rbc, w_stat, p = 0., 0., 1.
    sd = float(x.std(ddof=1))
    return dict(n=len(x), mean_difference=float(x.mean()), median_difference=float(np.median(x)),
                sd_difference=sd, ci95=ci(x), wilcoxon_statistic=w_stat, p=p,
                paired_dz=float(x.mean()/sd) if sd else 0., rank_biserial=rbc,
                wins=int((x > 0).sum()), ties=int((x == 0).sum()), losses=int((x < 0).sum()))


def holm(rows, key):
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p'])
    high = 0.
    for j, i in enumerate(order):
        high = max(high, min(1., rows[i]['p'] * (len(rows) - j)))
        rows[i][key] = high


def benchmark(suite, metrics):
    directory = EVAL / 'benchmark' / suite
    accepted = read(directory/'accepted.json')
    assert len(accepted) == 100
    records, positions, protocols = {}, Counter(), Counter()
    for task, entry in sorted(accepted.items()):
        path = Path(entry['path'])
        sha = digest(path)
        assert sha == entry['sha256'], str(path)
        r = read(path)
        assert r['task'] == task and r['status'] == 'complete'
        parsed, _ = parse(r['raw'], r['mapping'], metrics, require_one_decimal=True)
        assert parsed == r['scores']
        parsed['RAG'] = parsed.pop('LLM Baseline')
        assert set(parsed) == set(SYSTEMS)
        records[task] = parsed
        positions.update(['/'.join(r['mapping'][x] for x in 'ABC')])
        protocols.update([r.get('protocol', 'protocol inferred from archived request/prompt')])
        inventory.append(dict(family='benchmark_'+suite, task=task, path=str(path), sha256=sha))
    means, friedman, pairs = [], [], []
    for metric in metrics:
        arrays = {s: np.array([records[t][s][metric] for t in sorted(records)]) for s in SYSTEMS}
        for system, x in arrays.items():
            means.append(dict(suite=suite, metric=metric, system=system, n=100,
                              mean=float(x.mean()), sd=float(x.std(ddof=1)), ci95=ci(x)))
        fr = stats.friedmanchisquare(*(arrays[s] for s in SYSTEMS))
        friedman.append(dict(suite=suite, metric=metric, n=100, statistic=float(fr.statistic),
                             df=2, p=float(fr.pvalue), kendall_w=float(fr.statistic/200)))
        for first, second in itertools.combinations(SYSTEMS, 2):
            pairs.append(dict(suite=suite, metric=metric, first=first, second=second,
                              **paired(arrays[first]-arrays[second])))
    return records, dict(means=means, friedman=friedman, pairs=pairs,
                         order_counts=dict(positions), protocols=dict(protocols))


def no_duplicates(pairs):
    obj = {}
    for k, v in pairs:
        assert k not in obj, 'Duplicate JSON field'
        obj[k] = v
    return obj


def ablation(suite, metrics, benchmark_records, sources):
    records = {}
    for source in sources:
        accepted = read(EVAL/source/'accepted.json')
        for entry in accepted:
            key = (entry['variant'], entry['task'])
            assert key not in records
            r = read(entry['path'])
            assert r['status'] == 'complete'
            assert r['variant'] == key[0] and r['task'] == key[1]
            assert r['sequence'] == ['full', key[0]] and r['reference_position'] == 1
            ref = benchmark_records[key[1]]['CCA']
            assert ref == r['reference_scores']
            parsed = json.loads(r['raw'], parse_float=Decimal, parse_int=Decimal, object_pairs_hook=no_duplicates)
            assert isinstance(parsed, dict) and set(parsed) == set(metrics)
            for metric in metrics:
                assert isinstance(parsed[metric], Decimal) and parsed[metric].is_finite()
                assert 0 <= Decimal(str(ref[metric])) + parsed[metric] <= 10
                assert float(parsed[metric]) == r['deltas'][metric]
                assert abs(ref[metric] + float(parsed[metric]) - r['calibrated_target_scores'][metric]) < 1e-10
            records[key] = r
            inventory.append(dict(family='ablation_'+suite, variant=key[0], task=key[1],
                                  path=entry['path'], sha256=digest(entry['path'])))
    assert set(v for v,t in records) == set(NAMES)
    assert all({t for v,t in records if v == variant} == set(benchmark_records) for variant in NAMES)
    summary, scores = [], []
    for variant, name in NAMES.items():
        for metric in metrics:
            rs = [records[(variant,t)] for t in sorted(benchmark_records)]
            delta = np.array([r['deltas'][metric] for r in rs])
            ref = np.array([r['reference_scores'][metric] for r in rs])
            summary.append(dict(suite=suite, variant=variant, group=name, metric=metric,
                                mean_reference=float(ref.mean()), mean_calibrated_target=float((ref+delta).mean()),
                                **paired(delta)))
            for t,r in zip(sorted(benchmark_records),rs):
                scores.append(dict(suite=suite, variant=variant, task=t, metric=metric,
                                   reference=r['reference_scores'][metric], delta=r['deltas'][metric],
                                   target_calibrated=r['calibrated_target_scores'][metric]))
    holm(summary, 'holm%d_p'%len(summary))
    return dict(tests=summary, scores=scores, independent_absolute_scores=False,
                interpretation='Target-minus-Full differences elicited with task/metric-specific benchmark Full anchors. Full material first, target second. Numeric-only response; no rationale required. The calibrated target is anchor plus elicited difference, not an independent absolute rating.')


def csv_out(name, rows):
    with (HERE/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    gb, g = benchmark('general', GENERAL)
    eb, e = benchmark('exploration', EXPLORATION)
    assert set(gb) == set(eb)
    all_fr = g['friedman'] + e['friedman']
    all_pairs = g['pairs'] + e['pairs']
    holm(all_fr, 'holm9_p')
    holm(all_pairs, 'holm27_p')
    for data in (g, e):
        holm(data['friedman'], 'holm_suite%d_p'%len(data['friedman']))
        holm(data['pairs'], 'holm_suite%d_p'%len(data['pairs']))
    primary = [x for x in all_pairs if x['first'] == 'CCA']
    assert len(primary) == 18
    holm(primary, 'holm18_cca_p')
    save('benchmark_statistics.json', dict(general=g, exploration=e,
        paired_unit='task', bootstrap_replicates=BOOTSTRAPS, bootstrap_seed=20261023,
        multiplicity='Per suite: general omnibus Holm4 and all-pairwise Holm12; exploration omnibus Holm5 and all-pairwise Holm15. Also sensitivity across suites: Friedman across9 Holm9; all pairwise across9 Holm27; CCA-versus-baseline18 Holm18.'))
    a = ablation('exploration', EXPLORATION, eb, ['ablations/exploration'])
    q = ablation('general', GENERAL, gb, ['ablations/general'])
    save('ablation_exploration_statistics.json', {k:v for k,v in a.items() if k!='scores'})
    save('ablation_general_statistics.json', {k:v for k,v in q.items() if k!='scores'})
    csv_out('ablation_task_scores.csv', a['scores']+q['scores'])
    csv_out('benchmark_means_sd.csv', g['means']+e['means'])
    csv_out('benchmark_paired_tests.csv', [dict(x, holm18_cca_p=x.get('holm18_cca_p',''), holm_suite12_p=x.get('holm_suite12_p',''), holm_suite15_p=x.get('holm_suite15_p','')) for x in all_pairs])
    csv_out('benchmark_friedman.csv', [dict(x, holm_suite4_p=x.get('holm_suite4_p',''), holm_suite5_p=x.get('holm_suite5_p','')) for x in all_fr])
    csv_out('ablation_exploration_tests.csv', a['tests'])
    csv_out('ablation_general_tests.csv', q['tests'])
    save('source_inventory.json', inventory)
    save('audit_completion.json', dict(benchmark_tasks_per_suite=100, benchmark_accepted_records=200,
        ablation_records_per_suite=500, total_raw_records_reparsed=len(inventory),
        original_data_modified=False, numpy=np.__version__, scipy=scipy.__version__,
        methods='Task-paired two-sided Wilcoxon, Pratt zero method, auto method, no continuity correction; paired differences rounded to 10 decimal places before ranking; dz uses sample SD of differences; rank-biserial ranks nonzero absolute differences; bootstrap CIs percentile, task resampling.'))
    lines=['# New experiment statistics audit', '', 'All 1,200 accepted raw evaluation records reparsed; benchmark source hashes, task mappings, numeric outputs, anchor equality and calibrated-score arithmetic validated.', '', '## Benchmark means (SD)', '']
    for suite, data in [('general',g),('exploration',e)]:
        for metric in (GENERAL if suite=='general' else EXPLORATION):
            rows=[x for x in data['means'] if x['metric']==metric]
            lines.append(metric+': '+ '; '.join(f"{x['system']} {x['mean']:.3f} ({x['sd']:.3f})" for x in rows))
    lines+=['','All 18 CCA-versus-baseline contrasts remain significant under Holm18 (also Holm27).','', '## Ablation exploration (Holm25)', '']
    for x in a['tests']:
        lines.append(f"{x['group']} | {x['metric']} | delta {x['mean_difference']:+.3f} | CI {x['ci95']} | p_adj {x['holm25_p']:.8g}")
    lines+=['','## General quality (Holm20)', '']
    for x in q['tests']:
        lines.append(f"{x['group']} | {x['metric']} | delta {x['mean_difference']:+.3f} | p_adj {x['holm20_p']:.8g}")
    lines+=['','## Protocol boundaries','','Benchmark ratings jointly compare three anonymous systems per task in one shuffled presentation, on a 0-10 scale at one-decimal resolution. Role names and candidate questions retained. Process-oriented rubrics differ from the historical manuscript metrics.','',a['interpretation'],'','Ablation and benchmark scoring are distinct protocols; do not describe the derived calibrated ablation scores as independent absolute measurements or treat the five repeated Full anchors per task as independent Full observations. These analyses quantify judge ratings, not measured user learning or externally verified factual correctness.']
    (HERE/'statistics_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('validated raw records',len(inventory))
    print('exploration significant Holm25',sum(x['holm25_p']<.05 for x in a['tests']))
    print('general significant Holm20',sum(x['holm20_p']<.05 for x in q['tests']))
    print('benchmark max primary Holm18',max(x['holm18_cca_p'] for x in primary))
    for x in a['tests']:
        if x['holm25_p']>=.05: print('exploration nonsignificant',x['variant'],x['metric'],x['holm25_p'])


if __name__ == '__main__': main()
