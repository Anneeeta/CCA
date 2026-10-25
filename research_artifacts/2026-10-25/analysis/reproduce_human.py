"""Reproduce paired human-rating tests offline from de-identified sessions."""
import json,csv,math,random,statistics,argparse
from pathlib import Path

def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)

def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks

def exact_wilcoxon(differences: list[float]) -> tuple[float, float, float, int]:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 0.0, 0.0, 1.0, 0

    ranks = average_ranks([abs(value) for value in nonzero])
    w_plus = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    total = sum(ranks)
    center = total / 2.0
    observed_distance = abs(w_plus - center)

    extreme = 0
    permutations = 1 << len(ranks)
    for mask in range(permutations):
        candidate = sum(rank for index, rank in enumerate(ranks) if mask & (1 << index))
        if abs(candidate - center) >= observed_distance - 1e-12:
            extreme += 1

    p_value = extreme / permutations
    w_minus = total - w_plus
    rank_biserial = (w_plus - w_minus) / total
    return w_plus, rank_biserial, p_value, len(nonzero)

def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted

def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3e-14
    fp_min = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fp_min:
        d = fp_min
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + coefficient * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + coefficient / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        result *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + coefficient * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + coefficient / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise RuntimeError("incomplete beta continued fraction did not converge")

def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b

def paired_t_test(differences: list[float]) -> tuple[float, int, float]:
    sample_size = len(differences)
    degrees_freedom = sample_size - 1
    mean_difference = statistics.mean(differences)
    sd_difference = statistics.stdev(differences)
    if sd_difference == 0:
        return (0.0, degrees_freedom, 1.0) if mean_difference == 0 else (
            math.copysign(math.inf, mean_difference),
            degrees_freedom,
            0.0,
        )
    t_statistic = mean_difference / (sd_difference / math.sqrt(sample_size))
    x = degrees_freedom / (degrees_freedom + t_statistic * t_statistic)
    two_sided_p = regularized_incomplete_beta(degrees_freedom / 2.0, 0.5, x)
    return t_statistic, degrees_freedom, two_sided_p

def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction

def bootstrap_mean_ci(
    differences: list[float], *, seed: int, resamples: int = 50_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(differences)
    samples = []
    for _ in range(resamples):
        samples.append(sum(differences[rng.randrange(size)] for _ in range(size)) / size)
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)

METRICS = ['Aha Moment', 'UU Discovery', 'Guidance Ease & Question Insightfulness', 'Information Gain', 'Trajectory Coherence', 'Clarity', 'Engagement', 'Depth Appropriateness', 'Breadth']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('reproduced_human'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    cases=json.loads((root/'human/paired_sessions.json').read_text(encoding='utf-8'))
    rows=[];counter=0
    for baseline in ['RAG robot','Co-STORM']:
        paired=[x for x in cases if x['baseline']==baseline]
        assert len(paired)==20
        for metric in METRICS:
            differences=[x['ratings'][metric]['CCA']-x['ratings'][metric][baseline] for x in paired]
            counter+=1
            w,rbc,p,nz=exact_wilcoxon(differences)
            t,df,tp=paired_t_test(differences)
            low,high=bootstrap_mean_ci(differences,seed=20260727+counter)
            rows.append(dict(comparison='CCA vs '+baseline,metric=metric,n_pairs=20,n_nonzero=nz,
                mean_difference=statistics.mean(differences),bootstrap_CI_low=low,bootstrap_CI_high=high,
                wilcoxon_W_plus=w,exact_p=p,rank_biserial_r=rbc,paired_t=t,paired_t_df=df,paired_t_p=tp))
    for key,out in [('exact_p','holm_p'),('paired_t_p','paired_t_holm_p')]:
        for row,p in zip(rows,holm_adjust([r[key] for r in rows])):row[out]=p
    with (args.output/'paired_tests.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Reproduced 18 human-rating contrasts from 40 paired sessions; collected 2-10 scale.')

if __name__=='__main__': main()
