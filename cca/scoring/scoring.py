from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from ..schemas import CandidateQuestion, CandidateScore
from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_candidates(
    candidates: Sequence[CandidateQuestion],
    ms_entries: Sequence[str],
    guidance: Optional[str],
    history_questions: Sequence[str],
    provider: EmbeddingProvider,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int,
) -> List[CandidateScore]:
    """
    Compute I(Q) for each candidate.
    """
    scores: List[CandidateScore] = []
    if not provider.available:
        logger.warning("Embedding provider unavailable; returning zero scores.")
        for c in candidates:
            scores.append(CandidateScore(question=c.question, s=0.0, r=0.0, novel=1.0, I=gamma))
        return scores

    question_embs = provider.embed([c.question for c in candidates])
    ms_embs = provider.embed(ms_entries) if ms_entries else np.zeros((1, question_embs.shape[1]))
    guidance_emb = provider.embed_single(guidance) if guidance else None
    history_embs = provider.embed(history_questions) if history_questions else np.zeros(
        (0, question_embs.shape[1])
    )

    for idx, candidate in enumerate(candidates):
        q_emb = question_embs[idx]
        # s(Q): average of top-K similarities with Ms
        sims = [_cosine(q_emb, m_emb) for m_emb in ms_embs] if ms_entries else [0.0]
        sims_sorted = sorted(sims, reverse=True)
        topk = sims_sorted[: max(1, min(top_k, len(sims_sorted)))]
        s_q = float(np.mean(topk)) if topk else 0.0
        # r(Q)
        r_q = _cosine(q_emb, guidance_emb) if guidance_emb is not None else 0.0
        # Novelty
        if history_embs.shape[0] == 0:
            novel = 1.0
        else:
            hist_sims = [_cosine(q_emb, h_emb) for h_emb in history_embs]
            novel = 1.0 - max(hist_sims)
        novel = max(0.0, min(1.0, novel))
        I_q = alpha * s_q + beta * r_q + gamma * novel
        scores.append(CandidateScore(question=candidate.question, s=s_q, r=r_q, novel=novel, I=I_q))
    return scores


def select_question(scores: List[CandidateScore]) -> CandidateScore:
    return sorted(scores, key=lambda s: s.I, reverse=True)[0]


def select_top_two(
    scores: List[CandidateScore],
    provider: Optional[EmbeddingProvider] = None,
) -> tuple[CandidateScore, Optional[CandidateScore]]:
    """选两个得分最高且来自不同子话题的问题。"""
    ranked = sorted(scores, key=lambda s: s.I, reverse=True)
    first = ranked[0]
    if len(ranked) < 2:
        return first, None

    # 有 embedding 时用 cosine 相似度排除同子话题候选
    if provider is not None and provider.available:
        first_emb = provider.embed_single(first.question)
        rest = ranked[1:]
        rest_embs = provider.embed([s.question for s in rest])
        filtered = []
        for i, s in enumerate(rest):
            sim = _cosine(first_emb, rest_embs[i])
            if sim < 0.7:  # 相似度 < 0.7 视为不同子话题
                filtered.append(s)
        if filtered:
            return first, filtered[0]

    # 无 embedding 或全部同子话题时，直接取第二高分
    return first, ranked[1]
