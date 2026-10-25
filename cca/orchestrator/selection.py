"""Question selection controls with local, scheduling-independent randomness."""
import hashlib
import random

from ..config import OrchestratorConfig
from ..schemas import CandidateQuestion, CandidateScore
from ..scoring.scoring import select_top_two


def fixed_topic_questions(topic: str, guidance: str | None, count: int) -> list[CandidateQuestion]:
    focus = f"Topic: {topic}." + (f" Current task: {guidance}." if guidance else "")
    prompts = [f"{focus} Explain the key concepts, mechanisms and practical uses.",
               f"{focus} What evidence, assumptions, limitations and failure cases should be considered?"]
    return [CandidateQuestion(question=question, type="known_unknown", rationale="Fixed, non-adaptive topic prompt.") for question in prompts[:count]]


def choose_questions(scores: list[CandidateScore], options: OrchestratorConfig, provider,
                     topic: str, round_id: int) -> tuple[CandidateScore, CandidateScore | None]:
    if not scores:
        raise ValueError("No scored candidate questions")
    count = min(options.questions_per_round, len(scores))
    if options.selection_policy == "random":
        identity = f"{options.selection_seed}\0{topic}\0{round_id}".encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(identity).digest(), "big"))
        selected = rng.sample(scores, count)
    elif count == 2 and options.selection_diversity:
        return select_top_two(scores, provider)
    else:
        selected = sorted(scores, key=lambda score: score.I, reverse=True)[:count]
    return selected[0], selected[1] if len(selected) > 1 else None
