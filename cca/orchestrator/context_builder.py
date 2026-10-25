from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from ..config import OrchestratorConfig
from ..schemas import RoundSummary
from .summaries import prepare_summary, render_summary, source_urls


@dataclass
class DialogueTurn:
    question: str
    instructor_answer: str
    expert_viewpoint: Optional[str] = None
    round_id: Optional[int] = None
    summary: Optional[RoundSummary] = None
    guidance: Optional[str] = None
    selected_questions: Optional[List[str]] = None


class ContextBuilder:
    """
    Builds strict, readable contexts for each agent while enforcing access rules.
    """

    def __init__(self, options: OrchestratorConfig | None = None):
        self.options = options or OrchestratorConfig()

    def _format_history(self, history: List[DialogueTurn], max_chars: int | None = None) -> str:
        if not history:
            return "(no prior rounds)"
        if self.options.compact_context and self.options.history_mode == "summary":
            budget = max_chars or self.options.history_context_max_chars
            # Keep every round represented rather than dropping earlier rounds.
            per_round = max(1, (budget - len(history)) // len(history))
            summaries = []
            for index, turn in enumerate(history, 1):
                summary = turn.summary or prepare_summary(
                    None, round_id=turn.round_id if turn.round_id is not None else index,
                    question=turn.question, answer=turn.instructor_answer,
                    expert=turn.expert_viewpoint,
                    guidance=turn.guidance,
                    evidence_refs=source_urls(turn.expert_viewpoint or ""),
                    max_chars=self.options.round_summary_max_chars,
                )
                summaries.append(render_summary(summary, per_round))
            return "\n".join(summaries)
        lines = []
        for idx, h in enumerate(history, start=1):
            actual_round = h.round_id if h.round_id is not None else idx
            if h.guidance:
                lines.append(f"Round {actual_round}: User guidance: {h.guidance}")
            lines.append(f"Round {actual_round}: Q: {h.question}")
            if h.expert_viewpoint:
                lines.append(f"  Expert: {h.expert_viewpoint}")
            lines.append(f"  Instructor: {h.instructor_answer}")
        return "\n".join(lines)

    @staticmethod
    def get_round_details(history: List[DialogueTurn], round_id: int) -> DialogueTurn:
        """Local, explicit access to the retained full turn; no provider request."""
        for index, turn in enumerate(history, 1):
            if (turn.round_id if turn.round_id is not None else index) == round_id:
                return turn
        raise KeyError(f"No retained round {round_id}")

    # Student contexts ------------------------------------------------------

    def build_student_candidate_messages(
        self,
        topic: str,
        ms_text: Optional[str],
        history: List[DialogueTurn],
        guidance: Optional[str],
        num_candidates: int,
    ) -> List[dict]:
        history_text = self._format_history(history)
        drift_rule = (
            "Hard rule: if Ms/history contains drifted content, ignore off-topic parts.\n"
            if ms_text is not None
            else "Hard rule: if history contains drifted content, ignore off-topic parts.\n"
        )
        ms_section = (
            "=== Student Knowledge Pool Ms (read/write) ===\n" + ms_text + "\n\n"
            if ms_text is not None else ""
        )
        content = (
            f"Topic: {topic}\n"
            f"Guidance (optional): {guidance or 'None'}\n"
            "Your task: generate diverse candidate questions for the next discussion turn.\n"
            "Hard rule: keep all candidates tightly anchored to Topic and direct subtopics.\n"
            f"{drift_rule}"
            "Hard rule: every candidate must clearly reference Topic-specific concepts, not generic adjacent areas.\n"
            "Hard rule: do not pivot to sibling frameworks unless guidance explicitly requests comparison.\n"
            "Prefer candidate breadth across multiple subtopics, not one narrow angle only.\n"
            f"Number of candidates: {num_candidates}\n"
            "Context history:\n"
            f"{history_text}\n\n"
            f"{ms_section}"
            "Return JSON: {\"candidates\": [{\"question\": \"...\", \"rationale\": \"...\"}, ...]}"
        )
        return [{"role": "user", "content": content}]

    def build_student_update_messages(
        self,
        answer_text: str,
        ms_text: str,
        round_id: int,
    ) -> List[dict]:
        content = (
            f"Round {round_id} instructor answer (A^(ins)):\n{answer_text}\n\n"
            "Update your knowledge pool Ms. Keep unknown_unknown entries as questions/hypotheses only.\n"
            "Return ONLY new or changed entries. Do not return unchanged old entries.\n"
            "Reuse the supplied id when revising an entry or moving it to another bucket; "
            "use id=null for genuinely new entries. Entries omitted from your response are retained.\n"
            "Use remove_ids only for explicitly superseded entries. Never remove an item merely because it is not shown.\n"
            "Current Ms context view (may omit other stored entries):\n"
            f"{ms_text}\n\n"
            "Return JSON with lists for known, known_unknown, conflict, unknown_unknown, and remove_ids."
        )
        return [{"role": "user", "content": content}]

    # Instructor contexts ---------------------------------------------------

    def build_instructor_route_messages(
        self,
        topic: str,
        shared_k_text: str,
        question: str,
        history: List[DialogueTurn],
    ) -> List[dict]:
        history_text = self._format_history(history, self.options.helper_context_max_chars) if self.options.compact_context else self._format_history(history)
        knowledge_section = "" if self.options.compact_context else "=== Shared Knowledge Pool K (read-only) ===\n" + shared_k_text + "\n\n"
        content = (
            f"Topic: {topic}\n"
            f"Student question: {question}\n"
            f"History:\n{history_text}\n\n"
            f"{knowledge_section}"
            "Task: choose one concise domain label for expert routing.\n"
            "Output ONLY one short English domain string. No JSON. No extra text."
        )
        return [{"role": "user", "content": content}]

    def build_instructor_teach_messages(
        self,
        topic: str,
        shared_k_text: str,
        question: str,
        history: List[DialogueTurn],
        expert_viewpoint: Optional[str] = None,
    ) -> List[dict]:
        history_text = self._format_history(history)
        content = (
            f"Topic: {topic}\n"
            f"Student question: {question}\n"
            f"History:\n{history_text}\n\n"
            "=== Shared Knowledge Pool K (read-only) ===\n"
            f"{shared_k_text}\n\n"
            "=== Reference Material (from expert research) ===\n"
            f"{expert_viewpoint or 'None'}\n\n"
            "Task: Based on the knowledge pool and reference material above, provide a thorough, well-structured teaching answer in English.\n"
            "Cover the key concepts, mechanisms, concrete examples, and practical takeaways.\n"
            "Hard rule: keep answer anchored to Topic; if question drifts, steer back to Topic.\n"
            "Hard rule: NEVER start with meta-commentary like 'We need to answer', 'Let me address', or planning text. Start directly with the teaching content.\n"
            "Hard rule: NEVER repeat or rephrase the question as a heading or title. Do NOT start with '**Question 1:**' or similar. Begin your answer immediately with the content.\n"
            "Hard rule: If multiple questions are provided, answer EACH question fully and separately. Do NOT repeat the same content for different questions.\n"
            "After answering, add a short extension in 2-3 sentences about one closely related concept.\n"
            "Prefix the mini-extension with: Extension:\n"
            "Output natural language only. No JSON. No code fences."
        )
        return [{"role": "user", "content": content}]

    def build_instructor_messages(
        self,
        topic: str,
        shared_k_text: str,
        question: str,
        history: List[DialogueTurn],
        expert_viewpoint: Optional[str] = None,
    ) -> List[dict]:
        history_text = self._format_history(history)
        content = (
            f"Topic: {topic}\n"
            f"Student question: {question}\n"
            f"History:\n{history_text}\n\n"
            "=== Shared Knowledge Pool K (read-only) ===\n"
            f"{shared_k_text}\n\n"
            "If expert_viewpoint is provided, rewrite it into teaching language.\n"
            "Do NOT copy expert wording or headings; produce a pedagogical rewrite.\n"
            f"Expert viewpoint: {expert_viewpoint or 'None'}\n"
            "Return JSON with fields domain, intent (optional), answer."
        )
        return [{"role": "user", "content": content}]

    # Expert contexts -------------------------------------------------------

    def build_expert_query_messages(self, topic: str, question: str, domain: str,
                                    history: List[DialogueTurn]) -> List[dict]:
        return [{"role": "user", "content": (
            f"Topic: {topic}\nDomain: {domain}\nCurrent question: {question}\n"
            "Relevant dialogue notes:\n" + self._format_history(history, self.options.helper_context_max_chars)
            + "\nGenerate search terms for this question; do not answer it."
        )}]

    def build_expert_messages(
        self,
        topic: str,
        shared_k_text: str,
        question: str,
        domain: str,
        history: List[DialogueTurn],
        retrieved_context: Optional[str] = None,
    ) -> List[dict]:
        history_text = self._format_history(history)
        retrieval_section = ""
        if retrieved_context:
            retrieval_section = (
                "\n\n=== Retrieved Search Results (ground your answer in these) ===\n"
                f"{retrieved_context}\n"
            )
        content = (
            f"Topic: {topic}\n"
            f"Domain (from instructor): {domain}\n"
            f"Student question: {question}\n"
            f"History:\n{history_text}\n\n"
            "=== Shared Knowledge Pool K (read-only) ===\n"
            f"{shared_k_text}\n\n"
            f"{retrieval_section}"
            "Use the search results as a foundation, then supplement with your own knowledge "
            "to provide comprehensive coverage of the topic. Cover multiple sub-aspects, "
            "include concrete examples, mechanisms, and practical details.\n"
            "Hard rule: keep viewpoint strictly within Topic scope.\n"
            "Hard rule: do not switch the main subject to sibling frameworks unless comparison is explicitly requested.\n"
            "Return JSON: {\"viewpoint\": \"...\"}"
        )
        return [{"role": "user", "content": content}]

    # Arbiter contexts ------------------------------------------------------

    def build_arbiter_k_messages(
        self,
        topic: str,
        shared_k_text: str,
        ms_text: Optional[str],
        viewpoint: str,
        instructor_answer: str,
        round_id: int,
        question: str = "",
        guidance: Optional[str] = None,
        evidence_refs: Optional[List[str]] = None,
    ) -> List[dict]:
        ms_section = (
            "=== Student Knowledge Pool Ms (read-only for Arbiter) ===\n" + ms_text + "\n\n"
            if ms_text is not None else ""
        )
        summary_request = (
            "Also provide a short public round_summary based ONLY on the questions, V^(exp), A^(ins), and supplied evidence. "
            "Do not include private Ms interpretations in that summary. Preserve numbers, units, conditions, "
            "unresolved questions, conflicts and corrections. Do not add new claims.\n"
            f"Allowed evidence URLs: {evidence_refs or []}\n"
            f"Keep the summary near {self.options.round_summary_max_chars} characters or shorter.\n"
        ) if self.options.summary_mode == "arbiter" else (
            "Round summaries use a common local extractive policy in this run. "
            "Do not generate one; return round_summary=null.\n"
        )
        summary_shape = ('{"round_id":' + str(round_id)
                         + ',"questions":[],"key_points":[],"unresolved_questions":[],"conflicts":[],"resolved_questions":[],"evidence_refs":[]}') if self.options.summary_mode == "arbiter" else "null"
        content = (
            f"Topic: {topic}\n"
            f"Round {round_id} outputs:\n"
            f"Questions discussed: {question}\nUser guidance: {guidance or 'None'}\n"
            f"V^(exp): {viewpoint}\n"
            f"A^(ins): {instructor_answer}\n\n"
            "=== Shared Knowledge Pool K (current, read-only) ===\n"
            f"{shared_k_text}\n\n"
            f"{ms_section}"
            "Return only new or revised knowledge items. Reuse an existing id for revisions; "
            "omitted entries remain stored. Use remove_ids only to explicitly supersede old items.\n"
            f"{summary_request}"
            "HARD FORMAT RULES:\n"
            "- Return COMPACT SINGLE-LINE JSON only\n"
            "- No markdown fences / no code block\n"
            "- items length MUST be <= 8\n"
            "- each content MUST be a short concise sentence\n"
            "- keys must match exactly\n"
            "Required shape:\n"
            '{"items":[{"id":null,"content":"...","label":"...","source_role":"instructor","round_id":'
            f'{round_id}'
            ',"conflict_tag":null,"source_urls":[],"created_at":null,"updated_at":null}],'
            '"remove_ids":[],"round_summary":' + summary_shape + '}'
        )
        return [{"role": "user", "content": content}]

    def build_arbiter_patch_messages(
        self,
        topic: str,
        question: str,
        ms_update_text: Optional[str],
        scores_text: str,
        round_id: int,
    ) -> List[dict]:
        ms_summary = f"Ms update summary: {ms_update_text}\n" if ms_update_text is not None else ""
        content = (
            f"Topic: {topic}\n"
            f"Round {round_id} student question: {question}\n"
            f"Scores breakdown: {scores_text}\n"
            f"{ms_summary}"
            "Propose prompt/parameter patches for student, instructor, expert (and optionally arbiter) "
            "that improve the next round without violating access controls. Keep patches brief.\n"
            "Hard rule: proposed patches must preserve Topic focus and must not encourage cross-topic drift."
        )
        return [{"role": "user", "content": content}]
