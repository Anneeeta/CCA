from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DialogueTurn:
    question: str
    instructor_answer: str
    expert_viewpoint: Optional[str] = None


class ContextBuilder:
    """
    Builds strict, readable contexts for each agent while enforcing access rules.
    """

    @staticmethod
    def _format_history(history: List[DialogueTurn]) -> str:
        if not history:
            return "(no prior rounds)"
        lines = []
        for idx, h in enumerate(history, start=1):
            lines.append(f"Round {idx}: Q: {h.question}")
            if h.expert_viewpoint:
                lines.append(f"  Expert: {h.expert_viewpoint}")
            lines.append(f"  Instructor: {h.instructor_answer}")
        return "\n".join(lines)

    # Student contexts ------------------------------------------------------

    def build_student_candidate_messages(
        self,
        topic: str,
        ms_text: str,
        history: List[DialogueTurn],
        guidance: Optional[str],
        num_candidates: int,
    ) -> List[dict]:
        history_text = self._format_history(history)
        content = (
            f"Topic: {topic}\n"
            f"Guidance (optional): {guidance or 'None'}\n"
            "Your task: generate diverse candidate questions for the next discussion turn.\n"
            "Hard rule: keep all candidates tightly anchored to Topic and direct subtopics.\n"
            "Hard rule: if Ms/history contains drifted content, ignore off-topic parts.\n"
            "Hard rule: every candidate must clearly reference Topic-specific concepts, not generic adjacent areas.\n"
            "Hard rule: do not pivot to sibling frameworks unless guidance explicitly requests comparison.\n"
            "Prefer candidate breadth across multiple subtopics, not one narrow angle only.\n"
            f"Number of candidates: {num_candidates}\n"
            "Context history:\n"
            f"{history_text}\n\n"
            "=== Student Knowledge Pool Ms (read/write) ===\n"
            f"{ms_text}\n\n"
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
            "Current Ms (for reference, you may append):\n"
            f"{ms_text}\n\n"
            "Return JSON with lists for known, known_unknown, conflict, unknown_unknown."
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
        history_text = self._format_history(history)
        content = (
            f"Topic: {topic}\n"
            f"Student question: {question}\n"
            f"History:\n{history_text}\n\n"
            "=== Shared Knowledge Pool K (read-only) ===\n"
            f"{shared_k_text}\n\n"
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
        ms_text: str,
        viewpoint: str,
        instructor_answer: str,
        round_id: int,
    ) -> List[dict]:
        content = (
            f"Topic: {topic}\n"
            f"Round {round_id} outputs:\n"
            f"V^(exp): {viewpoint}\n"
            f"A^(ins): {instructor_answer}\n\n"
            "=== Shared Knowledge Pool K (current, read-only) ===\n"
            f"{shared_k_text}\n\n"
            "=== Student Knowledge Pool Ms (read-only for Arbiter) ===\n"
            f"{ms_text}\n\n"
            "Extract atomic knowledge items, merge/prune, mark conflicts if needed.\n"
            "HARD FORMAT RULES:\n"
            "- Return COMPACT SINGLE-LINE JSON only\n"
            "- No markdown fences / no code block\n"
            "- items length MUST be <= 8\n"
            "- each content MUST be a short concise sentence\n"
            "- keys must match exactly\n"
            "Required shape:\n"
            '{"items":[{"id":null,"content":"...","label":"...","source_role":"instructor","round_id":'
            f'{round_id}'
            ',"conflict_tag":null,"created_at":null,"updated_at":null}]}'
        )
        return [{"role": "user", "content": content}]

    def build_arbiter_patch_messages(
        self,
        topic: str,
        question: str,
        ms_update_text: str,
        scores_text: str,
        round_id: int,
    ) -> List[dict]:
        content = (
            f"Topic: {topic}\n"
            f"Round {round_id} student question: {question}\n"
            f"Scores breakdown: {scores_text}\n"
            f"Ms update summary: {ms_update_text}\n"
            "Propose prompt/parameter patches for student, instructor, expert (and optionally arbiter) "
            "that improve the next round without violating access controls. Keep patches brief.\n"
            "Hard rule: proposed patches must preserve Topic focus and must not encourage cross-topic drift."
        )
        return [{"role": "user", "content": content}]
