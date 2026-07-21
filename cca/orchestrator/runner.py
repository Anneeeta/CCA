from __future__ import annotations

import logging
from typing import List, Optional

from ..agents.arbiter import ArbiterAgent
from ..agents.expert import ExpertAgent
from ..agents.instructor import InstructorAgent
from ..agents.student import StudentAgent
from ..config import Settings, get_settings
from ..memory.shared_pool import SharedKnowledgePool
from ..memory.student_pool import StudentPool
from ..orchestrator.context_builder import ContextBuilder, DialogueTurn
from ..schemas import (
    AgentPatch,
    ArbiterPatchResponse,
    CandidateQuestion,
    CandidateScore,
    KnowledgePoolUpdate,
    LLMRankingResponse,
    RoundRecord,
    StudentMsUpdate,
)
from ..scoring.embeddings import EmbeddingProvider
from ..scoring.scoring import score_candidates, select_top_two
from ..storage import persistence
from ..utils.logging import setup_logging
from ..retriever import DuckDuckGoRetriever, TavilyRetriever

logger = logging.getLogger(__name__)


class CCAOrchestrator:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        setup_logging(self.settings.paths.log_dir)

        self.shared_pool = SharedKnowledgePool(self.settings.paths.shared_pool_path)
        self.student_pool = StudentPool(self.settings.paths.student_pool_path)
        self.context = ContextBuilder()
        self.embedding_provider = EmbeddingProvider(self.settings.orchestrator.embedding_model)

        self.student = StudentAgent(settings=self.settings)
        self.instructor = InstructorAgent(settings=self.settings)
        self.expert = ExpertAgent(settings=self.settings)
        self.arbiter = ArbiterAgent(settings=self.settings)

        # Initialize retriever if enabled
        self.retriever = None
        if self.settings.orchestrator.enable_retrieval:
            if self.settings.orchestrator.retriever_type == "duckduckgo":
                self.retriever = DuckDuckGoRetriever(
                    max_results=self.settings.orchestrator.retriever_max_results,
                )
                logger.info("DuckDuckGo retriever enabled (max_results=%d)", self.settings.orchestrator.retriever_max_results)
            elif self.settings.orchestrator.retriever_type == "tavily":
                if not self.settings.tavily_api_key:
                    logger.error("Tavily retriever requires TAVILY_API_KEY environment variable")
                else:
                    self.retriever = TavilyRetriever(
                        api_key=self.settings.tavily_api_key,
                        api_url=self.settings.orchestrator.tavily_api_url,
                        max_results=self.settings.orchestrator.retriever_max_results,
                        search_depth=self.settings.orchestrator.tavily_search_depth,
                    )
                    logger.info(
                        "Tavily retriever enabled (api_url=%s, max_results=%d)",
                        self.settings.orchestrator.tavily_api_url,
                        self.settings.orchestrator.retriever_max_results
                    )
            else:
                logger.warning("Unknown retriever type: %s", self.settings.orchestrator.retriever_type)

        self.history: List[DialogueTurn] = []

    # ------------------------------------------------------------------ #

    def bootstrap(self, topic: str, display: bool = True, domain: Optional[str] = None) -> Optional[str]:
        if not self.settings.orchestrator.bootstrap:
            return None
        logger.info("Bootstrapping with instructor overview.")
        k_text = self.shared_pool.format_for_context()
        topic_with_domain = topic if not domain else f"{topic}\nDomain: {domain}"

        # Step 1: identify at least three subfields for the topic and domain.
        subfields = self._bootstrap_identify_subfields(topic_with_domain)
        if display:
            print(f"\n=== Bootstrap Sub-fields ===\n{subfields}")

        # Step 2: retrieve material for the topic and each subfield.
        retrieved_context = None
        if self.retriever is not None:
            max_chars = self.settings.orchestrator.retriever_context_max_chars
            all_snippets = []
            # Retrieve broad topic context.
            topic_queries = [topic, f"{topic} overview", f"{topic} {domain}" if domain else topic]
            logger.info("[Bootstrap] Topic queries: %s", topic_queries)
            topic_snippets = self.retriever.search_multiple(topic_queries)
            all_snippets.extend(topic_snippets)
            if display:
                print(f"[Bootstrap] Topic: {len(topic_snippets)} snippets")

            # Retrieve context for each subfield.
            for sf in subfields:
                sf_queries = [sf, f"{sf} {topic}"]
                logger.info("[Bootstrap] Sub-field queries: %s", sf_queries)
                sf_snippets = self.retriever.search_multiple(sf_queries)
                all_snippets.extend(sf_snippets)
                if display:
                    print(f"[Bootstrap] {sf}: {len(sf_snippets)} snippets")

            if all_snippets:
                retrieved_context = DuckDuckGoRetriever.format_snippets(all_snippets, max_chars=max_chars)

        # Step 3: generate a comprehensive overview from retrieved context.
        overview_question = (
            f"Provide a comprehensive overview of {topic_with_domain}. "
            f"Cover the following sub-aspects in detail: {', '.join(subfields)}. "
            f"For each sub-aspect, explain key concepts, mechanisms, practical applications, and important considerations."
        )
        messages = self.context.build_instructor_teach_messages(
            topic=topic_with_domain,
            shared_k_text=k_text,
            question=overview_question,
            history=[],
            expert_viewpoint=retrieved_context,
        )
        overview_text = self.instructor.teach(messages)
        if display:
            print("\n=== Instructor Kickoff ===")
            print(overview_text)

        # Step 4: update shared pool K through the Arbiter.
        ms_text = self.student_pool.format_for_context()
        arbiter_msgs = self.context.build_arbiter_k_messages(
            topic=topic,
            shared_k_text=k_text,
            ms_text=ms_text,
            viewpoint=retrieved_context or overview_text,
            instructor_answer=overview_text,
            round_id=0,
        )
        try:
            k_update = self.arbiter.update_shared_pool(arbiter_msgs)
        except Exception as e:
            logger.warning("Bootstrap Arbiter K update failed; using empty update. Error: %s", e)
            k_update = KnowledgePoolUpdate(items=[])
        self.shared_pool.apply_update(k_update)

        # Step 5: add the bootstrap turn to dialogue history.
        self.history.append(
            DialogueTurn(
                question=f"Kickoff lecture on {topic_with_domain}",
                instructor_answer=overview_text,
                expert_viewpoint=retrieved_context,
            )
        )

        # Step 6: persist bootstrap content locally.
        bootstrap_data = {
            "topic": topic_with_domain,
            "subfields": subfields,
            "search_context": retrieved_context,
            "overview": overview_text,
        }
        import json as _json
        bootstrap_path = self.settings.paths.data_dir / "bootstrap.json"
        try:
            with open(bootstrap_path, "w", encoding="utf-8") as f:
                _json.dump(bootstrap_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return overview_text

    def _bootstrap_identify_subfields(self, topic_with_domain: str) -> List[str]:
        """Ask the Instructor to identify at least three topic subfields."""
        prompt = (
            f"Topic: {topic_with_domain}\n\n"
            "Analyze this topic and identify at least 3 distinct sub-aspects or sub-fields "
            "that are important for a comprehensive understanding. "
            "Each sub-aspect should be a specific angle, mechanism, application area, or key dimension.\n\n"
            "Output ONLY a numbered list, one sub-aspect per line. No other text.\n"
            "Example:\n"
            "1. Core concepts and definitions\n"
            "2. Key mechanisms and methodologies\n"
            "3. Practical applications and case studies\n"
            "4. Challenges and limitations"
        )
        raw = self.instructor.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        # Parse a numbered list while tolerating plain list entries.
        subfields = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove an optional numeric prefix.
            import re
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                subfields.append(m.group(1).strip())
            elif line and not line.startswith("#"):
                subfields.append(line)
        # Guarantee a minimum of three entries.
        if len(subfields) < 3:
            subfields = [topic_with_domain, "key concepts", "practical applications"]
        return subfields[:6]

    # ------------------------------------------------------------------ #

    def _flatten_ms_entries(self) -> List[str]:
        ms = self.student_pool.load()
        texts: List[str] = []
        for items in ms.values():
            texts.extend([entry.content for entry in items])
        return texts

    def _summarize_ms_update(self, update: StudentMsUpdate) -> str:
        parts = []
        for bucket_name in ["known", "known_unknown", "conflict", "unknown_unknown"]:
            entries = getattr(update, bucket_name)
            parts.append(f"{bucket_name}: {len(entries)} new")
        return "; ".join(parts)

    def _summarize_scores(self, scores: List[CandidateScore]) -> str:
        return "; ".join([f"{s.question[:40]}... -> I={s.I:.3f}" for s in scores])

    @staticmethod
    def _build_kickoff_comprehensive_question(topic: str) -> str:
        return (
            f"Kickoff comprehensive teaching request for topic '{topic}': "
            "provide a broad but structured lesson that covers foundations and key terminology, "
            "core principles or mechanisms, major approaches or variants, practical application choices, "
            "common failure modes or misconceptions, and how success is evaluated in real use. "
            "Assume the learner is starting from fundamentals. "
            "Keep the explanation coherent and focused on this topic only."
        )

    def _validate_patch(self, patch: AgentPatch) -> bool:
        """
        Reject patches that would break access control.
        """
        text = (patch.prompt_patch or "").lower()
        if patch.agent == "student" and ("shared" in text and "k" in text or "shared pool" in text):
            return False
        if patch.agent in ("instructor", "expert") and "student" in text and "ms" in text:
            return False
        return True

    def _llm_rank_candidates(
        self,
        topic: str,
        candidates: List[CandidateQuestion],
        ms_text: str,
        history: List[DialogueTurn],
        guidance: Optional[str],
    ) -> List[CandidateScore]:
        history_text = self.context._format_history(history)
        prompt = (
            f"Topic: {topic}\n"
            "Rank the candidate questions by utility and novelty. "
            "Use the student knowledge pool Ms and history; ignore shared pool K.\n"
            f"Guidance: {guidance or 'None'}\n"
            f"History:\n{history_text}\n\n"
            "=== Student Knowledge Pool Ms ===\n"
            f"{ms_text}\n\n"
            "Candidates:\n"
            + "\n".join([f"- {c.question} (rationale: {c.rationale})" for c in candidates])
            + "\nReturn JSON: {\"scores\": [{\"question\": \"...\", \"I\": 0.0-1.0}]}"
        )
        resp = self.student.call_json([{"role": "user", "content": prompt}], LLMRankingResponse)
        scores = []
        for entry in resp.scores:
            scores.append(CandidateScore(question=entry.question, s=0.0, r=0.0, novel=0.0, I=entry.I))
        return scores

    def apply_patches(self, patch_resp: ArbiterPatchResponse) -> None:
        for patch in patch_resp.patches:
            if not self._validate_patch(patch):
                logger.warning("Rejected patch for %s due to access control violation.", patch.agent)
                continue
            target = {
                "student": self.student,
                "instructor": self.instructor,
                "expert": self.expert,
                "arbiter": self.arbiter,
            }.get(patch.agent)
            if target:
                target.apply_patch(patch)

    def _select_candidates(
        self,
        candidates: List[CandidateQuestion],
        scores: List[CandidateScore],
        manual_select: bool,
        display: bool,
    ) -> tuple[CandidateScore, Optional[CandidateScore]]:
        if not manual_select or not display:
            return select_top_two(scores, self.embedding_provider)

        prompt = (
            f"Select one or two candidate questions [1-{len(candidates)}] "
            "(for example, 1 or 1,3), or press Enter for automatic selection: "
        )
        while True:
            raw = input(prompt).strip()
            if not raw:
                return select_top_two(scores, self.embedding_provider)

            parts = [part.strip() for part in raw.split(",")]
            if len(parts) not in (1, 2) or any(not part.isdigit() for part in parts):
                print("Invalid input. Enter one or two comma-separated numbers.")
                continue

            indices = [int(part) - 1 for part in parts]
            if len(set(indices)) != len(indices):
                print("Select distinct candidate questions.")
                continue
            if any(idx < 0 or idx >= len(candidates) for idx in indices):
                print(f"Out of range. Enter a number between 1 and {len(candidates)}.")
                continue

            def score_at(idx: int) -> CandidateScore:
                question = candidates[idx].question
                match = next((score for score in scores if score.question == question), None)
                if match is not None:
                    return match
                source = scores[idx]
                return CandidateScore(
                    question=question,
                    s=source.s,
                    r=source.r,
                    novel=source.novel,
                    I=source.I,
                )

            first = score_at(indices[0])
            second = score_at(indices[1]) if len(indices) == 2 else None
            return first, second

    # ------------------------------------------------------------------ #

    def run_round(
        self,
        topic: str,
        round_id: int,
        guidance: Optional[str],
        display: bool = True,
        persist_log: bool = True,
        manual_select: bool = False,
    ) -> RoundRecord:
        ms_text = self.student_pool.format_for_context()
        k_text = self.shared_pool.format_for_context()

        # Student generates candidate questions
        candidate_msgs = self.context.build_student_candidate_messages(
            topic=topic,
            ms_text=ms_text,
            history=self.history,
            guidance=guidance,
            num_candidates=self.settings.orchestrator.num_candidates,
        )
        candidates: List[CandidateQuestion] = self.student.generate_candidates(candidate_msgs)

        # Score and select
        ms_entries = self._flatten_ms_entries()
        history_questions = [h.question for h in self.history]
        weights = self.settings.orchestrator.weights
        if self.embedding_provider.available:
            scores = score_candidates(
                candidates,
                ms_entries=ms_entries,
                guidance=guidance,
                history_questions=history_questions,
                provider=self.embedding_provider,
                alpha=weights.alpha,
                beta=weights.beta,
                gamma=weights.gamma,
                top_k=self.settings.orchestrator.top_k_similarity,
            )
        elif self.settings.orchestrator.allow_llm_ranking_fallback:
            scores = self._llm_rank_candidates(
                topic=topic,
                candidates=candidates,
                ms_text=ms_text,
                history=self.history,
                guidance=guidance,
            )
        else:
            raise RuntimeError("No embedding provider available and LLM fallback disabled.")
        if display:
            print(f"\n=== Round {round_id} ===")
            print("[Student] Candidate questions and scores:")
            for idx, (cand, score) in enumerate(zip(candidates, scores), start=1):
                rationale = f" | rationale: {cand.rationale}" if cand.rationale else ""
                print(
                    f"[{idx}] Q: {cand.question}{rationale}\n"
                    f"  s={score.s:.3f}, r={score.r:.3f}, novel={score.novel:.3f}, I={score.I:.3f}"
                )

        # Select up to two questions from distinct subtopics.
        selected_1, selected_2 = self._select_candidates(
            candidates=candidates,
            scores=scores,
            manual_select=manual_select,
            display=display,
        )
        q1_text = selected_1.question
        q2_text = selected_2.question if selected_2 else None

        logger.info("Selected Q1: %s (I=%.3f)", q1_text, selected_1.I)
        if q2_text:
            logger.info("Selected Q2: %s (I=%.3f)", q2_text, selected_2.I)
        if display:
            print(f"[Student] Selected Q1: {q1_text} (I={selected_1.I:.3f})")
            if q2_text:
                print(f"[Student] Selected Q2: {q2_text} (I={selected_2.I:.3f})")

        # Combine both selected questions into a single downstream request.
        if q2_text:
            combined_question = (
                f"(a) {q1_text}\n"
                f"(b) {q2_text}"
            )
        else:
            combined_question = q1_text

        # Instructor routes domain
        instructor_msgs_route = self.context.build_instructor_route_messages(
            topic=topic,
            shared_k_text=k_text,
            question=combined_question,
            history=self.history,
        )
        domain = self.instructor.route_domain(instructor_msgs_route)

        # Retrieve evidence for each selected question independently.
        retrieved_context = None
        if self.retriever is not None:
            max_chars = self.settings.orchestrator.retriever_context_max_chars
            all_snippets = []

            # Retrieve evidence for the first question.
            q1_query_msgs = self.context.build_expert_messages(
                topic=topic, shared_k_text=k_text, question=q1_text,
                domain=domain, history=self.history,
            )
            q1_queries = self.expert.generate_queries(q1_query_msgs, num_queries=3)
            if q1_queries:
                logger.info("[Retriever] Q1 queries: %s", q1_queries)
                q1_snippets = self.retriever.search_multiple(q1_queries)
                all_snippets.extend(q1_snippets)
                if display:
                    print(f"[Retriever] Q1: {len(q1_snippets)} snippets from {len(q1_queries)} queries")

            # Retrieve evidence for the second question.
            if q2_text:
                q2_query_msgs = self.context.build_expert_messages(
                    topic=topic, shared_k_text=k_text, question=q2_text,
                    domain=domain, history=self.history,
                )
                q2_queries = self.expert.generate_queries(q2_query_msgs, num_queries=3)
                if q2_queries:
                    logger.info("[Retriever] Q2 queries: %s", q2_queries)
                    q2_snippets = self.retriever.search_multiple(q2_queries)
                    all_snippets.extend(q2_snippets)
                    if display:
                        print(f"[Retriever] Q2: {len(q2_snippets)} snippets from {len(q2_queries)} queries")

            if all_snippets:
                retrieved_context = DuckDuckGoRetriever.format_snippets(all_snippets, max_chars=max_chars)

        expert_msgs = self.context.build_expert_messages(
            topic=topic,
            shared_k_text=k_text,
            question=combined_question,
            domain=domain,
            history=self.history,
            retrieved_context=retrieved_context,
        )
        expert_resp = self.expert.respond(expert_msgs)
        if display:
            print(f"[Expert] Domain: {domain}")
            print(f"[Expert] Viewpoint:\n{expert_resp.viewpoint}\n")

        # The Instructor answers the selected question or question pair.
        instructor_msgs_final = self.context.build_instructor_teach_messages(
            topic=topic,
            shared_k_text=k_text,
            question=combined_question,
            history=self.history,
            expert_viewpoint=expert_resp.viewpoint,
        )
        instructor_answer = self.instructor.teach(instructor_msgs_final)
        if display:
            print(f"[Instructor] Answer:\n{instructor_answer}\n")

        # Update history
        self.history.append(
            DialogueTurn(
                question=combined_question,
                instructor_answer=instructor_answer,
                expert_viewpoint=expert_resp.viewpoint,
            )
        )

        # Student Ms update
        ms_update_messages = self.context.build_student_update_messages(
            answer_text=instructor_answer,
            ms_text=ms_text,
            round_id=round_id,
        )
        try:
            ms_update = self.student.update_ms(ms_update_messages, round_id=round_id)
        except Exception as e:
            logger.warning("Student Ms update failed, using empty update: %s", e)
            ms_update = StudentMsUpdate()
        new_ms_state = self.student_pool.apply_update(ms_update)
        ms_text_after = self.student_pool.format_for_context()

        # Arbiter updates shared K
        arbiter_msgs = self.context.build_arbiter_k_messages(
            topic=topic,
            shared_k_text=k_text,
            ms_text=ms_text_after,
            viewpoint=expert_resp.viewpoint,
            instructor_answer=instructor_answer,
            round_id=round_id,
        )
        k_update = self.arbiter.update_shared_pool(arbiter_msgs)
        new_k_state = self.shared_pool.apply_update(k_update)

        # Arbiter patches
        scores_text = self._summarize_scores(scores)
        ms_update_text = self._summarize_ms_update(ms_update)
        arbiter_patch_msgs = self.context.build_arbiter_patch_messages(
            topic=topic,
            question=combined_question,
            ms_update_text=ms_update_text,
            scores_text=scores_text,
            round_id=round_id,
        )
        patches = self.arbiter.generate_patch(arbiter_patch_msgs)
        self.apply_patches(patches)
        logger.info("[Arbiter] K update items: %s | Ms delta: %s", len(k_update.items), ms_update_text)
        if patches.patches:
            patch_brief = "; ".join(
                [f"{p.agent}: {(p.prompt_patch or 'param patch')[:80]}" for p in patches.patches]
            )
            logger.info("[Arbiter] Tuning patches -> %s", patch_brief)

        # Round log
        record = RoundRecord(
            topic=topic,
            round_id=round_id,
            user_guidance=guidance,
            candidate_questions=candidates,
            candidate_scores=scores,
            selected_question=q1_text,
            expert_viewpoint=expert_resp.viewpoint,
            instructor_answer=instructor_answer,
            selected_question_2=q2_text,
            expert_viewpoint_2=None,
            instructor_answer_2=None,
            student_update=ms_update,
            arbiter_update=k_update,
            arbiter_patch=patches,
        )
        if persist_log:
            persistence.append_jsonl(self.settings.paths.round_log_path, record.model_dump())

        logger.info("Round %s complete. Ms entries now: %s", round_id, len(self._flatten_ms_entries()))
        logger.info("Shared K size: %s", len(new_k_state))
        return record

    # ------------------------------------------------------------------ #

    def finalize(self, topic: str) -> str:
        ms_text = self.student_pool.format_for_context()
        k_text = self.shared_pool.format_for_context()

        # Build a complete dialogue transcript for final synthesis.
        history_lines = []
        for idx, h in enumerate(self.history, start=1):
            history_lines.append(f"Round {idx} Q: {h.question}")
            if h.expert_viewpoint:
                history_lines.append(f"Round {idx} Expert: {h.expert_viewpoint}")
            history_lines.append(f"Round {idx} Answer: {h.instructor_answer}")
            history_lines.append("")
        history_text = "\n".join(history_lines)

        summary_prompt = (
            f"Topic: {topic}\n\n"
            "You are a tutor. Based on the full tutoring dialogue, knowledge pools, and expert research below, "
            "write a comprehensive final summary that covers ALL sub-topics discussed across all rounds.\n\n"
            "Requirements:\n"
            "- Cover every sub-topic that was discussed in the dialogue\n"
            "- Synthesize insights from the knowledge pools and expert research\n"
            "- Include key concepts, mechanisms, practical examples, and important takeaways\n"
            "- Mention unresolved questions or areas for further exploration\n"
            "- Write in a clear, educational style suitable for a learner\n"
            "- Aim for 800-1500 words\n\n"
            "=== Dialogue History ===\n"
            f"{history_text}\n\n"
            "=== Shared Knowledge Pool K ===\n"
            f"{k_text}\n\n"
            "=== Student Knowledge Pool Ms ===\n"
            f"{ms_text}\n\n"
            "Write the final summary now. Output plain text only, no JSON."
        )
        summary = self.instructor.call([{"role": "user", "content": summary_prompt}])
        logger.info("Final summary generated.")
        return summary

    def run(self, topic: str, manual_select: bool = False) -> None:
        self.bootstrap(topic)
        round_id = 1
        while True:
            guidance = input("Optional guidance for this round (press Enter to skip): ").strip()
            guidance = guidance or None
            try:
                self.run_round(topic, round_id, guidance, manual_select=manual_select)
            except Exception as e:
                logger.exception("Round %s failed: %s", round_id, e)
                choice = input("Round failed. Continue to next round anyway? (y/n): ").strip().lower()
                if choice != "y":
                    break
            cont = input("Continue to next round? (y/n): ").strip().lower()
            if cont != "y":
                break
            round_id += 1
        summary = self.finalize(topic)
        print("\n=== Final Summary (Instructor) ===\n")
        print(summary)
