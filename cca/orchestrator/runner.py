from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ..agents.arbiter import ArbiterAgent
from ..agents.expert import ExpertAgent
from ..agents.instructor import InstructorAgent
from ..agents.student import StudentAgent
from ..agents.base import AgentOutputError
from ..config import Settings, get_settings
from ..deepseek_client import ProviderConfigurationError, is_fatal_api_error, is_provider_error
from ..memory.shared_pool import SharedKnowledgePool
from ..memory.student_pool import StudentPool
from ..orchestrator.context_builder import ContextBuilder, DialogueTurn
from ..orchestrator.summaries import prepare_summary, source_urls
from .selection import choose_questions, fixed_topic_questions
from ..schemas import (
    AgentPatch,
    ArbiterPatchResponse,
    CandidateQuestion,
    CandidateScore,
    KnowledgePoolUpdate,
    LLMRankingResponse,
    RoundRecord,
    StudentMsUpdate,
    ExpertResponse,
)
from ..scoring.embeddings import EmbeddingProvider
from ..scoring.scoring import score_candidates, select_question
from ..storage import persistence
from ..utils.logging import setup_logging
from ..retriever import DuckDuckGoRetriever, TavilyRetriever, interleave_snippets
from ..deepseek_search import DeepSeekRetriever
from ..tavily_key_rotation import TavilyKeyPoolError

logger = logging.getLogger(__name__)


class CCAOrchestrator:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        setup_logging(self.settings.paths.log_dir)

        self.shared_pool = (
            SharedKnowledgePool(self.settings.paths.shared_pool_path,
                                deduplicate=self.settings.orchestrator.deduplicate_shared_pool)
            if self.settings.orchestrator.shared_knowledge_mode != "off" else None
        )
        self.student_pool = (
            StudentPool(self.settings.paths.student_pool_path,
                        deduplicate=self.settings.orchestrator.deduplicate_student_pool)
            if self.settings.orchestrator.enable_student_pool else None
        )
        self.context = ContextBuilder(self.settings.orchestrator)
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
                    strict_errors=True,
                )
                logger.info("DuckDuckGo retriever enabled (max_results=%d)", self.settings.orchestrator.retriever_max_results)
            elif self.settings.orchestrator.retriever_type == "tavily":
                if not (self.settings.tavily_api_keys or self.settings.tavily_api_key):
                    raise ProviderConfigurationError("Tavily retriever requires TAVILY_API_KEYS or TAVILY_API_KEY")
                else:
                    self.retriever = TavilyRetriever(
                        api_key=self.settings.tavily_api_key,
                        api_keys=self.settings.tavily_api_keys,
                        usage_path=self.settings.tavily_usage_path,
                        per_key_limit=self.settings.tavily_key_call_limit,
                        strict_errors=self.settings.tavily_strict_errors,
                        api_url=self.settings.orchestrator.tavily_api_url,
                        max_results=self.settings.orchestrator.retriever_max_results,
                        search_depth=self.settings.orchestrator.tavily_search_depth,
                    )
                    logger.info(
                        "Tavily retriever enabled (api_url=%s, max_results=%d)",
                        self.retriever.api_url,
                        self.settings.orchestrator.retriever_max_results
                    )
            elif self.settings.orchestrator.retriever_type == "deepseek":
                self.retriever = DeepSeekRetriever(
                    api_key=self.settings.deepseek_search_api_key,
                    max_results=self.settings.orchestrator.retriever_max_results,
                    need_content=self.settings.orchestrator.deepseek_search_need_content,
                    timeout=self.settings.orchestrator.deepseek_search_timeout,
                    record_path=self.settings.paths.data_dir / "deepseek_search_calls.jsonl",
                )
                logger.info("DeepSeek official search enabled (max_results=%d)", self.retriever.max_results)
            else:
                raise ProviderConfigurationError("Unknown retriever type")

        self.history: List[DialogueTurn] = []

    # ------------------------------------------------------------------ #

    def bootstrap(self, topic: str, display: bool = True, domain: Optional[str] = None) -> Optional[str]:
        if not self.settings.orchestrator.bootstrap:
            return None
        logger.info("Bootstrapping with instructor overview.")
        k_text = self._k_context(topic)
        topic_with_domain = topic if not domain else f"{topic}\nDomain: {domain}"

        # Step 1: Instructor 分析 topic+domain，输出 3+ 细分领域
        subfields = self._bootstrap_identify_subfields(topic_with_domain)
        if display:
            print(f"\n=== Bootstrap Sub-fields ===\n{subfields}")

        # Step 2: 对 topic 和每个细分领域调用 Tavily 搜索
        retrieved_context = None
        retrieval_sources = []
        if self.retriever is not None:
            max_chars = self.settings.orchestrator.retriever_context_max_chars
            all_snippets = []
            snippet_groups = []
            # 搜索 topic+domain
            topic_queries = [topic, f"{topic} overview", f"{topic} {domain}" if domain else topic]
            logger.info("[Bootstrap] Topic queries: %s", topic_queries)
            topic_snippets = self.retriever.search_multiple(topic_queries)
            all_snippets.extend(topic_snippets)
            snippet_groups.append(topic_snippets)
            if display:
                print(f"[Bootstrap] Topic: {len(topic_snippets)} snippets")

            # 搜索每个细分领域
            for sf in subfields:
                sf_queries = [sf, f"{sf} {topic}"]
                logger.info("[Bootstrap] Sub-field queries: %s", sf_queries)
                sf_snippets = self.retriever.search_multiple(sf_queries)
                all_snippets.extend(sf_snippets)
                snippet_groups.append(sf_snippets)
                if display:
                    print(f"[Bootstrap] {sf}: {len(sf_snippets)} snippets")

            if all_snippets:
                all_snippets = interleave_snippets(snippet_groups)
                retrieval_sources = [vars(s).copy() for s in all_snippets]
                retrieved_context = DuckDuckGoRetriever.format_snippets(all_snippets[:self.settings.orchestrator.retriever_max_snippets_per_turn], max_chars=max_chars)

        # Step 3: 基于搜索结果生成综合 overview
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

        # Step 4: Arbiter 写入 K
        ms_text = self._ms_context()
        arbiter_msgs = self.context.build_arbiter_k_messages(
            topic=topic,
            shared_k_text=self._k_context(topic, for_arbiter=True),
            ms_text=ms_text,
            viewpoint=retrieved_context or overview_text,
            instructor_answer=overview_text,
            round_id=0,
            question=f"Kickoff lecture on {topic_with_domain}",
            evidence_refs=source_urls(retrieved_context or ""),
        )
        try:
            k_update = self.arbiter.update_shared_pool(arbiter_msgs) if self.settings.orchestrator.enable_arbiter and self.shared_pool is not None else KnowledgePoolUpdate()
        except Exception as e:
            logger.warning("Bootstrap Arbiter K update failed: %s", type(e).__name__)
            raise
        self._validate_knowledge_sources(k_update, source_urls(retrieved_context or ""))
        for item in k_update.items:
            item.round_id = 0
        if self.shared_pool is not None:
            self.shared_pool.apply_update(k_update)
        round_summary = prepare_summary(
            k_update.round_summary if self.settings.orchestrator.summary_mode == "arbiter" else None, round_id=0,
            question=f"Kickoff lecture on {topic_with_domain}", answer=overview_text,
            expert=retrieved_context, evidence_refs=source_urls(retrieved_context or ""),
            max_chars=self.settings.orchestrator.round_summary_max_chars,
        )

        # Step 5: 写入 history
        self.history.append(
            DialogueTurn(
                question=f"Kickoff lecture on {topic_with_domain}",
                instructor_answer=overview_text,
                expert_viewpoint=retrieved_context,
                round_id=0,
                summary=round_summary,
                selected_questions=[],
            )
        )

        # Step 6: 保存 bootstrap 内容
        bootstrap_data = {
            "topic": topic_with_domain,
            "subfields": subfields,
            "search_context": retrieved_context,
            "overview": overview_text,
            "round_summary": round_summary.model_dump(),
            "retrieval_sources": retrieval_sources,
            "context_policy": self._context_policy(),
            "context_settings": self.settings.orchestrator.model_dump(include={
                "compact_context", "summary_mode", "history_context_max_chars", "final_history_max_chars",
                "pool_context_max_chars", "helper_context_max_chars", "round_summary_max_chars", "pool_context_max_items",
            }),
            "execution_policy": self._execution_policy(),
        }
        bootstrap_path = self.settings.paths.data_dir / "bootstrap.json"
        persistence.write_json_file(bootstrap_path, bootstrap_data)

        return overview_text

    def _bootstrap_identify_subfields(self, topic_with_domain: str) -> List[str]:
        """让 Instructor 分析 topic+domain，输出至少 3 个细分领域"""
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
            model=self.instructor.model_override,
            temperature=self.instructor.param_overrides.get("temperature"),
            top_p=self.instructor.param_overrides.get("top_p"),
            max_tokens=min(500, int(self.instructor.param_overrides.get("max_tokens", self.settings.model.max_tokens))),
        )
        # 解析编号列表
        subfields = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 去掉编号前缀
            import re
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                subfields.append(m.group(1).strip())
            elif line and not line.startswith("#"):
                subfields.append(line)
        # 至少返回 3 个
        if len(subfields) < 3:
            raise AgentOutputError("Bootstrap subfield stage returned fewer than three usable subfields")
        return subfields[:6]  # 最多 6 个

    # ------------------------------------------------------------------ #

    def _ms_context(self, query: str = "") -> Optional[str]:
        if self.student_pool is None:
            return None
        options = self.settings.orchestrator
        return self.student_pool.format_for_context(
            query=query,
            max_chars=options.pool_context_max_chars if options.compact_context else None,
            max_items=options.pool_context_max_items if options.compact_context else None,
        )

    def _k_context(self, query: str = "", *, for_arbiter: bool = False) -> str:
        options = self.settings.orchestrator
        if self.shared_pool is None:
            return "(shared knowledge storage disabled)"
        if not for_arbiter and not options.use_shared_context:
            return "(shared knowledge not supplied for this stage)"
        return self.shared_pool.format_for_context(
            query=query,
            max_chars=options.pool_context_max_chars if options.compact_context else None,
            max_items=options.pool_context_max_items if options.compact_context else None,
        )

    def _validate_knowledge_sources(self, update: KnowledgePoolUpdate, current_urls: List[str]) -> None:
        known = set(current_urls)
        for item in self.shared_pool.load() if self.shared_pool is not None else []:
            known.update(item.source_urls)
        for item in update.items:
            item.source_urls = [url for url in item.source_urls if url in known]

    def _execution_policy(self) -> dict:
        from .policy import POLICY_FIELDS
        return self.settings.orchestrator.model_dump(include=POLICY_FIELDS)

    def _context_policy(self) -> str:
        if not self.settings.orchestrator.compact_context:
            return "full-history-incremental-memory-v1"
        if self.settings.orchestrator.history_mode == "raw":
            return "compact-pools-raw-history-v1"
        return "incremental-summary-v1"

    def _flatten_ms_entries(self) -> List[str]:
        if self.student_pool is None:
            return []
        ms = self.student_pool.load()
        texts: List[str] = []
        for items in ms.values():
            texts.extend([entry.content for entry in items])
        return texts

    def _summarize_ms_update(self, update: StudentMsUpdate) -> str:
        parts = []
        for bucket_name in ["known", "known_unknown", "conflict", "unknown_unknown"]:
            entries = getattr(update, bucket_name)
            parts.append(f"{bucket_name}: {len(entries)} returned")
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
        if self.student_pool is None and re.search(
            r"\bms\b|student (?:knowledge )?pool|learner[- ]state (?:pool|storage)|state pool", text
        ):
            return False
        if patch.agent == "student" and ("shared" in text and "k" in text or "shared pool" in text):
            return False
        if patch.agent in ("instructor", "expert") and "student" in text and "ms" in text:
            return False
        return True

    def _llm_rank_candidates(
        self,
        topic: str,
        candidates: List[CandidateQuestion],
        ms_text: Optional[str],
        history: List[DialogueTurn],
        guidance: Optional[str],
    ) -> List[CandidateScore]:
        history_text = self.context._format_history(history)
        ranking_basis = (
            "Use the student knowledge pool Ms and history; ignore shared pool K.\n"
            if ms_text is not None else
            "Use the dialogue history and user guidance; ignore shared pool K.\n"
        )
        ms_section = (
            "=== Student Knowledge Pool Ms ===\n" + ms_text + "\n\n"
            if ms_text is not None else ""
        )
        prompt = (
            f"Topic: {topic}\n"
            "Rank the candidate questions by utility and novelty. "
            f"{ranking_basis}"
            f"Guidance: {guidance or 'None'}\n"
            f"History:\n{history_text}\n\n"
            f"{ms_section}"
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

    def _select_candidate(
        self,
        candidates: List[CandidateQuestion],
        scores: List[CandidateScore],
        manual_select: bool,
        display: bool,
    ) -> CandidateScore:
        if not manual_select:
            return select_question(scores)

        for index, candidate in enumerate(candidates, 1):
            print(f"[{index}] {candidate.question}")
        prompt = (
            f"Select candidate question [1-{len(candidates)}], "
            "or press Enter to use the top-scoring one: "
        )
        while True:
            raw = input(prompt).strip()
            if not raw:
                return select_question(scores)
            if not raw.isdigit():
                print("Invalid input. Please enter a number.")
                continue
            idx = int(raw) - 1
            if idx < 0 or idx >= len(candidates):
                print(f"Out of range. Enter a number between 1 and {len(candidates)}.")
                continue
            selected_question = candidates[idx].question
            selected_score = next((s for s in scores if s.question == selected_question), None)
            if selected_score is not None:
                return selected_score
            # Fallback to positional mapping if question text matching fails.
            return CandidateScore(
                question=selected_question,
                s=scores[idx].s,
                r=scores[idx].r,
                novel=scores[idx].novel,
                I=scores[idx].I,
            )

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
        pending = {"topic": topic, "round_id": round_id, "user_guidance": guidance, "status": "running"}

        def checkpoint(stage: str, **values):
            if persist_log:
                pending.update(values, last_completed_stage=stage)
                persistence.write_json_file(self.settings.paths.data_dir / "pending_round.json", pending)

        checkpoint("started")
        ms_text = self._ms_context(f"{topic} {guidance or ''}")
        k_text = self._k_context(f"{topic} {guidance or ''}")

        # Student generates candidate questions
        candidate_msgs = self.context.build_student_candidate_messages(
            topic=topic,
            ms_text=ms_text if self.settings.orchestrator.student_candidate_state else None,
            history=self.history,
            guidance=guidance,
            num_candidates=self.settings.orchestrator.num_candidates,
        )
        options = self.settings.orchestrator
        candidates: List[CandidateQuestion] = (
            self.student.generate_candidates(candidate_msgs) if options.enable_student_questions
            else fixed_topic_questions(topic, guidance, options.questions_per_round)
        )
        checkpoint("candidates_generated", candidate_questions=[candidate.model_dump() for candidate in candidates])
        questions = [candidate.question for candidate in candidates]
        if len(questions) < options.questions_per_round or len(set(questions)) != len(questions):
            raise ValueError("Invalid candidate set: not enough distinct candidate questions.")
        if options.enable_student_questions and options.require_candidate_count and len(questions) != options.num_candidates:
            raise ValueError("Invalid candidate set: candidate count differs from the common experiment protocol.")
        if options.enable_student_questions and options.require_candidate_count and any(candidate.type is None for candidate in candidates):
            raise ValueError("Invalid candidate set: every candidate requires its generated type label.")

        # Score and select
        ms_entries = self._flatten_ms_entries()
        history_questions = [question for turn in self.history
                             for question in (turn.selected_questions if turn.selected_questions is not None else [turn.question])]
        weights = self.settings.orchestrator.weights
        if not options.enable_student_questions or options.selection_policy == "random":
            # These policies do not invoke a ranker or an LLM ranking fallback.
            scores = [CandidateScore(question=c.question, s=0, r=0, novel=0, I=0) for c in candidates]
        elif self.embedding_provider.available:
            scores = score_candidates(
                candidates,
                ms_entries=ms_entries,
                guidance=guidance,
                history_questions=history_questions,
                provider=self.embedding_provider,
                alpha=weights.alpha if self.student_pool is not None and options.score_use_state else 0.0,
                beta=weights.beta if options.score_use_guidance else 0.0,
                gamma=weights.gamma if options.score_use_novelty else 0.0,
                top_k=self.settings.orchestrator.top_k_similarity,
            )
        elif self.settings.orchestrator.allow_llm_ranking_fallback:
            if not all((options.score_use_state, options.score_use_guidance, options.score_use_novelty)):
                raise RuntimeError("Score-term ablations require local embeddings, not LLM ranking fallback.")
            scores = self._llm_rank_candidates(
                topic=topic,
                candidates=candidates,
                ms_text=ms_text,
                history=self.history,
                guidance=guidance,
            )
        else:
            raise RuntimeError("No embedding provider available and LLM fallback disabled.")
        scored_questions = [score.question for score in scores]
        if len(scored_questions) != len(questions) or set(scored_questions) != set(questions):
            raise ValueError("Invalid candidate scores: scores must cover each candidate exactly once.")
        if display:
            print(f"\n=== Round {round_id} ===")
            print("[Student] Candidate questions and scores:")
            for idx, (cand, score) in enumerate(zip(candidates, scores), start=1):
                rationale = f" | rationale: {cand.rationale}" if cand.rationale else ""
                print(
                    f"[{idx}] Q: {cand.question}{rationale}\n"
                    f"  s={score.s:.3f}, r={score.r:.3f}, novel={score.novel:.3f}, I={score.I:.3f}"
                )

        # 选两个不同子话题的问题
        selected_1, selected_2 = choose_questions(scores, options, self.embedding_provider, topic, round_id)
        if manual_select:
            selected_1 = self._select_candidate(candidates, scores, True, display)
            if options.questions_per_round == 2:
                remaining = [candidate for candidate in candidates if candidate.question != selected_1.question]
                remaining_scores = [score for score in scores if score.question != selected_1.question]
                selected_2 = self._select_candidate(remaining, remaining_scores, True, display)
        q1_text = selected_1.question
        q2_text = selected_2.question if selected_2 else None
        question_types = {candidate.question: candidate.type for candidate in candidates}
        checkpoint("questions_selected", candidate_scores=[score.model_dump() for score in scores],
                   selected_question=q1_text, selected_question_2=q2_text,
                   selected_question_type=question_types[q1_text], selected_question_2_type=question_types.get(q2_text))

        logger.info("Selected Q1: %s (I=%.3f)", q1_text, selected_1.I)
        if q2_text:
            logger.info("Selected Q2: %s (I=%.3f)", q2_text, selected_2.I)
        if display:
            print(f"[Student] Selected Q1: {q1_text} (I={selected_1.I:.3f})")
            if q2_text:
                print(f"[Student] Selected Q2: {q2_text} (I={selected_2.I:.3f})")

        # 合并两个问题一起提交
        if q2_text:
            combined_question = (
                f"(a) {q1_text}\n"
                f"(b) {q2_text}"
            )
        else:
            combined_question = q1_text

        # Re-select shared facts for the actual question(s), after selection.
        k_text = self._k_context(f"{topic} {combined_question}")

        # Instructor routes domain
        instructor_msgs_route = self.context.build_instructor_route_messages(
            topic=topic,
            shared_k_text=k_text,
            question=combined_question,
            history=self.history,
        )
        domain = self.instructor.route_domain(instructor_msgs_route)

        # Expert viewpoint (每个问题单独搜索)
        retrieved_context = None
        retrieval_sources = []
        if self.retriever is not None:
            max_chars = self.settings.orchestrator.retriever_context_max_chars
            all_snippets = []
            snippet_groups = []

            # Q1 搜索
            if self.settings.orchestrator.compact_context:
                q1_query_msgs = self.context.build_expert_query_messages(topic, q1_text, domain, self.history)
            else:
                q1_query_msgs = self.context.build_expert_messages(
                    topic=topic, shared_k_text=k_text, question=q1_text,
                    domain=domain, history=self.history,
                )
            q1_queries = self.expert.generate_queries(q1_query_msgs, num_queries=3)
            if q1_queries:
                logger.info("[Retriever] Q1 queries: %s", q1_queries)
                q1_snippets = self.retriever.search_multiple(q1_queries)
                all_snippets.extend(q1_snippets)
                snippet_groups.append(q1_snippets)
                checkpoint("first_question_retrieved", retrieval_sources=[vars(s).copy() for s in all_snippets])
                if display:
                    print(f"[Retriever] Q1: {len(q1_snippets)} snippets from {len(q1_queries)} queries")

            # Q2 搜索
            if q2_text:
                if self.settings.orchestrator.compact_context:
                    q2_query_msgs = self.context.build_expert_query_messages(topic, q2_text, domain, self.history)
                else:
                    q2_query_msgs = self.context.build_expert_messages(
                        topic=topic, shared_k_text=k_text, question=q2_text,
                        domain=domain, history=self.history,
                    )
                q2_queries = self.expert.generate_queries(q2_query_msgs, num_queries=3)
                if q2_queries:
                    logger.info("[Retriever] Q2 queries: %s", q2_queries)
                    q2_snippets = self.retriever.search_multiple(q2_queries)
                    all_snippets.extend(q2_snippets)
                    snippet_groups.append(q2_snippets)
                    if display:
                        print(f"[Retriever] Q2: {len(q2_snippets)} snippets from {len(q2_queries)} queries")

            if all_snippets:
                all_snippets = interleave_snippets(snippet_groups)
                retrieval_sources = [vars(s).copy() for s in all_snippets]
                retrieved_context = DuckDuckGoRetriever.format_snippets(all_snippets[:self.settings.orchestrator.retriever_max_snippets_per_turn], max_chars=max_chars)

        checkpoint("retrieval_finished", retrieval_sources=retrieval_sources, retrieved_context=retrieved_context)

        expert_msgs = self.context.build_expert_messages(
            topic=topic,
            shared_k_text=k_text,
            question=combined_question,
            domain=domain,
            history=self.history,
            retrieved_context=retrieved_context,
        )
        if options.separate_expert_instructor:
            expert_resp = (self.expert.respond(expert_msgs) if options.enable_expert_analysis
                           else ExpertResponse(viewpoint=retrieved_context or ""))
            checkpoint("expert_finished", expert_viewpoint=expert_resp.viewpoint)
            if display:
                print(f"[Expert] Domain: {domain}")
                print(f"[Expert] Viewpoint:\n{expert_resp.viewpoint}\n")

            # Instructor final answer（一起回答两个问题）
            instructor_msgs_final = self.context.build_instructor_teach_messages(
                topic=topic,
                shared_k_text=k_text,
                question=combined_question,
                history=self.history,
                expert_viewpoint=expert_resp.viewpoint,
            )
            instructor_answer = self.instructor.teach(instructor_msgs_final)
        else:
            # Same evidence, history and K; one combined analysis/explanation call.
            combined_messages = [{"role": msg["role"], "content": msg["content"].replace(
                'Return JSON: {"viewpoint": "..."}',
                'Return JSON with exactly two string fields: viewpoint and answer.')}
                for msg in expert_msgs]
            viewpoint, instructor_answer = self.instructor.analyze_and_teach(combined_messages)
            expert_resp = ExpertResponse(viewpoint=viewpoint)
            checkpoint("expert_finished", expert_viewpoint=viewpoint,
                       response_role="Expert-Instructor", combined_call=True)
        checkpoint("instructor_finished", instructor_answer=instructor_answer)
        if display:
            print(f"[Instructor] Answer:\n{instructor_answer}\n")

        # The learner-state ablation keeps Student question generation, but omits
        # the state-update model call and all learner-state storage access.
        ms_update = StudentMsUpdate()
        if self.student_pool is not None:
            ms_update_messages = self.context.build_student_update_messages(
                answer_text=instructor_answer,
                ms_text=self._ms_context(f"{topic} {combined_question}"),
                round_id=round_id,
            )
            try:
                ms_update = self.student.update_ms(ms_update_messages, round_id=round_id)
            except Exception as e:
                logger.warning("Student Ms update failed: %s", type(e).__name__)
                raise
            for bucket in ("known", "known_unknown", "conflict", "unknown_unknown"):
                for entry in getattr(ms_update, bucket):
                    entry.round_id = round_id
            self.student_pool.apply_update(ms_update)
        checkpoint("learner_state_finished", student_update=ms_update.model_dump())
        ms_text_after = self._ms_context(combined_question)

        # Arbiter updates shared K
        arbiter_msgs = self.context.build_arbiter_k_messages(
            topic=topic,
            shared_k_text=self._k_context(combined_question, for_arbiter=True),
            ms_text=ms_text_after,
            viewpoint=expert_resp.viewpoint,
            instructor_answer=instructor_answer,
            round_id=round_id,
            question=combined_question,
            guidance=guidance,
            evidence_refs=source_urls(retrieved_context or ""),
        )
        update_k = options.enable_arbiter and self.shared_pool is not None and options.shared_knowledge_mode == "update"
        k_update = self.arbiter.update_shared_pool(arbiter_msgs) if update_k else KnowledgePoolUpdate()
        for item in k_update.items:
            item.round_id = round_id
        self._validate_knowledge_sources(k_update, source_urls(retrieved_context or ""))
        new_k_state = self.shared_pool.apply_update(k_update) if update_k else (self.shared_pool.load() if self.shared_pool is not None else [])
        round_summary = prepare_summary(
            k_update.round_summary if self.settings.orchestrator.summary_mode == "arbiter" else None, round_id=round_id, question=combined_question,
            answer=instructor_answer, expert=expert_resp.viewpoint, guidance=guidance,
            evidence_refs=source_urls(retrieved_context or ""),
            max_chars=self.settings.orchestrator.round_summary_max_chars,
        )
        checkpoint("shared_state_finished", arbiter_update=k_update.model_dump(), round_summary=round_summary.model_dump())
        self.history.append(DialogueTurn(
            question=combined_question, instructor_answer=instructor_answer,
            expert_viewpoint=expert_resp.viewpoint, round_id=round_id, summary=round_summary,
            guidance=guidance, selected_questions=[q1_text] + ([q2_text] if q2_text else []),
        ))

        # Arbiter patches
        scores_text = self._summarize_scores(scores)
        ms_update_text = self._summarize_ms_update(ms_update) if self.student_pool is not None else None
        arbiter_patch_msgs = self.context.build_arbiter_patch_messages(
            topic=topic,
            question=combined_question,
            ms_update_text=ms_update_text,
            scores_text=scores_text,
            round_id=round_id,
        )
        patches = self.arbiter.generate_patch(arbiter_patch_msgs) if options.enable_arbiter and options.enable_auto_patches else ArbiterPatchResponse(patches=[])
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
            selected_question_type=question_types[q1_text],
            expert_viewpoint=expert_resp.viewpoint,
            instructor_answer=instructor_answer,
            selected_question_2=q2_text,
            selected_question_2_type=question_types.get(q2_text),
            expert_viewpoint_2=None,
            instructor_answer_2=None,
            student_update=ms_update,
            arbiter_update=k_update,
            arbiter_patch=patches,
            round_summary=round_summary,
            retrieved_context=retrieved_context,
            retrieval_sources=retrieval_sources,
            state_update_stats=getattr(self.student_pool, "last_update_stats", {}) if self.student_pool is not None else {},
            shared_update_stats=self.shared_pool.last_update_stats if update_k else {},
            context_policy=self._context_policy(),
            context_settings=self.settings.orchestrator.model_dump(include={
                "compact_context", "summary_mode", "history_context_max_chars", "final_history_max_chars",
                "pool_context_max_chars", "helper_context_max_chars", "round_summary_max_chars", "pool_context_max_items",
            }),
            execution_policy=self._execution_policy(),
        )
        if persist_log:
            persistence.append_jsonl(self.settings.paths.round_log_path, record.model_dump())
            checkpoint("complete", status="complete", arbiter_patch=patches.model_dump())

        logger.info("Round %s complete. Ms entries now: %s", round_id, len(self._flatten_ms_entries()))
        logger.info("Shared K size: %s", len(new_k_state))
        return record

    # ------------------------------------------------------------------ #

    def finalize(self, topic: str) -> str:
        ms_text = self._ms_context(topic)
        k_text = self._k_context(topic)

        history_text = self.context._format_history(self.history, self.settings.orchestrator.final_history_max_chars)
        ms_section = (
            "=== Student Knowledge Pool Ms ===\n" + ms_text + "\n\n"
            if ms_text is not None else ""
        )

        summary_prompt = (
            f"Topic: {topic}\n\n"
            "You are a tutor. Based on the supplied dialogue notes/history, knowledge pools, and expert research below, "
            "write a comprehensive final summary that covers ALL sub-topics discussed across all rounds.\n\n"
            "Requirements:\n"
            "- Cover every sub-topic that was discussed in the dialogue\n"
            "- Synthesize insights from the knowledge pools and expert research\n"
            "- Include key concepts, mechanisms, practical examples, and important takeaways\n"
            "- Mention unresolved questions or areas for further exploration\n"
            "- Preserve evidence references, numbers, units and limitations; do not invent omitted details\n"
            "- Write in a clear, educational style suitable for a learner\n"
            "- Aim for 800-1500 words\n\n"
            "=== Dialogue History ===\n"
            f"{history_text}\n\n"
            "=== Shared Knowledge Pool K ===\n"
            f"{k_text}\n\n"
            f"{ms_section}"
            "Write the final summary now. Output plain text only, no JSON."
        )
        summary = self.instructor.synthesize([{"role": "user", "content": summary_prompt}])
        logger.info("Final summary generated.")
        return summary

    def run_fixed_three_rounds(
        self,
        topic: str,
        round1_input: Optional[str],
        round2_input: Optional[str],
        display: bool = True,
        manual_select: bool = False,
        domain: Optional[str] = None,
    ) -> Dict[str, object]:
        self.bootstrap(topic, display=display, domain=domain)
        records: List[RoundRecord] = []
        records.append(
            self.run_round(
                topic=topic,
                round_id=1,
                guidance=round1_input,
                display=display,
                persist_log=True,
                manual_select=manual_select,
            )
        )
        records.append(
            self.run_round(
                topic=topic,
                round_id=2,
                guidance=round2_input,
                display=display,
                persist_log=True,
                manual_select=manual_select,
            )
        )
        records.append(
            self.run_round(
                topic=topic,
                round_id=3,
                guidance=None,
                display=display,
                persist_log=True,
                manual_select=manual_select,
            )
        )
        summary = self.finalize(topic)
        if display:
            print("\n=== Final Summary (Instructor) ===\n")
            print(summary)
        return {"topic": topic, "round_records": records, "final_summary": summary}

    def run_fixed_n_rounds(
        self,
        topic: str,
        round1_input: Optional[str],
        round2_input: Optional[str],
        num_rounds: int = 5,
        display: bool = True,
        manual_select: bool = False,
        domain: Optional[str] = None,
    ) -> Dict[str, object]:
        self.bootstrap(topic, display=display, domain=domain)
        records: List[RoundRecord] = []
        for round_id in range(1, num_rounds + 1):
            if round_id == 1:
                guidance = round1_input
            elif round_id == 2:
                guidance = round2_input
            else:
                guidance = None
            records.append(
                self.run_round(
                    topic=topic,
                    round_id=round_id,
                    guidance=guidance,
                    display=display,
                    persist_log=True,
                    manual_select=manual_select,
                )
            )
        summary = self.finalize(topic)
        if display:
            print("\n=== Final Summary ===\n")
            print(summary)
        return {"topic": topic, "round_records": records, "final_summary": summary}

    def run(self, topic: str, manual_select: bool = False) -> None:
        self.bootstrap(topic)
        round_id = 1
        while True:
            guidance = input("Optional guidance for this round (press Enter to skip): ").strip()
            guidance = guidance or None
            try:
                self.run_round(topic, round_id, guidance, manual_select=manual_select)
            except TavilyKeyPoolError:
                raise
            except Exception as e:
                if is_fatal_api_error(e):
                    raise
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
