from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# Student candidate questions -------------------------------------------------

CandidateQuestionType = Literal["known_unknown", "conflict", "unknown_unknown"]


class CandidateQuestion(BaseModel):
    question: str
    # Missing only when reading legacy records or non-strict interactive input.
    # New experiment runs reject missing labels before scoring/downstream calls.
    type: Optional[CandidateQuestionType] = None
    rationale: Optional[str] = None

    @field_validator("question")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()


class CandidateQuestionsResponse(BaseModel):
    candidates: List[CandidateQuestion]

    @field_validator("candidates")
    @classmethod
    def non_empty(cls, v: List[CandidateQuestion]) -> List[CandidateQuestion]:
        if not v:
            raise ValueError("candidates cannot be empty")
        return v


# Student Ms update ----------------------------------------------------------


StudentBucketType = Literal["known", "known_unknown", "conflict", "unknown_unknown"]


class StudentEntry(BaseModel):
    id: Optional[str] = None
    type: StudentBucketType
    content: str
    round_id: int
    rationale: Optional[str] = None
    priority: Optional[int] = None

    @field_validator("content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()

    def ensure_id(self) -> str:
        if not self.id:
            self.id = str(uuid.uuid4())
        return self.id


class StudentMsUpdate(BaseModel):
    model_config = {"extra": "forbid"}
    known: List[StudentEntry] = Field(default_factory=list)
    known_unknown: List[StudentEntry] = Field(default_factory=list)
    conflict: List[StudentEntry] = Field(default_factory=list)
    unknown_unknown: List[StudentEntry] = Field(default_factory=list)
    # Explicit removals only; omitted entries remain in the active pool.
    remove_ids: List[str] = Field(default_factory=list)


# Shared knowledge pool K -----------------------------------------------------


class KnowledgeConflictTag(BaseModel):
    conflict_with: List[str]
    conflict_type: str
    note: Optional[str] = None


class KnowledgeItem(BaseModel):
    id: Optional[str] = None
    content: str
    label: str
    source_role: Literal["instructor", "expert", "arbiter"]
    round_id: int
    conflict_tag: Optional[KnowledgeConflictTag] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)

    @field_validator("source_role", mode="before")
    @classmethod
    def normalize_source_role(cls, v):
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Qwen 可能输出 student/explorer/teacher 等非法值
            role_map = {
                "student": "instructor",
                "teacher": "instructor",
                "explorer": "expert",
                "researcher": "expert",
                "moderator": "arbiter",
                "экспериментатор": "expert",
                "исследователь": "expert",
                "преподаватель": "instructor",
            }
            return role_map.get(v_lower, v_lower)
        return v

    @classmethod
    def model_validate(cls, obj):
        """
        Accept lenient field aliases from LLM outputs, e.g., 'source', 'labels', 'round'.
        """
        if isinstance(obj, dict):
            data = dict(obj)
            if "label" not in data and isinstance(data.get("labels"), list) and data["labels"]:
                data["label"] = data["labels"][0]
            if "source_role" not in data and "source" in data:
                src = data["source"]
                if isinstance(src, str):
                    data["source_role"] = src
            if "round_id" not in data:
                if "round" in data:
                    data["round_id"] = data["round"]
                elif "roundid" in data:
                    data["round_id"] = data["roundid"]
            obj = data
        return super().model_validate(obj)

    @field_validator("content", "label")
    @classmethod
    def strip_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Knowledge text and labels cannot be empty")
        return v.strip()

    def ensure_ids(self) -> str:
        if not self.id:
            self.id = str(uuid.uuid4())
        ts = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = ts
        self.updated_at = ts
        return self.id


class RoundSummary(BaseModel):
    """Public dialogue notes, separate from private learner-state interpretations."""

    round_id: int
    questions: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    resolved_questions: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    guidance: Optional[str] = None
    origin: Literal["arbiter", "extractive_fallback"] = "arbiter"


class KnowledgePoolUpdate(BaseModel):
    model_config = {"extra": "forbid"}
    items: List[KnowledgeItem] = Field(default_factory=list)
    remove_ids: List[str] = Field(default_factory=list)
    round_summary: Optional[RoundSummary] = None

    @field_validator("round_summary", mode="before")
    @classmethod
    def optional_summary_does_not_invalidate_knowledge(cls, value):
        if value is None or isinstance(value, RoundSummary):
            return value
        try:
            return RoundSummary.model_validate(value)
        except (ValueError, TypeError):
            # The runner creates a marked local excerpt instead of retrying a
            # paid model call solely to repair auxiliary summary formatting.
            return None


# Arbiter tuning patch -------------------------------------------------------


class AgentParamPatch(BaseModel):
    model_config = {"allow_inf_nan": False}
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class AgentPatch(BaseModel):
    agent: Literal["student", "instructor", "expert", "arbiter"]
    prompt_patch: Optional[str] = None
    param_patch: Optional[AgentParamPatch] = None

    @field_validator("agent", mode="before")
    @classmethod
    def normalize_agent(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v
    rationale: Optional[str] = None


class ArbiterPatchResponse(BaseModel):
    patches: List[AgentPatch]


# Round log ------------------------------------------------------------------


class CandidateScore(BaseModel):
    model_config = {"allow_inf_nan": False}
    question: str
    s: float
    r: float
    novel: float
    I: float


class LLMRankingEntry(BaseModel):
    question: str
    I: float


class LLMRankingResponse(BaseModel):
    scores: List[LLMRankingEntry]


# Instructor / Expert structured outputs ------------------------------------


class InstructorResponse(BaseModel):
    domain: str
    intent: Optional[str] = None
    answer: str


class ExpertResponse(BaseModel):
    viewpoint: str


class RoundRecord(BaseModel):
    topic: str
    round_id: int
    user_guidance: Optional[str] = None
    candidate_questions: List[CandidateQuestion]
    candidate_scores: List[CandidateScore]
    selected_question: str
    selected_question_type: Optional[CandidateQuestionType] = None
    expert_viewpoint: str
    instructor_answer: str
    # 第二题（可选）
    selected_question_2: Optional[str] = None
    selected_question_2_type: Optional[CandidateQuestionType] = None
    expert_viewpoint_2: Optional[str] = None
    instructor_answer_2: Optional[str] = None
    student_update: StudentMsUpdate
    arbiter_update: KnowledgePoolUpdate
    arbiter_patch: ArbiterPatchResponse
    round_summary: Optional[RoundSummary] = None
    retrieved_context: Optional[str] = None
    retrieval_sources: List[dict] = Field(default_factory=list)
    state_update_stats: dict = Field(default_factory=dict)
    shared_update_stats: dict = Field(default_factory=dict)
    context_policy: Optional[str] = None
    context_settings: dict = Field(default_factory=dict)
    execution_policy: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
