from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# Student candidate questions -------------------------------------------------


class CandidateQuestion(BaseModel):
    question: str
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
    known: List[StudentEntry] = Field(default_factory=list)
    known_unknown: List[StudentEntry] = Field(default_factory=list)
    conflict: List[StudentEntry] = Field(default_factory=list)
    unknown_unknown: List[StudentEntry] = Field(default_factory=list)


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

    @field_validator("source_role", mode="before")
    @classmethod
    def normalize_source_role(cls, v):
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Normalize common role aliases returned by language models.
            role_map = {
                "student": "instructor",
                "teacher": "instructor",
                "explorer": "expert",
                "researcher": "expert",
                "moderator": "arbiter",
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
        return v.strip()

    def ensure_ids(self) -> str:
        if not self.id:
            self.id = str(uuid.uuid4())
        ts = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = ts
        self.updated_at = ts
        return self.id


class KnowledgePoolUpdate(BaseModel):
    items: List[KnowledgeItem]


# Arbiter tuning patch -------------------------------------------------------


class AgentParamPatch(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


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
    expert_viewpoint: str
    instructor_answer: str
    # Optional second selected question.
    selected_question_2: Optional[str] = None
    expert_viewpoint_2: Optional[str] = None
    instructor_answer_2: Optional[str] = None
    student_update: StudentMsUpdate
    arbiter_update: KnowledgePoolUpdate
    arbiter_patch: ArbiterPatchResponse
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
