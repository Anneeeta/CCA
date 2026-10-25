from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import AliasChoices, BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings


class ScoringWeights(BaseModel):
    model_config = {"allow_inf_nan": False, "extra": "forbid"}
    alpha: float = 0.3
    beta: float = 0.3
    # Validate the sum even when gamma is supplied through its default.
    gamma: float = Field(default=0.4, validate_default=True)

    @field_validator("alpha", "beta", "gamma")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weights must be non-negative")
        return v

    @field_validator("gamma")
    @classmethod
    def sum_to_one(cls, v: float, info: ValidationInfo) -> float:
        alpha = info.data.get("alpha", 0.0)
        beta = info.data.get("beta", 0.0)
        total = alpha + beta + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError("alpha + beta + gamma must sum to 1")
        return v


class ModelConfig(BaseModel):
    model_config = {"allow_inf_nan": False, "extra": "forbid"}
    class AgentRuntimeConfig(BaseModel):
        model_config = {"allow_inf_nan": False, "extra": "forbid"}
        model: Optional[str] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        max_tokens: Optional[int] = None

        @field_validator("temperature")
        @classmethod
        def valid_temperature(cls, value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            if value < 0:
                raise ValueError("temperature must be >= 0")
            return value

        @field_validator("top_p")
        @classmethod
        def valid_top_p(cls, value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            if value <= 0 or value > 1:
                raise ValueError("top_p must be in (0, 1]")
            return value

        @field_validator("max_tokens")
        @classmethod
        def valid_max_tokens(cls, value: Optional[int]) -> Optional[int]:
            if value is None:
                return None
            if value <= 0:
                raise ValueError("max_tokens must be > 0")
            return value

    chat_model: str = "deepseek-flash"
    # Kept for backward compatibility; set to chat by default to disable deep-thinking mode.
    reasoner_model: str = "deepseek-flash"
    temperature: float = Field(default=0.6, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int = Field(default=81920, gt=0)
    # Agent-level token caps. If unset, falls back to max_tokens.
    instructor_max_tokens: Optional[int] = Field(default=81920, gt=0)
    expert_max_tokens: Optional[int] = Field(default=81920, gt=0)
    # Per-agent runtime parameters for CCA mode.
    # Batch evaluators can also use these overrides.
    student: AgentRuntimeConfig = Field(
        default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.7, max_tokens=81920)
    )
    instructor: AgentRuntimeConfig = Field(default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.4))
    expert: AgentRuntimeConfig = Field(default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.2))
    arbiter: AgentRuntimeConfig = Field(
        default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.0, max_tokens=81920)
    )
    evaluator: AgentRuntimeConfig = Field(default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.0, max_tokens=81920))

    def get_agent_runtime(self, agent_name: str) -> AgentRuntimeConfig:
        runtime = getattr(self, agent_name, None)
        if isinstance(runtime, self.AgentRuntimeConfig):
            return runtime
        return self.AgentRuntimeConfig()

    def get_agent_model(self, agent_name: str) -> Optional[str]:
        runtime = self.get_agent_runtime(agent_name)
        return runtime.model

    def get_agent_param_overrides(self, agent_name: str) -> dict[str, float | int]:
        runtime = self.get_agent_runtime(agent_name)
        overrides: dict[str, float | int] = {}
        if runtime.temperature is not None:
            overrides["temperature"] = runtime.temperature
        if runtime.top_p is not None:
            overrides["top_p"] = runtime.top_p
        if runtime.max_tokens is not None:
            overrides["max_tokens"] = runtime.max_tokens
        # Backward-compatible max_tokens fields (legacy config).
        if "max_tokens" not in overrides:
            if agent_name == "instructor" and self.instructor_max_tokens is not None:
                overrides["max_tokens"] = int(self.instructor_max_tokens)
            elif agent_name == "expert" and self.expert_max_tokens is not None:
                overrides["max_tokens"] = int(self.expert_max_tokens)
        return overrides


class OrchestratorConfig(BaseModel):
    model_config = {"allow_inf_nan": False, "extra": "forbid"}
    num_candidates: int = Field(default=5, ge=1)
    # Common ablation trials require the declared count; interactive/legacy
    # callers may retain their variable-size parsing fallback.
    require_candidate_count: bool = False
    # Retain Student question planning while disabling its learner-state pool.
    enable_student_pool: bool = True
    enable_student_questions: bool = True
    student_candidate_state: bool = True
    selection_policy: Literal["utility", "random"] = "utility"
    selection_seed: int = 17
    questions_per_round: int = Field(default=2, ge=1, le=2)
    selection_diversity: bool = True
    score_use_state: bool = True
    score_use_guidance: bool = True
    score_use_novelty: bool = True
    enable_arbiter: bool = True
    enable_auto_patches: bool = True
    shared_knowledge_mode: Literal["update", "frozen", "off"] = "update"
    use_shared_context: bool = True
    enable_expert_analysis: bool = True
    separate_expert_instructor: bool = True
    deduplicate_student_pool: bool = True
    deduplicate_shared_pool: bool = True
    # Section-size limits are characters, not measured token guarantees.
    compact_context: bool = True
    history_mode: Literal["summary", "raw"] = "summary"
    # Use extractive for every arm when isolating Arbiter from compression.
    summary_mode: Literal["arbiter", "extractive"] = "arbiter"
    history_context_max_chars: int = Field(default=6500, ge=800)
    final_history_max_chars: int = Field(default=10000, ge=800)
    pool_context_max_chars: int = Field(default=5000, ge=500)
    helper_context_max_chars: int = Field(default=1800, ge=300)
    round_summary_max_chars: int = Field(default=2200, ge=600)
    pool_context_max_items: int = Field(default=12, ge=4)
    top_k_similarity: int = Field(default=3, ge=1)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    bootstrap: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    allow_llm_ranking_fallback: bool = True
    # Web retrieval settings
    enable_retrieval: bool = False
    retriever_type: Literal["duckduckgo", "tavily", "deepseek"] = "duckduckgo"
    retriever_max_results: int = Field(default=5, ge=1, le=50)
    retriever_max_snippets_per_turn: int = Field(default=5, ge=1)
    retriever_context_max_chars: int = Field(default=3000, ge=1)
    # Tavily settings
    tavily_api_url: str = "https://api.tavily.com/search"
    tavily_search_depth: str = "basic"  # "basic" or "advanced"
    # Independent official Search API; does not change the answer model.
    deepseek_search_need_content: bool = False
    deepseek_search_timeout: float = Field(default=20.0, gt=0)

    @model_validator(mode="after")
    def coherent_component_switches(self):
        if self.enable_student_questions and self.num_candidates < self.questions_per_round:
            raise ValueError("num_candidates must cover questions_per_round.")
        if not self.enable_student_questions and self.enable_student_pool:
            raise ValueError("Disabling the Student subsystem also requires enable_student_pool=false.")
        if not self.enable_arbiter and (self.enable_auto_patches or self.shared_knowledge_mode != "off"):
            raise ValueError("Without Arbiter, disable auto patches and set shared_knowledge_mode=off.")
        if (not self.enable_arbiter or self.shared_knowledge_mode != "update") and self.summary_mode == "arbiter":
            raise ValueError("Disabled/frozen K updates require the common extractive summary policy.")
        return self


class Paths(BaseModel):
    model_config = {"extra": "forbid"}
    data_dir: Path = Path("data")
    shared_pool_path: Path = Path("data/shared_pool.json")
    student_pool_path: Path = Path("data/student_pool.json")
    round_log_path: Path = Path("data/round_logs.jsonl")
    log_dir: Path = Path("logs")

    @model_validator(mode="after")
    def distinct_storage_targets(self):
        files = [path.expanduser().resolve() for path in (self.shared_pool_path, self.student_pool_path, self.round_log_path)]
        identities = [os.path.normcase(str(path)) for path in files]
        if len(set(identities)) != len(identities):
            raise ValueError("Knowledge pools and round logs require distinct files")
        directories = [self.data_dir.expanduser().resolve(), self.log_dir.expanduser().resolve()]
        if any(file == directory or file in directory.parents for file in files for directory in directories):
            raise ValueError("Storage files cannot also be runtime/log directories or their parents")
        if any(first in second.parents for first in files for second in files if first != second):
            raise ValueError("Storage file paths cannot contain one another")
        return self


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or optional YAML file.
    """

    # Search must not fall back to proxy/legacy GONGYI_API_KEY or CCA_API_KEY.
    deepseek_search_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("deepseek_search_api_key", "DEEPSEEK_SEARCH_API_KEY", "DEEPSEEK_API_KEY"),
        repr=False, exclude=True,
    )
    deepseek_official_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("deepseek_official_api_key", "DEEPSEEK_JUDGE_API_KEY", "DEEPSEEK_API_KEY"),
        repr=False, exclude=True,
    )
    deepseek_enable_thinking: bool = False
    # This legacy credential is used only by unrecognized/custom model routes.
    deepseek_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("deepseek_api_key", "GONGYI_API_KEY", "DEEPSEEK_API_KEY", "CCA_API_KEY", "OPENAI_API_KEY"),
        repr=False,
        exclude=True,
    )
    mimo_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("mimo_api_key", "MIMO_API_KEY"), repr=False, exclude=True
    )
    siliconflow_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("siliconflow_api_key", "SILICONFLOW_API_KEY"), repr=False, exclude=True
    )
    openrouter_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("openrouter_api_key", "OPENROUTER_API_KEY"), repr=False, exclude=True
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_enable_thinking: bool = False
    openrouter_provider_only: list[str] = Field(default_factory=list)
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    # Disable thinking explicitly so configured sampling parameters remain effective.
    mimo_enable_thinking: bool = False
    siliconflow_enable_thinking: bool = False
    siliconflow_reasoning_effort: Optional[Literal["high", "max"]] = None
    tavily_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("tavily_api_key", "TAVILY_API_KEY"), repr=False, exclude=True)
    tavily_api_keys: list[str] = Field(
        default_factory=list, validation_alias=AliasChoices("tavily_api_keys", "TAVILY_API_KEYS"),
        repr=False, exclude=True,
    )
    tavily_key_call_limit: int = Field(default=950, gt=0)
    tavily_strict_errors: bool = False
    tavily_usage_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "tavily_key_usage.sqlite3")

    @field_validator("tavily_api_keys")
    @classmethod
    def valid_tavily_keys(cls, keys: list[str]) -> list[str]:
        normalized = [key.strip() for key in keys]
        if any(not key for key in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("Tavily keys must be non-empty and distinct")
        return normalized

    bing_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("bing_api_key", "BING_API_KEY"))
    # Used only for custom/legacy models; known providers have separate routes.
    base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=AliasChoices("base_url", "CCA_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"),
    )
    tavily_base_url: str = "https://api.tavily.com/search"
    bing_base_url: str = "https://api.bing.microsoft.com/v7.0/search"
    model: ModelConfig = Field(default_factory=ModelConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    paths: Paths = Field(default_factory=Paths)
    request_timeout: int = Field(default=120, gt=0)
    max_retries: int = Field(default=5, ge=1)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


def _load_yaml_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Configuration YAML must contain a mapping")
    return data


@lru_cache
def get_settings(config_path: str | os.PathLike | None = None) -> Settings:
    """
    Load explicit YAML settings, then fill unspecified fields from the environment.
    """
    data = {}
    if config_path:
        yaml_path = Path(config_path)
        data = _load_yaml_settings(yaml_path)
    settings = Settings(**data)
    # Ensure directories exist
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
    return settings


def to_json(obj: BaseModel) -> str:
    return obj.model_dump_json(indent=2)
