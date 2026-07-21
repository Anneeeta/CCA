from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import AliasChoices, BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings


class ScoringWeights(BaseModel):
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.4

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
    class AgentRuntimeConfig(BaseModel):
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

    chat_model: str = "deepseek-chat"
    # Kept for backward compatibility; set to chat by default to disable deep-thinking mode.
    reasoner_model: str = "deepseek-chat"
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 8192
    # Agent-level token caps. If unset, falls back to max_tokens.
    instructor_max_tokens: Optional[int] = None
    expert_max_tokens: Optional[int] = None
    # Per-agent runtime parameters for CCA mode.
    student: AgentRuntimeConfig = Field(default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.7))
    instructor: AgentRuntimeConfig = Field(default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.4))
    expert: AgentRuntimeConfig = Field(default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.2))
    arbiter: AgentRuntimeConfig = Field(default_factory=lambda: ModelConfig.AgentRuntimeConfig(temperature=0.0))

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
    num_candidates: int = 5
    top_k_similarity: int = 3
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    bootstrap: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    allow_llm_ranking_fallback: bool = True
    # Web retrieval settings
    enable_retrieval: bool = False
    retriever_type: str = "duckduckgo"  # "duckduckgo" or "tavily"
    retriever_max_results: int = 5
    retriever_max_snippets_per_turn: int = 5
    retriever_context_max_chars: int = 3000
    # Tavily settings
    tavily_api_url: str = "https://api.tavily.com/search"
    tavily_search_depth: str = "basic"  # "basic" or "advanced"


class Paths(BaseModel):
    data_dir: Path = Path("data")
    shared_pool_path: Path = Path("data/shared_pool.json")
    student_pool_path: Path = Path("data/student_pool.json")
    round_log_path: Path = Path("data/round_logs.jsonl")
    log_dir: Path = Path("logs")


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or optional YAML file.
    """

    deepseek_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CCA_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    )
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=AliasChoices("CCA_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"),
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    paths: Paths = Field(default_factory=Paths)
    request_timeout: int = 120
    max_retries: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


def _load_yaml_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings(config_path: str | os.PathLike | None = None) -> Settings:
    """
    Load settings. If config_path is provided, merge YAML with env overrides.
    """
    data = {}
    if config_path:
        yaml_path = Path(config_path)
        if yaml_path.exists():
            data = _load_yaml_settings(yaml_path)
    settings = Settings(**data)
    # Ensure directories exist
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
    return settings


def to_json(obj: BaseModel) -> str:
    return obj.model_dump_json(indent=2)
