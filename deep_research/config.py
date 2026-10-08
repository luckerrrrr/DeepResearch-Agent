from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    llm_timeout: float
    reasoning_effort: str | None
    json_mode: bool
    judge_model: str
    judge_reasoning_effort: str | None
    serpapi_key: str | None
    search_hl: str
    search_gl: str | None
    embedding_provider: str
    embedding_model: str
    data_dir: Path
    kb_dir: Path


def load_settings(require_llm: bool = True) -> Settings:
    model = _env("LLM_MODEL_ID", "")
    api_key = _env("LLM_API_KEY", "")
    base_url = _env("LLM_BASE_URL", "")
    if require_llm:
        missing = [n for n, v in (("LLM_MODEL_ID", model), ("LLM_API_KEY", api_key), ("LLM_BASE_URL", base_url)) if not v]
        if missing:
            raise ValueError(f"缺少环境变量：{', '.join(missing)}。请参考 .env.example 配置 .env 文件。")

    provider = (_env("EMBEDDING_PROVIDER", "local") or "local").lower()
    default_embedding = "text-embedding-3-small" if provider == "openai" else "BAAI/bge-small-zh-v1.5"
    reasoning_effort = _env("LLM_REASONING_EFFORT")

    return Settings(
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_timeout=float(_env("LLM_TIMEOUT", "120")),
        reasoning_effort=reasoning_effort,
        json_mode=_env_bool("LLM_JSON_MODE", True),
        judge_model=_env("JUDGE_MODEL_ID", model),
        judge_reasoning_effort=_env("JUDGE_REASONING_EFFORT", reasoning_effort),
        serpapi_key=_env("SERPAPI_API_KEY"),
        search_hl=_env("SEARCH_HL", "zh-cn"),
        search_gl=_env("SEARCH_GL"),
        embedding_provider=provider,
        embedding_model=_env("EMBEDDING_MODEL", default_embedding),
        data_dir=Path(_env("DATA_DIR", str(ROOT / "data"))),
        kb_dir=Path(_env("KB_DIR", str(ROOT / "knowledge_base"))),
    )
