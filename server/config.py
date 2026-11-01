"""Server configuration: model aliases, limits and tunables.

DeepSeek's web backend has two orthogonal axes that OpenAI's single `model`
string cannot express: the model itself (`default` / `expert` / `vision`) and
the `thinking` / `search` toggles. The bridge therefore exposes one alias per
combination, e.g. `deepseek-expert-think-search`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 120)
RATE_LIMIT_EXEMPT_LOCALHOST = _env_bool("RATE_LIMIT_EXEMPT_LOCALHOST", True)

SERVER_INTERACTIVE_LOGIN = _env_bool("SERVER_INTERACTIVE_LOGIN", False)
SESSION_PREWARM = _env_bool("SESSION_PREWARM", True)

UPSTREAM_SERIALIZE = _env_bool("UPSTREAM_SERIALIZE", True)
THREAD_TTL_HOURS = _env_int("THREAD_TTL_HOURS", 12)
MAX_TOOL_RESULT_CHARS = _env_int("MAX_TOOL_RESULT_CHARS", 16000)
APPEND_SOURCES = _env_bool("APPEND_SOURCES", True)

DEFAULT_CONTEXT = _env_int("DEFAULT_CONTEXT", 800000)
DEFAULT_OUTPUT = _env_int("DEFAULT_OUTPUT", 32768)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    model_type: Optional[str]
    thinking: bool
    search: bool
    vision: bool
    label: str
    context: int = DEFAULT_CONTEXT
    output: int = DEFAULT_OUTPUT


def _variants(prefix: str, model_type: str, label: str, vision: bool = False) -> Dict[str, ModelSpec]:
    combos = (
        ("", False, False, ""),
        ("-think", True, False, " + Think"),
        ("-search", False, True, " + Web"),
        ("-think-search", True, True, " + Think + Web"),
    )
    out: Dict[str, ModelSpec] = {}
    for suffix, thinking, search, extra in combos:
        out[prefix + suffix] = ModelSpec(
            id=prefix + suffix,
            model_type=model_type,
            thinking=thinking,
            search=search,
            vision=vision,
            label=label + extra,
        )
    return out


MODELS: Dict[str, ModelSpec] = {}
MODELS.update(_variants("deepseek-chat", "default", "DeepSeek Chat"))
MODELS.update(_variants("deepseek-expert", "expert", "DeepSeek Expert"))
MODELS["deepseek-vision"] = ModelSpec(
    id="deepseek-vision",
    model_type="vision",
    thinking=False,
    search=False,
    vision=True,
    label="DeepSeek Vision",
)

ALIASES: Dict[str, str] = {
    "deepseek-reasoner": "deepseek-chat-think",
    "deepseek-r1": "deepseek-chat-think",
    "deepseek": "deepseek-chat",
}

DEFAULT_MODEL = "deepseek-chat"

MODEL_MAP: Dict[str, ModelSpec] = MODELS


def resolve_model(name: Optional[str]) -> ModelSpec:
    """Resolve an OpenAI-facing model id (or alias) to its spec."""
    key = (name or DEFAULT_MODEL).strip()
    key = ALIASES.get(key, key)
    spec = MODELS.get(key)
    if spec is None:
        raise KeyError(key)
    return spec


def is_known_model(name: Optional[str]) -> bool:
    try:
        resolve_model(name)
    except KeyError:
        return False
    return True


def resolve_model_type(name: str) -> Optional[str]:
    return resolve_model(name).model_type


