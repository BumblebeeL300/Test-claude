"""NIM Lab core: async NVIDIA NIM client, mock backend, model registry, tool loop, utilities."""

from __future__ import annotations

from .client import NIMClient
from .config import NIMConfig
from .errors import (
    NIMAuthError,
    NIMError,
    NIMJSONError,
    NIMModelUnavailableError,
    NIMRateLimitError,
    NIMResponseError,
    NIMTimeoutError,
)
from .logging import get_console
from .registry import (
    ModelInfo,
    ModelRegistry,
    Role,
    default_registry,
    model_scope,
    set_default_registry,
)
from .results import ChatResult, RerankResult, ToolCall, Usage
from .text import chunk_text, cosine, extract_json, strip_think
from .tools import AgentEvent, AgentRun, AgentStep, Tool, ToolResult, run_agent, tool
from .usage import UsageTracker

__all__ = [
    "AgentEvent",
    "AgentRun",
    "AgentStep",
    "ChatResult",
    "ModelInfo",
    "ModelRegistry",
    "NIMAuthError",
    "NIMClient",
    "NIMConfig",
    "NIMError",
    "NIMJSONError",
    "NIMModelUnavailableError",
    "NIMRateLimitError",
    "NIMResponseError",
    "NIMTimeoutError",
    "RerankResult",
    "Role",
    "Tool",
    "ToolCall",
    "ToolResult",
    "Usage",
    "UsageTracker",
    "chunk_text",
    "cosine",
    "default_registry",
    "extract_json",
    "get_console",
    "model_scope",
    "run_agent",
    "set_default_registry",
    "strip_think",
    "tool",
]
