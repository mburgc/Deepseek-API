"""Pydantic models for the OpenAI-compatible surface we accept and return.

The request model is permissive on purpose: OpenAI clients send plenty of
fields we do not use (`temperature`, `response_format`, `providerOptions`-style
extras), and rejecting them would break drop-in compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_MODEL


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    arguments: str = ""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    type: str = "function"
    index: Optional[int] = None
    function: FunctionCall


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: Optional[str] = None
    image_url: Optional[Union[str, Dict[str, Any]]] = None
    file: Optional[Dict[str, Any]] = None
    file_id: Optional[str] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, List[Any], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class Tool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL
    messages: List[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None

    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    conversation_id: Optional[str] = None
    thinking: Optional[bool] = None
    search: Optional[bool] = None

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def includes_usage(self) -> bool:
        options = self.stream_options or {}
        return bool(options.get("include_usage"))
