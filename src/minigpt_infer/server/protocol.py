"""Pydantic schemas mirroring the OpenAI API shapes (docs/PLAN.md §7 Phase 7
point 1). Only fields this server actually implements are included --
unsupported OpenAI fields are simply absent rather than accepted-and-ignored,
so a client sending them gets a clear 422 instead of silent non-behavior.

`seed` is accepted for shape-compatibility but not honored except at
temperature=0 (already deterministic): docs/ARCHITECTURE.md's engine notes
already document that per-request-seeded sampling isn't implemented (batched
multinomial sampling shares one RNG stream across the whole decode batch).
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]


class CompletionStreamChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None


class CompletionStreamChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamChoice]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]


class ChatCompletionStreamDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: str | None


class ChatCompletionStreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "minigpt-infer"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
