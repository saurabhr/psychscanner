"""Unit tests for ChatMockModel.with_structured_output — regression test for the
NotImplementedError that broke any task card using a structured-output parser
(card- or trial-level) when validated/dry-run against the mock LLM."""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from psychscanner.memories.base.mock_llm import ChatMockModel, _mock_schema_instance


class _Likert(BaseModel):
    rating: Literal[1, 2, 3, 4, 5]


class _Recall(BaseModel):
    recalled_word: str
    confidence: Literal[1, 2, 3, 4, 5, 6]


class _Nested(BaseModel):
    inner: _Likert
    label: str = "given-default"


def _model() -> ChatMockModel:
    return ChatMockModel(repeat_buffer_length=10, model="mock-chat-model")


def test_mock_schema_instance_fills_literal_and_str():
    instance = _mock_schema_instance(_Recall)
    assert instance.recalled_word == "mock"
    assert instance.confidence == 1  # first Literal option


def test_mock_schema_instance_preserves_existing_default():
    instance = _mock_schema_instance(_Nested)
    assert instance.label == "given-default"
    assert instance.inner.rating == 1  # recurses into nested BaseModel


def test_with_structured_output_returns_parsed_instance():
    result = _model().with_structured_output(_Likert).invoke([HumanMessage(content="hi")])
    assert isinstance(result, _Likert)
    assert result.rating == 1


def test_with_structured_output_include_raw():
    result = (
        _model()
        .with_structured_output(_Recall, include_raw=True)
        .invoke([HumanMessage(content="hi")])
    )
    assert result["parsing_error"] is None
    assert isinstance(result["parsed"], _Recall)
    assert result["raw"].content  # real AIMessage from the mock's own _generate
