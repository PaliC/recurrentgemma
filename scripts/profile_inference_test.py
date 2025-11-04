"""Unit tests for scripts.profile_inference."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from scripts import profile_inference


def test_resolve_device_prefers_flag(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  device = profile_inference._resolve_device("cpu")
  assert device.type == "cpu"


def test_resolve_device_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  device = profile_inference._resolve_device(None)
  assert device.type == "cpu"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ],
)
def test_resolve_dtype(name: str, expected: torch.dtype) -> None:
  assert profile_inference._resolve_dtype(name) is expected


def test_allocate_inputs_returns_expected_shapes() -> None:
  tokens, positions = profile_inference._allocate_inputs(
      batch_size=2,
      sequence_length=5,
      vocab_size=11,
      device=torch.device("cpu"),
  )
  assert tokens.shape == (2, 5)
  assert positions.shape == (2, 5)
  assert tokens.dtype == torch.long
  assert positions.dtype == torch.long
  assert torch.all(positions[0] == torch.arange(5))
  assert torch.all(positions[1] == torch.arange(5))


class _DummyKeyAverages:
  def __init__(self) -> None:
    self.calls: list[tuple[str, int]] = []

  def table(self, sort_by: str, row_limit: int) -> str:
    self.calls.append((sort_by, row_limit))
    return "dummy-summary"


class _DummyProfiler:
  def __init__(self) -> None:
    self.steps = 0
    self.key_averages_obj = _DummyKeyAverages()

  def __enter__(self) -> "_DummyProfiler":
    return self

  def __exit__(self, exc_type, exc, exc_tb) -> bool:
    return False

  def step(self) -> None:
    self.steps += 1

  def key_averages(self) -> _DummyKeyAverages:
    return self.key_averages_obj


@dataclass
class _DummyConfig:
  vocab_size: int


class _DummyModel(torch.nn.Module):
  def __init__(self, vocab_size: int):
    super().__init__()
    self.config = _DummyConfig(vocab_size=vocab_size)
    self.register_parameter("dummy", torch.nn.Parameter(torch.ones(1)))

  def forward(self, tokens, segment_pos, cache=None):
    batch, seq = tokens.shape
    logits = torch.zeros(batch, seq, self.config.vocab_size, device=tokens.device)
    next_cache = {"segment_pos": segment_pos}
    return logits, next_cache


def test_profile_inference_writes_summary(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  dummy_profiler = _DummyProfiler()
  monkeypatch.setattr(
      profile_inference,
      "_build_profiler",
      lambda *args, **kwargs: dummy_profiler,
  )

  model = _DummyModel(vocab_size=32)
  config = _DummyConfig(vocab_size=32)

  profile_inference._profile_inference(
      model=model,
      config=config,
      batch_size=1,
      prompt_tokens=4,
      decode_steps=3,
      output_dir=tmp_path,
  )

  assert dummy_profiler.steps == 4  # 1 prefill + 3 decode steps.
  summary_path = tmp_path / "operator_summary.txt"
  assert summary_path.read_text() == "dummy-summary"
  assert dummy_profiler.key_averages_obj.calls == [("self_cpu_time_total", 200)]
