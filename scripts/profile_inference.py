#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Torch profiler entry point for RecurrentGemma inference.

This script runs a full prompt prefill followed by autoregressive decoding
while capturing a PyTorch profiler trace. The resulting trace can be opened
with TensorBoard for an end-to-end performance breakdown of the inference
stack.
"""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
from typing import Any

from absl import app
from absl import flags
from absl import logging
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function
from torch.profiler import tensorboard_trace_handler


_PATH_CHECKPOINT = flags.DEFINE_string(
    "path_checkpoint",
    None,
    "Optional path to a Torch checkpoint produced by the conversion utilities.",
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 1, "Batch size for profiling.")
_PROMPT_TOKENS = flags.DEFINE_integer(
    "prompt_tokens",
    4096,
    "Number of tokens used during the prefill stage.",
)
_DECODE_STEPS = flags.DEFINE_integer(
    "decode_steps",
    256,
    "Number of autoregressive tokens to decode.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "profiling_artifacts",
    "Directory used to write profiler traces and summaries.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    None,
    "Torch device to run on. Defaults to CUDA when available.",
)
_DTYPE = flags.DEFINE_enum(
    "dtype",
    "bfloat16",
    ("bfloat16", "float16", "float32"),
    "Computation dtype for the model weights.",
)
_DEBUG_MODEL = flags.DEFINE_bool(
    "debug_model",
    False,
    "Run a tiny Griffin config instead of the default 2B preset.",
)


def _resolve_device(device_flag: str | None) -> torch.device:
  if device_flag:
    return torch.device(device_flag)
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(name: str) -> torch.dtype:
  return {
      "bfloat16": torch.bfloat16,
      "float16": torch.float16,
      "float32": torch.float32,
  }[name]


def _import_recurrentgemma() -> Any:
  """Imports the RecurrentGemma torch module lazily."""
  from recurrentgemma import torch as recurrentgemma  # pylint: disable=import-error

  return recurrentgemma


def _load_model(
    *,
    checkpoint: str | None,
    debug: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
  recurrentgemma = _import_recurrentgemma()
  params: dict[str, torch.Tensor] | None = None
  if debug:
    logging.info("Using debug Griffin configuration.")
    config = recurrentgemma.GriffinConfig(
        vocab_size=512,
        width=128,
        mlp_expanded_width=384,
        lru_width=256,
        num_heads=4,
        block_types=(
            recurrentgemma.TemporalBlockType.RECURRENT,
            recurrentgemma.TemporalBlockType.ATTENTION,
        ),
        attention_window_size=2048,
        logits_soft_cap=30.0,
    )
  elif checkpoint:
    logging.info("Loading checkpoint from: %s", checkpoint)
    params = torch.load(checkpoint, map_location=device)
    logging.info("Checkpoint loaded. Building config from parameters.")
    config = recurrentgemma.GriffinConfig.from_torch_params(
        params, preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
    )
  else:
    logging.info("Falling back to Griffin 2B preset without weights.")
    config = recurrentgemma.GriffinConfig.from_preset(
        vocab_size=256_000,
        preset=recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1,
    )

  model = recurrentgemma.Griffin(config, device=device, dtype=dtype)
  if not debug and checkpoint and params is not None:
    model.load_state_dict(params)
  model.eval()
  return model, config


def _allocate_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  tokens = torch.randint(
      low=0,
      high=vocab_size,
      size=(batch_size, sequence_length),
      device=device,
      dtype=torch.long,
  )
  pos = torch.arange(sequence_length, device=device, dtype=torch.long)
  pos = pos.unsqueeze(0).expand(batch_size, -1).contiguous()
  return tokens, pos


def _build_profiler(output_dir: pathlib.Path,
                    include_cuda: bool) -> torch.profiler.profile:
  activities = [ProfilerActivity.CPU]
  if include_cuda:
    activities.append(ProfilerActivity.CUDA)
  handler = tensorboard_trace_handler(str(output_dir))
  return profile(
      activities=activities,
      record_shapes=True,
      profile_memory=True,
      with_stack=True,
      on_trace_ready=handler,
  )


def _profile_inference(
    *,
    model: Any,
    config: Any,
    batch_size: int,
    prompt_tokens: int,
    decode_steps: int,
    output_dir: pathlib.Path,
) -> None:
  tokens, segment_pos = _allocate_inputs(
      batch_size=batch_size,
      sequence_length=prompt_tokens,
      vocab_size=config.vocab_size,
      device=next(model.parameters()).device,
  )

  logging.info(
      "Starting profiling run: batch=%d prompt_tokens=%d decode_steps=%d",
      batch_size,
      prompt_tokens,
      decode_steps,
  )

  profiler = _build_profiler(output_dir, include_cuda=torch.cuda.is_available())
  with torch.no_grad(), profiler:
    with record_function("prefill"):
      logits, cache = model.forward(tokens=tokens, segment_pos=segment_pos)
    profiler.step()

    decode_pos = segment_pos[:, -1:] + 1
    for step in range(decode_steps):
      with record_function("decode_step"):
        step_logits = logits[:, -1].to(torch.float32)
        probs = torch.nn.functional.softmax(step_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        logits, cache = model(tokens=next_token, segment_pos=decode_pos + step, cache=cache)
      profiler.step()

  if tokens.device.type == "cuda":
    torch.cuda.synchronize()

  table_sort_key = (
      "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
  )
  summary_path = output_dir / "operator_summary.txt"
  summary = profiler.key_averages().table(sort_by=table_sort_key, row_limit=200)
  summary_path.write_text(summary)
  logging.info("Profiler run complete. Summary written to %s", summary_path)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  device = _resolve_device(_DEVICE.value)
  dtype = _resolve_dtype(_DTYPE.value)
  output_dir = pathlib.Path(_OUTPUT_DIR.value).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  model, config = _load_model(
      checkpoint=_PATH_CHECKPOINT.value,
      debug=_DEBUG_MODEL.value,
      device=device,
      dtype=dtype,
  )
  _profile_inference(
      model=model,
      config=config,
      batch_size=_BATCH_SIZE.value,
      prompt_tokens=_PROMPT_TOKENS.value,
      decode_steps=_DECODE_STEPS.value,
      output_dir=output_dir,
  )


if __name__ == "__main__":
  app.run(main)
