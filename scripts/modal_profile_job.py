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
"""Modal entrypoint for running the Torch profiler remotely."""

from __future__ import annotations

import os
import pathlib
import subprocess

import modal


APP_NAME = "recurrentgemma-profiler"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REMOTE_REPO_PATH = pathlib.Path("/root/recurrentgemma")

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime")
    .pip_install("absl-py", "sentencepiece", "tensorboard")
)

repo_mount = modal.Mount.from_local_dir(
    REPO_ROOT,
    remote_path=str(REMOTE_REPO_PATH),
)


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    timeout=60 * 30,
    mounts=[repo_mount],
)
def run_remote_profiler(
    checkpoint_path: str | None = None,
    prompt_tokens: int = 4096,
    decode_steps: int = 256,
    batch_size: int = 1,
    dtype: str = "bfloat16",
    output_dir: str = "/root/modal_profile_artifacts",
    debug_model: bool = False,
) -> None:
  """Executes the local profiling script inside a Modal worker."""
  cmd = [
      "python",
      str(REMOTE_REPO_PATH / "scripts" / "profile_inference.py"),
      f"--prompt_tokens={prompt_tokens}",
      f"--decode_steps={decode_steps}",
      f"--batch_size={batch_size}",
      f"--dtype={dtype}",
      f"--output_dir={output_dir}",
      "--device=cuda",
  ]
  if checkpoint_path:
    cmd.append(f"--path_checkpoint={checkpoint_path}")
  if debug_model:
    cmd.append("--debug_model")

  env = os.environ.copy()
  env["PYTHONPATH"] = str(REMOTE_REPO_PATH)

  subprocess.run(
      cmd,
      check=True,
      cwd=REMOTE_REPO_PATH,
      env=env,
  )


@app.local_entrypoint()
def main(
    checkpoint_path: str | None = None,
    prompt_tokens: int = 4096,
    decode_steps: int = 256,
    batch_size: int = 1,
    dtype: str = "bfloat16",
    output_dir: str = "modal_profile_artifacts",
    debug_model: bool = False,
) -> None:
  """Convenience local entry-point: `modal run scripts/modal_profile_job.py`."""
  remote_output_dir = str(REMOTE_REPO_PATH / output_dir)
  run_remote_profiler.remote(
      checkpoint_path=checkpoint_path,
      prompt_tokens=prompt_tokens,
      decode_steps=decode_steps,
      batch_size=batch_size,
      dtype=dtype,
      output_dir=remote_output_dir,
      debug_model=debug_model,
  )
