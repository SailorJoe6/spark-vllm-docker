#!/usr/bin/env python3
"""Extend CUDA sleep memory settling for the B12X vLLM source shape.

Level-2 sleep can trigger a Triton allocator/JIT transition after the initial
synchronization.  The existing vLLM check retries only on ROCm, so a transient
CUDA free-memory dip is reported as a permanent increase.  This opt-in,
source-guarded patch gives CUDA five seconds to settle, then keeps the original
assertion for a persistent increase.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FLAG_NAME = "VLLM_PATCH_SLEEP_MEMORY_SETTLE"
TARGET_REL = Path("vllm/v1/worker/gpu_worker.py")
OLD = "deadline = time.monotonic() + (5.0 if current_platform.is_rocm() else 0)"
NEW = "deadline = time.monotonic() + 5.0"
TRUE_VALUES = {"1", "true", "TRUE", "yes", "YES"}
FALSE_VALUES = {"", "0", "false", "FALSE", "no", "NO"}

raw_flag = os.environ.get(FLAG_NAME, "0")
if raw_flag not in TRUE_VALUES | FALSE_VALUES:
    raise SystemExit(f"Invalid {FLAG_NAME} value: {raw_flag!r}")
if raw_flag not in TRUE_VALUES:
    print("CUDA sleep memory-settle workaround not requested; skipping")
    raise SystemExit(0)

source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / TARGET_REL
if not target.exists():
    print(f"{TARGET_REL} is absent; CUDA sleep memory-settle workaround is not applicable")
    raise SystemExit(0)

text = target.read_text()
if NEW in text and OLD not in text:
    print("CUDA sleep memory-settle workaround is already applied; skipping")
    raise SystemExit(0)
if text.count(OLD) != 1:
    raise SystemExit(
        f"Expected one CUDA/ROCm sleep deadline, found {text.count(OLD)}; "
        "refusing to patch an unknown source shape"
    )
if "assert freed_bytes >= 0" not in text:
    raise SystemExit("Sleep memory assertion is absent; refusing to patch unknown source")
updated = text.replace(OLD, NEW, 1)
compile(updated, str(target), "exec")
target.write_text(updated)
print("Applied CUDA sleep memory-settle workaround")
