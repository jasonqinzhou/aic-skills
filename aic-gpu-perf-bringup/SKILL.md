---
name: aic-gpu-perf-bringup
description: Use when running on a GPU node to collect AIC/aiconfigurator perf data for an existing or new GPU type, fix collector errors autonomously, recollect until data quality is acceptable, verify AIC can use the new perf files, and prepare a GitHub PR to ai-dynamo/aiconfigurator.
---

# AIC GPU Perf Bring-Up

## Goal

You are already on a GPU node with one or more GPUs. Your job is to collect AIC perf data for a target GPU/system/backend/version, fix collector failures, recollect until the perf files are good enough, verify AIC can consume the new data, and prepare a PR to `ai-dynamo/aiconfigurator`.

This is a long-running workflow. Be persistent. Iterate carefully. Do not hide failures by shrinking coverage or deleting hard cases unless the configuration is genuinely unsupported and documented.

## Required Inputs

Establish these before collecting:

- Target AIC repo and branch.
- Target backend and version: `sglang`, `vllm`, or `trtllm`.
- Target system name, such as `rtxpro6000_blackwell_server`.
- Target ops: start narrow, then expand to all relevant registry ops.
- Whether to collect power columns.
- Whether the GPU type is already supported by AIC.

Discover the node:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv
python3 - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

## Phase 1: Prepare Repo

1. Clone or update `ai-dynamo/aiconfigurator`.
2. Create a branch named like `data/<system>-<backend>-<version>`.
3. Install only lightweight local dev dependencies needed for tests. Do not install the target backend locally unless the collector is not running in a backend container.
4. Set:

```bash
export PYTHONPATH="$PWD"
export COLLECTOR_LOG_DIR="$PWD/collector_logs"
mkdir -p "$COLLECTOR_LOG_DIR"
```

## Phase 2: Add Or Validate System Metadata

If the system is new or incomplete:

1. Add `src/aiconfigurator/systems/<system>.yaml`.
2. Add `<system>` to `SupportedSystems` in `src/aiconfigurator/sdk/common.py`.
3. Create `src/aiconfigurator/systems/data/<system>/`.
4. Populate YAML with conservative, documented values:
   - `gpu.mem_bw`
   - `gpu.mem_capacity`
   - tensor core FLOPS fields relevant to the architecture
   - `gpu.power`
   - `gpu.sm_version`
   - `node.num_gpus_per_node`
   - inter-node, intra-node, PCIe bandwidth
   - `misc.nccl_version`
   - `misc.other_mem`

Prefer verified values from `nvidia-smi`, node docs, or nearby existing YAML files. If a value is an estimate, leave a comment.

Run:

```bash
pytest tests/unit/sdk/test_common.py -q
```

## Phase 3: Choose Collection Plan

Read the backend registry:

```bash
python3 - <<'PY'
from collector.sglang.registry import REGISTRY as SGLANG
from collector.vllm.registry import REGISTRY as VLLM
from collector.trtllm.registry import REGISTRY as TRTLLM
for name, reg in [("sglang", SGLANG), ("vllm", VLLM), ("trtllm", TRTLLM)]:
    print(name, [e.op for e in reg])
PY
```

Recommended order:

1. `gemm`
2. `moe`
3. attention / MLA ops
4. module-level ops such as DSA, GDN, MHC, WideEP
5. communication ops separately, because they may require multi-GPU or full-node ownership

For new GPUs, prioritize ops used by the target models and backend first, but do not claim full support until all expected ops for that backend/version are collected or explicitly marked unsupported.

## Phase 4: Progressive Collection Loop

For each op:

```bash
python3 collector/collect.py --backend <backend> --ops <op> --smoke
python3 collector/collect.py --backend <backend> --ops <op> --shuffle --limit 20
python3 collector/collect.py --backend <backend> --ops <op> --shuffle --limit 100
python3 collector/collect.py --backend <backend> --ops <op> --resume
```

Use a separate log directory per op when iterating:

```bash
export COLLECTOR_LOG_DIR="$PWD/collector_logs/<backend>/<version>/<op>"
mkdir -p "$COLLECTOR_LOG_DIR"
```

If the process times out or crashes after partial progress, rerun with `--resume`. Keep checkpoints until the op is accepted.

## Phase 5: Fix Collector Errors

When a run fails, inspect:

- `collection_summary_<backend>.json`
- `errors_*.json`
- worker logs in `COLLECTOR_LOG_DIR`
- traceback module and task parameters

Classify each error group:

- **collector_bug**: import/API/mock object/signature/test generation issue.
- **unsupported_config**: generated case is invalid for this SM/backend/kernel.
- **framework_bug**: backend kernel crashes or rejects a valid production-like case.
- **resource_issue**: OOM, shared memory, graph capture, worker restart, timeout.
- **transient**: rare cache/race/infra issue that passes on retry.

Fix policy:

- For API changes, use version routing and `__compat__` where needed.
- For GPU capability gaps, filter test cases before execution using `get_sm_version()`.
- For Blackwell/SM100+ features, gate FP4/NVFP4/FP8 paths carefully.
- For dimension constraints, filter on per-rank dimensions such as `inter_size // tp`.
- For CUDA-fatal errors, prefer worker restart and skip only the exact invalid region.
- Preserve data quality. Do not reduce benchmark repetitions or broad test dimensions just to pass.
- Keep older backend versions working.

After a fix:

```bash
pytest tests/unit/collector -q
python3 collector/collect.py --backend <backend> --ops <op> --smoke
python3 collector/collect.py --backend <backend> --ops <op> --shuffle --limit 100
python3 collector/collect.py --backend <backend> --ops <op> --resume
```

Commit small fixes:

```bash
git add collector tests
git commit -m "fix <backend> <op> collector for <system>"
```

## Phase 6: Perf File Quality Gates

Accept a perf file only when:

- It exists under `src/aiconfigurator/systems/data/<system>/<backend>/<version>/`.
- It is not empty and has the expected CSV header.
- Rows contain the expected framework, version, and device name.
- Latency values are positive, finite, and plausible.
- Power values, when collected, are positive and below/near the configured power limit.
- There are no unexplained duplicate rows for identical keys.
- Coverage is comparable to the nearest existing system/backend/version.
- Missing cases are explained by documented unsupported config filters.
- Re-running sample does not produce large unexplained variance.

Useful checks:

```bash
find src/aiconfigurator/systems/data/<system>/<backend>/<version> -maxdepth 1 -type f -name '*_perf.txt' -print
python3 - <<'PY'
from pathlib import Path
import csv, math
root = Path("src/aiconfigurator/systems/data/<system>/<backend>/<version>")
for path in sorted(root.glob("*_perf.txt")):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if "latency" in r and (not r["latency"] or not math.isfinite(float(r["latency"])) or float(r["latency"]) <= 0)]
    print(path.name, "rows", len(rows), "bad_latency", len(bad))
PY
```

Compare row counts and latency ranges with nearby systems, such as H100/H200 for Hopper or B200/B300 for Blackwell.

## Phase 7: Verify AIC Consumption

Run loader and SDK tests:

```bash
pytest tests/unit/sdk/test_common.py -q
pytest tests/unit/sdk/database -q
pytest tests/unit/collector -q
```

Instantiate the database:

```bash
python3 - <<'PY'
from aiconfigurator.sdk.perf_database import get_database
db = get_database("<system>", "<backend>", "<version>")
print(db is not None)
print(db.system_spec["gpu"])
PY
```

Run representative CLI queries for models expected to use this backend/system:

```bash
aiconfigurator cli default --model <model> --total-gpus <n> --system <system> --backend <backend>
aiconfigurator cli generate --model-path <model> --total-gpus <n> --system <system> --backend <backend>
```

If AIC raises `PerfDataNotAvailableError`, either collect the missing op or document that the model/backend mode is not supported. Do not claim support for a model path that still hits missing data.

## Phase 8: Support Matrix And Documentation

If the new GPU should become user-visible:

1. Update system support metadata.
2. Update support matrix outputs if required by the repo workflow.
3. Add or update tests for new SM filters or version routes.
4. Document known gaps in the PR body.

## Phase 9: PR Preparation

Keep commits reviewable:

1. `add <system> system spec`
2. `fix <backend> collectors for <system>`
3. `add <system> <backend> <version> perf data`
4. `test <system> data loading`

Before opening the PR:

```bash
git status --short
git diff --stat main...HEAD
pytest tests/unit/collector -q
pytest tests/unit/sdk/test_common.py -q
```

PR body checklist:

- GPU node identity and `nvidia-smi` summary.
- CUDA driver/container/backend image.
- Backend version detected at runtime.
- System YAML changes.
- Ops collected and perf files added.
- Collector errors fixed.
- Unsupported/skipped configs with reasons.
- Validation commands and results.
- Remaining risks.

Open the PR:

```bash
gh repo fork ai-dynamo/aiconfigurator --clone=false
git push -u <your-fork-remote> HEAD
gh pr create --repo ai-dynamo/aiconfigurator --head <your-user>:<branch> --title "<title>" --body-file <body-file>
```

## Stop And Escalate

Escalate instead of looping forever when:

- The framework kernel crashes consistently on valid production-like configs.
- The backend container cannot run on the node.
- The GPU platform or driver is unstable.
- The system spec values are unknown and materially affect AIC decisions.
- Data quality is suspect and cannot be explained.
- AIC requires broad SDK/model changes beyond collector/data bring-up.

In the final report, separate collected data, collector fixes, known gaps, and validation evidence.
