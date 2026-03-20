# Experiment Log

## March 18, 2026 — Workspace Setup

**Actions:**

- Created workspace structure (data/, notebooks/, src/, configs/, adapters/, submissions/)
- Installed core dependencies: kaggle, pandas, numpy, scipy, datasets, jupyter
- Created `src/evaluate.py` — local metric implementation (boxed answer extraction, numerical tolerance)
- Created `src/data_prep.py` — data loading utilities
- Created `src/utils.py` — submission packaging helper
- Created `notebooks/01_explore_data.ipynb` — EDA notebook (ready for data)
- All 5 evaluation self-tests passing

**Kaggle API configured:** username=adriano313, credentials at `C:\Users\ayman\.kaggle\kaggle.json`

**Data Downloaded:** train.csv (2997 KB, 9500 rows), test.csv (1.4 KB, 3 demo rows)

### Data Analysis Findings

**Dataset: 9,500 training problems across 6 equal categories (~16.5% each)**

| Category | Count | Answer Type | Avg Answer Len | Examples Given |
|----------|-------|-------------|-----------------|----------------|
| bit_manipulation | 1,602 | Binary (8-bit) | 8.0 chars | 8-11 per problem |
| encryption | 1,576 | Text (words) | 25.5 chars | 3-5 per problem |
| numeral_system | 1,576 | Roman numerals | 4.0 chars | 3-5 per problem |
| gravity_physics | 1,597 | Decimal number | 5.0 chars | 0 (implied formula) |
| unit_conversion | 1,594 | Decimal number | 4.9 chars | 0 (implied formula) |
| equation_transform | 1,555 | Mixed symbols | 2.9 chars | 0-2 per problem |

**Key Insights:**

- All tasks are **pattern recognition / in-context learning** — observe examples, infer rule, apply to new input
- Prompts are short: mean 75 tokens, max 128 tokens → plenty of room in 7,680 max_tokens
- Test set (public) has only 3 samples that overlap with train — real eval is hidden
- The benchmark is a **novel "Alice's Wonderland"** puzzle suite, not standard math/code

**Answer Formats:**

- Binary strings (bit_manipulation): exactly 8 chars, like `10010111`
- Text phrases (encryption): decrypted words, like `cat imagines book`
- Roman numerals: standard format, like `XXXVIII`
- Decimal numbers (gravity + unit): 2 decimal places, like `154.62`
- Symbol strings (equation_transform): mixed chars, like `@&` or `17/`

### Deep Puzzle Mechanics (verified computationally)

| Category | Underlying Rule | Difficulty for LLM |
|----------|----------------|-------------------|
| numeral_system | Standard integer→Roman numeral conversion | EASY — LLM knows this |
| gravity_physics | d = 0.5 *g* t² with secret g; infer g then compute | MEDIUM — needs arithmetic |
| unit_conversion | Linear ratio: output = input × ratio; infer ratio | MEDIUM — needs arithmetic |
| encryption | Substitution cipher; build mapping from examples | HARD — systematic pattern matching |
| bit_manipulation | Complex bitwise ops (not simple XOR); different per problem | HARD — pattern induction |
| equation_transform | Symbol substitution rules on equations | HARD — abstract reasoning |

### Leaderboard Snapshot (March 18, 2026 — Day 2)

- **Baseline (submission demo):** 0.50 (50% accuracy)
- **Top score:** 0.687 (CausalLM.org)
- **Top 10 range:** 0.60-0.69
- **165 active teams, 374 submissions**
- Competition is EARLY — 3 months remain, scores will climb significantly

### Strategic Assessment: Prompt Engineering vs LoRA

**VERDICT: LoRA training is the ONLY viable path. Prompt engineering is irrelevant.**

Reasoning:

1. **We cannot control the prompt** — the eval system feeds the problem text directly to the model via a fixed chat template. We only submit a LoRA adapter.
2. **temperature=0.0** — deterministic output. The model either gets it right or not. No prompt tricks can help.
3. **The puzzle types are novel** — "Alice's Wonderland" is a custom benchmark. The base model has never seen these specific task types.
4. LoRA training can teach the model the META-SKILL of solving these 6 puzzle types.

### Hardware Assessment

- **Local GPU:** Radeon RX 580 (4GB) — **COMPLETELY INADEQUATE** for any ML work with a 30B model
- **Google Cloud credits:** Competition partnered with Google Cloud for G4 VMs with RTX PRO 6000 Blackwell GPUs — **MUST apply for these**
- **Kaggle notebooks:** Free GPU (RTX PRO 6000 on Kaggle) — can be used for training + submission
- **Alternative cloud:** RunPod A100 80GB (~$0.74/hr), Lambda (~$1.10/hr)

### Next Steps

1. ~~Apply for Google Cloud competition credits~~ ✅ User has credits ready
2. Use Kaggle notebooks for initial LoRA training (free GPU)
3. ~~Generate synthetic training data for all 6 puzzle types~~ ✅ Done (16K examples)
4. Train LoRA adapter targeting reasoning improvement on these specific tasks

---

## March 18, 2026 — Phase 2: Training Data & Kaggle Notebook

### Synthetic Data Generator (`src/synthetic_gen.py`)

Created comprehensive training data pipeline:
1. **9,500 existing problems** → added step-by-step reasoning chains
2. **6,500 synthetic problems** generated:
   - 2,000 Roman numerals (range 1-3999)
   - 2,000 gravity physics (random g, t, d)
   - 2,000 unit conversions (random ratios)
   - 500 substitution cipher encryption

**Total: 16,000 training examples** → `data/processed/train_with_reasoning.jsonl`

### Reasoning Chain Quality by Category

| Category | Reasoning Chain | Quality |
|----------|----------------|---------|
| numeral_system | Step-by-step (thousands→hundreds→tens→ones) | HIGH |
| gravity_physics | Extract g=2d/t², verify, compute d=0.5gt² | HIGH |
| unit_conversion | Extract ratio, verify, multiply | HIGH |
| encryption | Build substitution mapping, decrypt letter-by-letter | HIGH |
| bit_manipulation | Show examples + answer (pattern too complex) | LOW |
| equation_transform | Show examples + answer (pattern too complex) | LOW |

### Kaggle Notebook (`notebooks/kaggle_lora_train_v1.ipynb`)

- QLoRA 4-bit NF4 + double quant
- LoRA rank=16, alpha=32, targets: q/k/v/o_proj + gate/up/down_proj
- SFT + packing, cosine LR, paged AdamW 8-bit
- 2 epochs, batch 1×8 grad accum, lr=2e-4, max_seq_len=4096
- Training format: `<think>[reasoning]</think>\boxed{answer}`

### Expected Score After v1 Training

- Target: Beat baseline 0.50 → 0.60-0.65
- Numeral system: ~95%+ | Gravity/Unit: ~80%+ | Encryption: ~60-70%
- Hard categories: Modest improvement from format training

---

## March 18, 2026 — Kaggle Runtime Debugging (5 iterations)

### Environment: Kaggle GPU RTX PRO 6000 Blackwell (sm_120, 102GB VRAM)

**Issue: mamba-ssm refuses to install on Blackwell GPU**

The Nemotron-3-Nano-30B model requires `mamba-ssm` for its hybrid Mamba-2 architecture. 
The `modeling_nemotron_h.py` imports `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn`.

| Attempt | Error | Fix Applied |
|---------|-------|-------------|
| 1 | FileNotFoundError: train.csv not found | glob.glob auto-discovery |
| 2 | `total_mem` AttributeError | Fixed to `total_memory` |
| 3 | mamba-ssm ImportError | Added pip install mamba-ssm |
| 4 | mamba-ssm install silent failure | Made installs verbose, added fallbacks |
| 5 | mamba-ssm build fails (sm_120 unsupported by nvcc) | Set TORCH_CUDA_ARCH_LIST=9.0+PTX |
| 6 | Still failing — env vars may not reach build subprocess | Inline shell env vars + GitHub clone fallback |
| 7 | User's Cell 1 was not updated on Kaggle (placeholder text still running) | Moved ALL mamba-ssm logic into Cell 5 with 3 cascading strategies |
| 8 | **Internet is OFF** — DNS resolution fails, all pip/git strategies fail | Created Kaggle dataset `adriano313/mamba-ssm-offline-deps` with source tarballs; added Strategy 0 (local install); pushed notebook via Kaggle API |

**Root cause (Attempt 8):** Kaggle notebook has internet disabled. All previous strategies 
(pip install from PyPI, git clone from GitHub) require network access.

**Solution:** Downloaded `mamba_ssm-2.3.1.tar.gz` and `causal_conv1d-1.6.1.tar.gz` from PyPI 
locally, uploaded as Kaggle dataset `adriano313/mamba-ssm-offline-deps`. Cell 5 now has 
**Strategy 0** that installs from `/kaggle/input/mamba-ssm-offline-deps/` with 
`MAMBA_SKIP_CUDA_BUILD=TRUE`.

**Notebook pushed via Kaggle API** to `adriano313/nemotron-lora-train-v1` — no more manual cell copying.

**Cell 5 strategies (in order):**
0. Install from local Kaggle dataset tarballs (OFFLINE — no internet needed)
1. pip install from PyPI (requires internet)
2. Git clone + patch setup.py + local install (requires internet)
3. Direct file copy from cloned repos to site-packages (requires internet)

All use `importlib.invalidate_caches()` + module cache clearing between attempts.

### Attempt 9-10: Internet CANNOT be enabled

**Discovery:** Internet cannot be enabled for this competition with the RTX PRO 6000 accelerator.
This is a Kaggle competition restriction — not just a settings issue.

| Attempt | Error | Fix Applied |
|---------|-------|-------------|
| 9 | ALL pip installs fail (internet OFF for ALL packages) | Rewrote Cell 1 to import-first approach (use pre-installed Kaggle packages) |
| 10 | **Internet cannot be enabled** — competition restriction with Blackwell GPU | Full offline overhaul: downloaded wheel files for peft, trl, bitsandbytes, einops, tyro, etc. Uploaded as Kaggle dataset v2. Switched from QLoRA 4-bit to BF16 (95.6 GiB VRAM is enough for 30B model without quantization) |

**Key changes in Attempt 10:**
- **BF16 mode**: Dropped bitsandbytes entirely. 30B × 2 bytes = ~60 GB, fits in 95.6 GiB VRAM.
- **Offline wheels dataset**: `adriano313/mamba-ssm-offline-deps` v2 now contains:
  - mamba_ssm-2.3.1.tar.gz, causal_conv1d-1.6.1.tar.gz (original)
  - peft-0.18.1, trl-0.29.0, bitsandbytes-0.42.0, einops-0.8.2 (new wheels)
  - tyro-1.0.10, docstring_parser-0.17.0, shtab-1.8.0, rich-14.3.3 (trl deps)
- **Cell 1**: Discovers offline deps dir, tries `__import__` first, then installs from local `.whl` files
- **Cell 5**: BF16 model loading (no BitsAndBytesConfig), streamlined mamba-ssm strategies
- **Cell 6**: Removed `prepare_model_for_kbit_training`, uses `gradient_checkpointing_enable()`
- **CONFIG**: `load_in_4bit: False`, `optim: adamw_torch` (was `paged_adamw_8bit`)
- **Notebook v4** pushed via Kaggle API

### Attempt 11-12 (v5-v6): requirements.txt fix, Dataset path discovery

| Attempt | Error | Fix Applied |
|---------|-------|-------------|
| 11 (v5) | `ValueError: Only install commands are supported: # Core ML` in requirements.txt | Stripped comments from requirements.txt |
| 12 (v6) | `No local tarballs found` — Cell 5 glob didn't find .tar.gz files. Logs revealed: (a) Kaggle auto-extracts .tar.gz into directories (`mamba_ssm-2.3.1/` not `mamba_ssm-2.3.1.tar.gz`), (b) pushed notebook got P100 (16GB) not RTX PRO 6000 | Added os.walk diagnostics in Cell 1, deep input listing in Cell 2, CUDA functional test |

**Key discovery from v6 logs:**
- DEPS_DIR correctly found at `/kaggle/input/datasets/adriano313/mamba-ssm-offline-deps`
- Contents show `causal_conv1d-1.6.1` and `mamba_ssm-2.3.1` as **directories** (not .tar.gz files!)
- `.whl` files preserved as-is, but tarballs were extracted by Kaggle
- GPU was **Tesla P100-PCIE-16GB**, not RTX PRO 6000 (push runs get arbitrary GPUs)
- CUDA functional test PASSED on P100

### Attempt 13 (v7): Install from extracted source dirs + adaptive VRAM

**Key changes:**
- **Cell 5**: `_find_mamba_sources()` now looks for extracted source directories (with `setup.py`/`pyproject.toml`) not just `.tar.gz` files
- **Cell 5**: Adaptive model loading — detects VRAM: ≥80 GiB → BF16, <80 GiB → QLoRA 4-bit
- **Cell 6**: Adaptive LoRA setup — `prepare_model_for_kbit_training` for 4-bit, `gradient_checkpointing_enable` for BF16
- **Notebook v7** pushed via Kaggle API
- Install uses list-based `subprocess.run` instead of shell strings (safer)

### Attempt 14 (v8): RTX PRO 6000 DEAD — switch to T4 x2

**Critical discovery from v7 interactive editor (RTX PRO 6000):**
- CUDA functional test **FAILED**: `CUDA error: no kernel image is available for execution on the device`
- PyTorch supports sm_50–sm_90, but RTX PRO 6000 Blackwell is **sm_120**
- ALL GPU operations fail — tensor ops, Triton kernels, model loading — nothing works
- mamba-ssm source build also fails: `Building wheel for causal_conv1d (pyproject.toml): error`
- RTX PRO 6000 is **completely unusable** until Kaggle updates PyTorch

**Positive findings from v7:**
- Source dirs found correctly: `mamba_ssm-2.3.1/mamba_ssm-2.3.1/`, `causal_conv1d-1.6.1/causal_conv1d-1.6.1/`
- Kaggle auto-extracts tar.gz → double-nested directories with setup.py/pyproject.toml
- All wheel deps installed correctly (peft, trl, etc.)

**Key changes in v8:**
- **Strategy 0 (NEW): sys.path injection** — adds source dirs directly to `sys.path`, bypasses compilation entirely
- **Cell 1**: Added `bitsandbytes` to packages list (needed for QLoRA 4-bit on T4)
- **Cell 5**: Three-strategy mamba install: sys.path → pip source → PyPI
- **Error message**: Now tells user to switch to T4 x2 if on RTX PRO 6000
- **Action required**: Switch accelerator from RTX PRO 6000 → **T4 x2**
  - T4 (sm_75) is fully supported by installed PyTorch
  - QLoRA 4-bit auto-activates for <80 GiB VRAM
  - 2×T4 = 30 GiB total, model in 4-bit ≈ 15 GiB → fits

### Attempt 15 (v9): bitsandbytes RuntimeError fix + hard CUDA halt

**Issue from v8 on RTX PRO 6000:**
- bitsandbytes 0.42.0 throws `RuntimeError` (not `ImportError`) on sm_120: `CUDA Setup failed despite GPU being available`
- `_install_wheel()` only caught `ImportError`, so the RuntimeError propagated and killed Cell 1

**Key changes in v9:**
- **Cell 1**: `_install_wheel()` now catches `Exception` (not just `ImportError`). If post-install import fails (bitsandbytes RuntimeError on sm_120), returns `True, "installed (CUDA init deferred)"` instead of crashing.
- **Markdown**: Updated to recommend **T4 x2** accelerator. Added warning that RTX PRO 6000 Blackwell sm_120 is NOT supported by PyTorch.
- **Cell 2**: Complete rewrite — hard CUDA functional test that creates `torch.tensor([1,2,3], device="cuda")`. If it fails, **raises RuntimeError** with clear message: "Switch to T4 x2 accelerator". No more silent failures — notebook halts immediately at Cell 2 if GPU is broken.
- **Removed** deep input listing from Cell 2 (was diagnostic, no longer needed).
- **Notebook v9** pushed via Kaggle API.

**Action required (UNCHANGED):** Switch accelerator from RTX PRO 6000 → **T4 x2**

### Attempt 16 (v10): Mock selective_scan_cuda for Strategy 0

**Root cause analysis of v9 failure:**
- Strategy 0 (sys.path injection) failed: `No module named 'selective_scan_cuda'`
- Traced import chain: `mamba_ssm/__init__.py` → `selective_scan_interface.py` line 20 has bare `import selective_scan_cuda` (NO try/except)
- This is the ONLY bare CUDA extension import in the entire mamba_ssm package
- `causal_conv1d_cuda` failures are already caught by try/except in `mamba2.py` and `ssd_combined.py`

**Key insight:** Nemotron uses Mamba-2, not Mamba-1. Mamba-2 uses Triton SSD kernels (`ssd_combined.py`), NOT `selective_scan_cuda`. The CUDA module is imported by `__init__.py` but never called in the Mamba-2 path.

**Fix in v10:**
- **Cell 5 Strategy 0**: Before adding source dirs to sys.path, inject `selective_scan_cuda` as a mock `types.ModuleType` into `sys.modules`
- Import of `mamba_ssm.__init__` → `selective_scan_interface.py` now finds the mock → import succeeds
- `causal_conv1d_cuda` import fails → caught by try/except → `causal_conv1d_fn = None`
- Mamba2 falls back to PyTorch Conv1d (slightly slower but correct)
- All Triton kernels (SSD, layernorm_gated) work without compilation

**Expected behavior on T4 x2:**
1. `selective_scan_cuda` mock injected → mamba_ssm imports clean
2. `causal_conv1d_fn = None` → Mamba2 uses PyTorch Conv1d fallback
3. Model loads in QLoRA 4-bit (T4 has ~15 GiB VRAM)
4. Training proceeds with Triton SSD + PyTorch Conv1d

### Attempt 17 (v11): Mock causal_conv1d_cuda too

**Issue from v10 on T4:**
- Strategy 0 succeeded (selective_scan_cuda mock worked!)
- Model loading with QLoRA 4-bit config triggered `from_pretrained`
- `modeling_nemotron_h.py` called `is_causal_conv1d_available()` → True (package on sys.path)
- `from causal_conv1d import causal_conv1d_fn` → imported `causal_conv1d_interface.py`
- `from causal_conv1d.cpp_functions import ...` → `import causal_conv1d_cuda` → bare import, no try/except → CRASH

**Fix:** Added `causal_conv1d_cuda` to mock list alongside `selective_scan_cuda`. Both are injected before and re-injected after cache clearing in `_try_import()`.

### Attempt 18 (v12): bitsandbytes 0.42.0 → 0.45.0

**Issue from v11 on T4:**
- mamba-ssm Strategy 0 succeeded ✓
- Tokenizer loaded ✓
- `AutoModelForCausalLM.from_pretrained` → transformers checks `is_bitsandbytes_available()`
- `import bitsandbytes` → `cextension.py` → `RuntimeError: CUDA Setup failed`
- bitsandbytes 0.42.0 has buggy CUDA library detection, can't find libs for CUDA 12.x on Kaggle T4

**Fix:**
- Downloaded `bitsandbytes-0.45.0-py3-none-manylinux_2_24_x86_64.whl` (69 MB)
- Replaced old 0.42.0 wheel in Kaggle dataset, uploaded v3
- Cell 1: bitsandbytes now force-installed (`--force-reinstall`) to override any broken system version
- bitsandbytes 0.45.0 has proper CUDA 12.x support with improved library detection

### Attempt 19 (v13): triton.ops mock + torch_dtype fix

**Issue from v12 on T4:**
- Force-install of bnb 0.45.0 didn't override system bnb 0.42.0 (or wasn't picked up)
- Old bnb 0.42.0 imports `triton.ops.matmul_perf_model` which was removed in newer Triton
- `bitsandbytes/nn/triton_based_modules.py` → `bitsandbytes/triton/int8_matmul_mixed_dequantize.py` → `from triton.ops.matmul_perf_model import ...` → ModuleNotFoundError
- Also: `torch_dtype` deprecation warning (should use `dtype`)

**Fix:**
- Cell 5: Mock `triton.ops` and `triton.ops.matmul_perf_model` with dummy functions before bnb import
- Cell 5: Pre-import bitsandbytes to verify it loads, clear transformers' `lru_cache` on `is_bitsandbytes_available`
- Cell 5: Changed `torch_dtype` → `dtype` in `load_kwargs` (both 4-bit and BF16 paths)

### Attempt 20 (v14): bitsandbytes 0.45.0 → 0.49.2 (version requirement)

**Issue from v13 on T4:**
- triton.ops mock worked ✓
- bitsandbytes 0.45.0 loaded ✓
- BUT: transformers `BITSANDBYTES_MIN_VERSION = "0.46.1"` — version check fails
- `is_bitsandbytes_available()` returns False because 0.45.0 < 0.46.1

**Fix:**
- Downloaded `bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl` (60.7 MB)
- Uploaded dataset v4 (replaces old 0.45.0 wheel)
- bnb 0.49.2 has its own `bitsandbytes/triton/matmul_perf_model.py` (no longer needs triton.ops)
- Kept triton.ops mock as safety net
- Added version fallback: if loaded bnb < 0.46.1, patches `BITSANDBYTES_MIN_VERSION`
- Added `is_bitsandbytes_available()` verification print
- Reverted `dtype` back to `torch_dtype` (the latter still works, just deprecated warning)

### Attempt 21 (v15): bitsandbytes double torch operator registration

**Issue from v14 on T4:**
- bitsandbytes 0.49.2 force-installed ✓ (wheel overrode system 0.42.0)
- Our verification step imported bnb successfully, registering torch operators
- Then we cleared sys.modules (`del sys.modules[key]` for all bitsandbytes keys)
- When transformers re-imported bnb, `_ops.py` tried `torch.library.define()` again
- RuntimeError: "Tried to register operator bitsandbytes::int8_mixed_scaled_mm multiple times"
- Also: still using `torch_dtype` (deprecated) instead of `dtype`

**Root cause:** bnb 0.49+ registers torch custom operators on first import via
`torch.library.define()`. These registrations persist even after clearing sys.modules.
Re-importing triggers duplicate registration.

**Fix:**
- Removed the `del sys.modules[key]` loop for bitsandbytes — import once and keep it cached
- If `is_bitsandbytes_available()` returns False, force-patch `BITSANDBYTES_MIN_VERSION = "0.0.0"`
- Changed `torch_dtype` → `dtype` in load_kwargs (both 4-bit and BF16 paths)

### Attempt 22 (v16): OOM during model loading on T4 x2

**Issue from v15 on T4:**
- All dependency/import issues are resolved (mamba-ssm, causal_conv1d, bitsandbytes 0.49.2, triton.ops)
- Kernel crashed 2 times: "Your notebook tried to allocate more memory than is available."
- 30B model in 4-bit ≈ 15 GiB, exceeding single T4's 14.6 GiB
- `device_map="auto"` wasn't splitting across both T4s effectively
- `bnb_4bit_use_double_quant=True` added ~0.5 GiB overhead

**Memory analysis:**
- Model 4-bit: 30B × 4 bits / 8 = ~15 GiB + quantization overhead
- Single T4: 14.6 GiB (not enough)
- T4 x2: 29.2 GiB total (must split properly)
- CPU RAM: ~29 GiB (safetensors on disk = ~56 GiB BF16)

**Fixes applied:**
- Added `gc.collect()` + `torch.cuda.empty_cache()` for all GPUs before model load
- Added `max_memory` map: `{0: "13GiB", 1: "13GiB", "cpu": "24GiB"}` (1.5 GiB headroom per GPU)
- Added `low_cpu_mem_usage=True` for shard-by-shard loading
- Disabled `bnb_4bit_use_double_quant` (True → False) to save ~0.5 GiB
- Reverted `dtype` → `torch_dtype` (known working, just deprecated warning)
- Reduced `synthetic_per_easy_category` 2000→1000, `synthetic_encryption` 500→250
- Reduced `max_seq_length` 4096→2048
- Added per-GPU VRAM usage reporting after model load

### Attempt 23 (v17): OOM at 54% — device_map="auto" overloads GPU 0

**Issue from v16 on T4 x2:**
- Kernel died 3 times in a row, always at exactly 54% of model loading
- Root cause: `device_map="auto"` fills GPU 0 sequentially until limit, then transitions
  to GPU 1. The MoE expert layers are large, and the transition fails when a big layer
  can't fit in the remaining space on GPU 0 but hasn't started offloading to GPU 1 yet.
- Also: `max_memory["cpu"] = "24GiB"` may have caused accelerate to PARK model layers
  on CPU (not just use CPU as loading buffer), consuming most of the 29 GiB system RAM.

**Fixes applied:**
- Changed `device_map="auto"` → `"balanced"` — splits layers evenly across both T4s
  from the start, ~8.5 GiB per GPU instead of filling one first
- Increased per-GPU headroom from 1.5 → 2.5 GiB: `{0: "12GiB", 1: "12GiB"}`
- Reduced `max_memory["cpu"]` from `"24GiB"` → `"1GiB"` — prevent CPU layer offloading
- Added `offload_folder="/kaggle/working/offload"` as disk offload safety net
- Added `gc.collect()` + `torch.cuda.empty_cache()` after model load (cleanup loading buffers)
- Added `torch.cuda.reset_peak_memory_stats()` + peak memory tracking per GPU

### Attempt 24 (v18): CPU RAM overflow during loading — GC monkey-patch

**Issue from v17 on T4 x2:**
- SAME crash at exactly 54% (param 3354/6243 = `backbone.layers.29.mixer.experts.41.up_proj.weight`)
- Crashed with BOTH `device_map="auto"` (v16) AND `"balanced"` (v17)
- Identical crash point proves it's NOT about GPU placement strategy

**Root cause analysis:**
- Each param loaded as BF16 on CPU (~9.6 MB avg), quantized to NF4, moved to GPU
- Python's GC doesn't collect freed BF16 CPU buffers fast enough
- After 3354 params: 3354 × 9.6 MB ≈ 32 GiB of accumulated dead CPU buffers
- Kaggle CPU RAM: ~29 GiB → OOM-killed by system

**Fix — monkey-patch accelerate's weight dispatcher:**
- Patched `accelerate.utils.set_module_tensor_to_device` with a wrapper
- Every 100 params: `gc.collect()` to free dead BF16 CPU buffers
- Every 500 params: `gc.collect()` + `torch.cuda.empty_cache()` + progress print
- Also patches `accelerate.utils.modeling.set_module_tensor_to_device` (direct import site)
- Post-load: restore original function + full cleanup

**Other changes:**
- Back to `device_map="auto"` (simpler, proven — the crash was CPU RAM, not GPU placement)
- `max_memory`: `{0: "12GiB", 1: "12GiB", "cpu": "4GiB"}` — moderate CPU for overflow
- `offload_folder` kept as disk safety net

### Attempt 25 (v19): malloc_trim watchdog — the real fix for 54% OOM

**Issue from v18 on T4 x2:**
- SAME crash at 54%. No `[GC]` messages appeared → the monkey-patch NEVER fired.
- Root cause of patch failure: `from accelerate.utils import set_module_tensor_to_device`
  creates a LOCAL reference in each importing module. Patching the module attribute
  doesn't affect existing local bindings in accelerate.big_modeling, transformers, etc.

**Deeper root cause (why gc.collect alone can't work):**
- `gc.collect()` only handles Python objects with cyclic references
- The real problem is **glibc's malloc allocator**: when torch frees BF16 CPU tensors,
  glibc's free() adds pages to its internal free list but does NOT return them to the OS
- `malloc_trim(0)` forces glibc to scan its free lists and return unused pages to the OS
- Without malloc_trim, freed-but-unreturned pages accumulate → OOM kill at 54%

**Fix — background memory watchdog thread:**
- Spawns a daemon thread that runs every 1 second during model loading
- Each iteration: `gc.collect()` + `ctypes.CDLL("libc.so.6").malloc_trim(0)`
- Monitors RSS via `/proc/self/status` and logs every 10 seconds
- No monkey-patching needed — works regardless of internal code paths
- Automatically stopped after model loading completes

**Additional changes:**
- Free `df` (DataFrame) at end of Cell 4 to reduce baseline RSS
- Temporarily delete `all_data` during loading, restore after (saves ~50 MB)
- Reduced `max_memory["cpu"]` from `"4GiB"` → `"1GiB"` to minimize CPU residency
- RSS monitoring printed to track actual memory usage during loading

---

## March 18, 2026 — v41: Offline Torch cu128 + Demo LoRA Config

### Root Cause of All Previous Failures

**RTX PRO 6000 Blackwell (sm_120) requires PyTorch with CUDA ≥ 12.8.**
Default Kaggle PyTorch 2.9.0+cu126 only supports up to sm_90.
Internet is BLOCKED for this competition with RTX Pro 6000 accelerator.
Cannot add utility scripts via Kaggle UI API push.

### Solution: Offline Torch cu128 Wheel Dataset

1. Downloaded PyTorch nightly cu128 wheels locally (801 MB):
   - `torch-2.12.0.dev20260319+cu128` (792 MB)
   - `torchvision-0.26.0.dev20260319+cu128` (8 MB)
   - `torchaudio-2.11.0.dev20260319+cu128` (1.6 MB)
2. Uploaded as Kaggle dataset: `adriano313/pytorch-cu128-blackwell`
3. Cell 1 auto-discovers and force-reinstalls from `/kaggle/input/pytorch-cu128-blackwell/`

### Key Notebook Changes (v41)

| Component | Before | After |
|-----------|--------|-------|
| Cell 1 | 4-phase complex logic (utility script + internet fallback) | 3-phase simple offline install |
| `lora_r` | 16 | 32 (competition max) |
| `lora_dropout` | 0.0 | 0.05 (matches demo) |
| `target_modules` | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` | `r".*\.(in_proj\|out_proj\|up_proj\|down_proj)$"` (demo regex) |
| `attn_implementation` | (default/unset for BF16) | `"eager"` (avoids flash-attn dependency) |
| Cell 5 mamba search | Searched only DEPS_DIR | Searches all of `/kaggle/input` |
| `enable_internet` | true | false |
| `kernel_sources` | `["ryanholbrook/nvidia-utility-script"]` | `[]` |
| `dataset_sources` | 1 dataset | 2 datasets (+ pytorch-cu128-blackwell) |

### Expected Behavior

1. Cell 1: Detects sm_120, finds cu128 wheels in dataset, force-reinstalls torch
2. Cell 2: CUDA functional test PASSES with cu128 torch
3. Cell 5: mamba-ssm loads via sys.path injection from offline deps
4. Model loads BF16 on 95 GiB VRAM with eager attention
5. Training runs with rank-32 LoRA targeting Mamba/MoE projections

### v41 Result: FAILED — PyTorch still 2.9.0+cu126

Cell 2 showed `PyTorch 2.9.0+cu126` → CUDA test failed on sm_120.

**Root cause:** Kaggle strips `+` from uploaded filenames during dataset creation.
- Uploaded: `torch-2.12.0.dev20260319+cu128-cp312-cp312-...whl`
- On Kaggle: `torch-2.12.0.dev20260319cu128-cp312-cp312-...whl`
- pip rejects the wheel because `2.12.0.dev20260319cu128` is not valid PEP 440 (needs `+cu128`)

### v42 Fix: Restore `+` in wheel filenames before pip install

- Cell 1 now copies wheels to `/tmp` with `+` restored via regex: `(\d)(cu\d+)` → `\1+\2`
- Added diagnostics: lists `/kaggle/input/` contents, prints pip stdout/stderr
- Added verbose output at each phase for debugging

### v42 Result: FAILED — PyTorch still 2.9.0+cu126

Cell 2 output identical to v41. User did not share Cell 1 output, so unclear whether:
- Dataset was found (may not have been manually attached in web UI)
- Wheel rename worked
- pip install ran at all

### v43 Fix: Uninstall-first + kernel restart

- Before installing cu128, explicitly `pip uninstall -y torch torchvision torchaudio triton`
- Fallback `shutil.rmtree` if pip uninstall doesn't fully remove old torch
- `IPython.Application.instance().kernel.do_shutdown(restart=True)` after install
- Enhanced diagnostics: `torch.__file__` path, all inputs listed, pip output shown

### v43 Result: FAILED — PyTorch still 2.9.0+cu126

Cell 2 still shows PyTorch 2.9.0+cu126 with CUDA test FAILED.
User did not share Cell 1 output — **diagnosis inconclusive**.

**Most likely root causes (in order of probability):**

1. **Dataset not attached** — `kaggle kernels push` metadata specifies dataset_sources, but
   if the notebook is already open in the interactive editor, datasets must be manually added
   via the web UI "Add Input" button. If `/kaggle/input/pytorch-cu128-blackwell` doesn't exist,
   the entire Phase 2 is skipped silently.

2. **Kernel restart interrupts execution** — When Cell 1 calls `do_shutdown(restart=True)`,
   the kernel restarts but "Run All" execution stops. Cell 2 would only run if the user
   manually re-runs. But user IS seeing Cell 2 output... this is inconsistent unless
   it ran before the restart or the restart never triggered.

3. **Wheel filename fix might not work** — Perhaps the regex `(\d)(cu\d+)` → `\1+\2`
   doesn't match the actual on-disk filename due to additional Kaggle transformations.

---

## March 19, 2026 — v44 Preparation: tar.gz Archive Approach

### Why tar.gz is the definitive fix

All v41-v43 failures stem from one root problem: **Kaggle strips `+` from uploaded filenames**.
- Individual `.whl` files: `torch-2.12.0.dev+cu128` → `torch-2.12.0.devcu128` (INVALID)
- Our regex fix tries to restore `+` at runtime, but unclear if it's even reaching that code

**Solution: Pack wheels inside a `.tar.gz` archive.**
- Archive filenames are NOT modified by Kaggle (only top-level files are renamed)
- `tarfile.extract()` in Python restores original filenames including `+`
- Single file upload: `torch_cu128_wheels.tar.gz` (797 MB, no `+` in its name)

### Changes Made

1. **Dataset v2**: Replaced 3 individual `.whl` files with `torch_cu128_wheels.tar.gz`
   - Archive verified: all 3 wheels inside have `+cu128` in names
   
2. **Cell 1 split into Cell 1a + Cell 1b**:
   - **Cell 1a** (torch install + restart):
     - Primary strategy: Extract wheels from tar.gz → pip install
     - Fallback strategy: Rename loose .whl files (same as v42/v43)
     - Uninstalls old torch first, then installs cu128
     - Restarts kernel after successful install
   - **Cell 1b** (other deps): Installs peft, trl, datasets, einops from offline wheels

3. **Better diagnostics**: Lists all dataset contents with file sizes before attempting install

### Blockers

- **Kaggle API key expired/revoked** — Cannot push dataset v2 or notebook v44
- User must regenerate API key at https://www.kaggle.com/settings → API → Create New Token

### Alternative Approaches (if tar.gz also fails)

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| A. tar.gz archive | Pack wheels in tar.gz, extract at runtime | Bulletproof filename preservation | Requires dataset re-upload |
| B. Kaggle Custom Packages | Add `pip install torch --index-url https://download.pytorch.org/whl/cu128` in Settings → Custom Packages | Runs at Docker build time WITH internet | May not support `--index-url` syntax; untested |
| C. Stable cu128 wheels | Find `torch==2.9.0+cu128` (same version as Kaggle's cu126) | Smaller download, stable release | May not exist for this version |
| D. Zip inside wheel | Use `.zip` extension instead of `.tar.gz` | Same concept as tar.gz | No advantage over tar.gz |

### Next Steps (for next session) — SUPERSEDED by Approach B below

---

## Approach B: Kaggle Dependency Manager (v45)

**Date:** Current session  
**Decision:** Pivot from tar.gz offline wheels to Kaggle's built-in Dependency Manager.

**Why:**
- Kaggle docs confirm: Settings → Custom Packages runs pip WITH internet at Docker build time
- Creates a "Dependency Installation Notebook" with wheels, auto-attached, runs BEFORE notebook
- Works with internet-disabled competitions — completely bypasses offline wheel problem
- Eliminates: `+` filename stripping, kernel restart, tar.gz extraction, dataset uploads

**Changes (v45):**
1. **Cell 0 (markdown):** Updated requirements — references Dependency Manager instead of pytorch-cu128-blackwell
2. **Cell 1:** Merged old Cell 1a + Cell 1b into single cell: env verification + offline deps install. Removed 130 lines of tar.gz extraction/kernel restart. Now just checks torch version, runs CUDA test, installs peft/trl/datasets/einops from offline wheels if needed.
3. **Cell 1b:** Deleted (functionality merged into Cell 1)
4. **kernel-metadata.json:** Removed `adriano313/pytorch-cu128-blackwell` from dataset_sources

**Dependency Manager entry (user must add in Kaggle web UI):**
```
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Fallback entries if `--index-url` not supported:**
- `pip install torch==2.9.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128`
- `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`

### Next Steps

1. User regenerates Kaggle API key
2. Push notebook v45: `kaggle kernels push -p notebooks`
3. In Kaggle web UI: Settings → Custom Packages → add the pip install command above
4. Commit & run → check if Dependency Installation Notebook appears
5. If torch cu128 loads → CUDA test passes → training begins

---

## Session — v46 to v47: Dependency Manager Failed → tar.gz Success

### v46 Result: Dependency Manager Completely Failed

- `pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` in Custom Packages did NOT work
- Cell 1 correctly detected cu126, searched `/kaggle/input` — **zero cu128 wheels found anywhere**
- Dependency Manager either failed silently on large download, doesn't support `--index-url`, or stores wheels in inaccessible location
- **Conclusion: Dependency Manager approach is DEAD**

### Fallback to tar.gz Dataset Upload — SUCCESS

1. **Found correct Kaggle upload API** by reading `kagglesdk` source code:
   - Step 1: `POST /api/v1/blobs/upload` → `{token, createUrl}` (GCS signed URL)
   - Step 2: `PUT createUrl` with raw file data
   - Step 3: `POST /api/v1/datasets/create/version/{owner}/{slug}` with file token
   
2. **Uploaded `torch_cu128_wheels.tar.gz`** (797 MB) as v2 of `adriano313/pytorch-cu128-blackwell`
   - Upload took ~264s at ~3 MB/s
   - Dataset version created successfully

3. **Updated notebook (v47)**:
   - Cell 0 (markdown): Updated requirements to reference tar.gz dataset
   - Cell 1 (code): Rewritten to extract tar.gz from `/kaggle/input/pytorch-cu128-blackwell/`, find cu128 wheels, uninstall cu126, install cu128, restart kernel
   - Removed Dependency Manager references
   - Added `adriano313/pytorch-cu128-blackwell` back to `kernel-metadata.json`

4. **Pushed v47** via Bearer token API — success

### Key Technical Notes
- Kaggle API: `KGAT_` tokens require Bearer auth (not Basic). `kaggle` CLI v1.7.4.5 doesn't support this.
- Push API field: use `slug` not `id` for the kernel identifier
- tar.gz preserves `+` in filenames inside the archive (Kaggle strips `+` from loose files)

### Next Steps
1. Run v47 on Kaggle with RTX Pro 6000 + pytorch-cu128-blackwell dataset attached
2. Remove Custom Packages entry from Kaggle web UI (no longer needed)
3. Verify: Cell 1 extracts tar.gz → installs cu128 → restarts → CUDA passes → deps install → training begins

### v48 Result: tar.gz Found But Kernel Restart Reverted Install

- Cell 1 found tar.gz, extracted wheels, installed cu128, called `do_shutdown(True)` to restart kernel
- Kernel restart message appeared: "The kernel appears to have died. It will restart automatically."
- After restart, Kaggle resumed at **Cell 2** (not Cell 1), and torch was **still cu126**
- **Root cause**: Kaggle kernel restart reverts filesystem changes — pip install is lost on restart

### v49: No-Restart Approach (CURRENT)

**Key design change**: Cell 1 NEVER imports torch directly and NEVER restarts the kernel.

1. **Cell 1** (setup):
   - Checks torch version via `subprocess` (separate Python process) — avoids module caching
   - If cu126: finds tar.gz, extracts to `/kaggle/working/cu128_wheels/`, pip uninstalls cu126, pip installs cu128
   - Verifies install via subprocess — confirms cu128 is active
   - Installs remaining deps (peft, trl, datasets, einops) from offline wheels
   - **NO kernel restart** — proceeds directly to Cell 2

2. **Cell 2** does `import torch` for the first time — gets the freshly installed cu128 version
   - CUDA functional test
   - Model path discovery
   - Config setup

**Why this works**: Since torch was never imported in Cell 1 (only in subprocess), Python has no cached cu126 modules. When Cell 2 imports torch, it gets cu128 from the pip-modified site-packages.

### v49 Result: FAIL - NCCL Library Conflict
- The new notebook successfully found and installed cu128 without restarting the kernel.
- However, import torch failed with ImportError: /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommResume
- **Conclusion**: The cu128 wheels have deep system-level conflicts with the base Kaggle Docker image (which is tied to CUDA 12.6 and older NCCL libraries).
- **Strategic Decision**: ABANDON Kaggle Notebooks for training on the Blackwell GPU. The environment is fundamentally broken for sm_120 training. Pivot to Google Cloud G4 VMs where we have root access and can cleanly install CUDA 12.8.
