# # NVIDIA Nemotron Reasoning Challenge - LoRA Training v1
# 
# **Strategy:** Target easy wins first (Roman numerals, Gravity, Unit conversion, Encryption)  
# **Approach:** BF16 on RTX Pro 6000 (95 GiB), QLoRA 4-bit fallback if VRAM < 40 GiB  
# **Data:** 9,500 original problems with reasoning + 6,500 synthetic = 16,000 total  
# 
# **Requirements:**
# - Add **competition data** as input (nvidia-nemotron-model-reasoning-challenge)
# - Add **model** as input (metric/nemotron-3-nano-30b-a3b-bf16)
# - Add **dataset** `adriano313/mamba-ssm-offline-deps` as input
# - Select **GPU RTX Pro 6000** accelerator
# - **Dependency Manager** (Settings → Custom Packages): `pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
# - Internet is **NOT** required (Dependency Manager handles offline install)

# Cell 1: Verify Environment & Install Dependencies
import subprocess, sys, os, glob

def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=kw.get("timeout", 600))
    return r.returncode == 0, r.stdout, r.stderr

def _pip(args, **kw):
    return _run([sys.executable, "-m", "pip"] + args, **kw)

# ── Environment info ─────────────────────────────────────────────────
print("=" * 70)
print("Environment Check")
print("=" * 70)

import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"  Location: {torch.__file__}")

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_properties(0).major
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu} (sm_{cc}0, {vram:.1f} GiB)")

    # Quick CUDA sanity check
    try:
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        assert (t * t).sum().item() == 14.0
        del t
        print("✓ CUDA functional test PASSED")
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"CUDA test FAILED: {e}")
        print(f"\nPyTorch {torch.__version__} (CUDA {torch.version.cuda}) does not support sm_{cc}0.")
        print(f"\nFIX: In Kaggle notebook Settings → Custom Packages, add:")
        print(f"  pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print(f"Then Save & Run All.")
        print(f"{'!'*70}")
        raise
else:
    print("⚠ No GPU detected — enable GPU in notebook settings")

# ── Install remaining dependencies ───────────────────────────────────
DEPS_DIR = None
for search_dir in ["/kaggle/input/mamba-ssm-offline-deps", "/kaggle/input"]:
    if not os.path.exists(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir):
        if any(f.endswith(".whl") for f in files):
            DEPS_DIR = root
            break
        if "config.json" in files:
            dirs.clear()
    if DEPS_DIR:
        break

if DEPS_DIR:
    print(f"Offline deps: {DEPS_DIR}")

def _install_pkg(name):
    import_name = name.replace("-", "_")
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass
    if DEPS_DIR:
        for pattern in [os.path.join(DEPS_DIR, f"{import_name}-*.whl"),
                        os.path.join(DEPS_DIR, "**", f"{import_name}-*.whl")]:
            found = sorted(glob.glob(pattern, recursive=True))
            if found:
                ok, _, _ = _pip(["install", "-q", "--no-deps", found[-1]], timeout=120)
                if ok:
                    return True
    ok, _, _ = _pip(["install", "-q", name], timeout=120)
    return ok

for pkg in ["peft", "trl", "datasets", "einops"]:
    _install_pkg(pkg)

print("✓ All dependencies ready.")

# Cell 2: Imports & Configuration
import os
# MUST be set before any CUDA allocation to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, re, math, random, string, gc, torch, glob, subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

CONFIG = {
    "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "max_seq_length": 1024,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "synthetic_per_easy_category": 1000,
    "synthetic_encryption": 250,
    "output_dir": "/kaggle/working/lora_adapter",
    "submission_path": "/kaggle/working/submission.zip",
}

# --- GPU Check + CUDA functional test ---
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
CUDA_OK = False
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    n_gpus = torch.cuda.device_count()
    cc = torch.cuda.get_device_properties(0).major
    print(f"🖥️  GPU: {n_gpus}× {gpu} ({vram:.1f} GiB each, sm_{cc}0)")
    try:
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        r = (t * t).sum().item()
        assert r == 14.0
        print(f"   ✓ CUDA functional test PASSED")
        CUDA_OK = True
        del t, r
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL: CUDA test FAILED — GPU cannot run tensor operations!")
        print(f"Error: {e}")
        print(f"\nPyTorch {torch.__version__} (CUDA {torch.version.cuda}) does not support sm_{cc}0.")
        print(f"Cell 1 should have upgraded PyTorch automatically.")
        print(f"If this persists, try: Restart kernel → Run All")
        print(f"{'='*70}")
        raise RuntimeError(
            f"GPU {gpu} (sm_{cc}0) is not supported by PyTorch {torch.__version__}. "
            f"Restart kernel and re-run to pick up the upgraded PyTorch from Cell 1.")
else:
    print("⚠ No GPU detected!")
    raise RuntimeError("No GPU available. Enable GPU in notebook settings.")

# --- Auto-discover model path ---
model_found = False
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "config.json" in files:
            cfg_path = os.path.join(root, "config.json")
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                arch = cfg.get("architectures", [""])[0].lower()
                model_type = cfg.get("model_type", "").lower()
                if "nemotron" in arch or "nemotron" in model_type or "mamba" in model_type:
                    CONFIG["model_name"] = root
                    model_found = True
                    print(f"✓ Model found at: {root}")
                    print(f"  Architecture: {cfg.get('architectures', ['unknown'])}")
                    break
            except:
                pass

if not model_found:
    print(f"⚠ Model not found locally, will use: {CONFIG['model_name']}")

total_vram = vram * n_gpus
USE_4BIT = total_vram < 40
print(f"\nConfig: {n_gpus}× {vram:.0f} GiB = {total_vram:.0f} GiB total, {'QLoRA 4-bit' if USE_4BIT else 'BF16'} mode")
print(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")


# Cell 3: Data Generation Functions

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

def int_to_roman(num):
    result = ""
    for value, symbol in ROMAN_VALUES:
        while num >= value:
            result += symbol
            num -= value
    return result

def int_to_roman_reasoning(num):
    roman = int_to_roman(num)
    reasoning = f"I need to convert {num} to Roman numerals.\n"
    if num >= 1000:
        reasoning += f"Thousands: {num // 1000} × M\n"
    hundreds = (num % 1000) // 100
    if hundreds > 0:
        reasoning += f"Hundreds place: {hundreds}00 = {int_to_roman(hundreds * 100)}\n"
    tens = (num % 100) // 10
    if tens > 0:
        reasoning += f"Tens place: {tens}0 = {int_to_roman(tens * 10)}\n"
    ones = num % 10
    if ones > 0:
        reasoning += f"Ones place: {ones} = {int_to_roman(ones)}\n"
    reasoning += f"Combining: {roman}"
    return reasoning, roman

def detect_category(prompt):
    if "bit manipulation" in prompt: return "bit_manipulation"
    elif "encryption rules" in prompt: return "encryption"
    elif "numeral system" in prompt: return "numeral_system"
    elif "gravitational" in prompt: return "gravity_physics"
    elif "unit conversion" in prompt: return "unit_conversion"
    elif "transformation rules" in prompt: return "equation_transform"
    return "unknown"

def extract_gravity_data(prompt):
    pairs = re.findall(r't\s*=\s*([\d.]+)s.*?distance\s*=\s*([\d.]+)', prompt)
    query_match = re.search(r't\s*=\s*([\d.]+)s\s+given', prompt)
    if pairs and query_match:
        return [(float(t), float(d)) for t, d in pairs], float(query_match.group(1))
    return None, None

def gravity_reasoning(examples, query_t, answer):
    t0, d0 = examples[0]
    g = 2 * d0 / (t0 ** 2)
    r = f"I need to find g from the examples using d = 0.5*g*t², so g = 2d/t²\n\n"
    r += f"From first example: t={t0}s, d={d0}m\n"
    r += f"g = 2×{d0}/{t0}² = {g:.4f}\n\n"
    if len(examples) > 1:
        t1, d1 = examples[1]
        r += f"Check: t={t1}s, d={d1}m → g={2*d1/t1**2:.4f} ✓\n\n"
    predicted = 0.5 * g * query_t ** 2
    r += f"For t={query_t}s: d = 0.5×{g:.4f}×{query_t}² = {predicted:.2f}"
    return r

def extract_unit_data(prompt):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    query_match = re.search(r'convert.*?:\s*([\d.]+)\s*m', prompt)
    if pairs and query_match:
        return [(float(i), float(o)) for i, o in pairs], float(query_match.group(1))
    return None, None

def unit_conversion_reasoning(examples, query, answer):
    in0, out0 = examples[0]
    ratio = out0 / in0
    r = f"Finding conversion ratio from examples.\n\n"
    r += f"First example: {in0}m → {out0}, ratio = {ratio:.6f}\n"
    if len(examples) > 1:
        in1, out1 = examples[1]
        r += f"Check: {in1}m → {out1}, ratio = {out1/in1:.6f} ✓\n"
    r += f"\nFor {query}m: {query} × {ratio:.6f} = {query*ratio:.2f}"
    return r

def extract_encryption_data(prompt):
    parts = prompt.split("Now, decrypt")
    if len(parts) != 2: return None, None
    pairs = re.findall(r'([a-z]+(?:\s+[a-z]+)*)\s*->\s*([a-z]+(?:\s+[a-z]+)*)', parts[0])
    query_match = re.search(r':\s*([a-z]+(?:\s+[a-z]+)*)', parts[1])
    if pairs and query_match:
        return pairs, query_match.group(1).strip()
    return None, None

def build_cipher_mapping(pairs):
    mapping = {}
    for enc, dec in pairs:
        ec, dc = enc.replace(" ", ""), dec.replace(" ", "")
        if len(ec) == len(dc):
            for e, d in zip(ec, dc): mapping[e] = d
    return mapping

def encryption_reasoning(pairs, query, answer):
    mapping = build_cipher_mapping(pairs)
    r = "Building letter substitution mapping from examples.\n\n"
    for enc, dec in pairs[:3]:
        ec = enc.replace(" ", "")
        dc = dec.replace(" ", "")
        if len(ec) == len(dc):
            maps = [f"{e}→{d}" for e,d in zip(ec[:5],dc[:5])]
            r += f"  '{enc[:30]}' → '{dec[:30]}': {', '.join(maps)}...\n"
    sm = dict(sorted(mapping.items()))
    r += f"\nMapping: {', '.join(f'{k}→{v}' for k,v in sm.items())}\n"
    r += f"\nDecrypting '{query}': {answer}"
    return r

def extract_bit_data(prompt):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    query = re.search(r'for:\s*([01]{8})', prompt)
    if pairs and query: return pairs, query.group(1)
    return None, None

def bit_reasoning(pairs, query, answer):
    r = "Analyzing input-output pairs for the bit pattern.\n\n"
    for inp, out in pairs[:4]: r += f"  {inp} → {out}\n"
    r += f"\nApplying pattern to {query}: {answer}"
    return r

def extract_equation_data(prompt):
    parts = prompt.split("Now, determine the result for:")
    if len(parts) != 2: return None, None
    pairs = re.findall(r'([^\n=]+?)\s*=\s*([^\n]+)', parts[0])
    actual = [(p.strip(), r.strip()) for p, r in pairs
              if not any(w in p.lower() for w in ["wonderland", "secret", "below"])]
    return actual, parts[1].strip()

def equation_reasoning(pairs, query, answer):
    r = "Finding symbol transformation rules.\n\n"
    for inp, out in pairs: r += f"  {inp} = {out}\n"
    r += f"\nApplying to '{query}': {answer}"
    return r

def generate_reasoning_chain(prompt, answer):
    cat = detect_category(prompt)
    reasoning = ""
    if cat == "numeral_system":
        m = re.search(r'write the number\s+(\d+)', prompt)
        if m: reasoning, _ = int_to_roman_reasoning(int(m.group(1)))
        else: reasoning = f"Converting to Roman numerals: {answer}"
    elif cat == "gravity_physics":
        ex, qt = extract_gravity_data(prompt)
        if ex and qt: reasoning = gravity_reasoning(ex, qt, answer)
        else: reasoning = f"d = 0.5*g*t² → {answer}"
    elif cat == "unit_conversion":
        ex, q = extract_unit_data(prompt)
        if ex and q: reasoning = unit_conversion_reasoning(ex, q, answer)
        else: reasoning = f"Applying ratio → {answer}"
    elif cat == "encryption":
        pairs, q = extract_encryption_data(prompt)
        if pairs and q: reasoning = encryption_reasoning(pairs, q, answer)
        else: reasoning = f"Decrypting: {answer}"
    elif cat == "bit_manipulation":
        pairs, q = extract_bit_data(prompt)
        if pairs and q: reasoning = bit_reasoning(pairs, q, answer)
        else: reasoning = f"Bit transform → {answer}"
    elif cat == "equation_transform":
        pairs, q = extract_equation_data(prompt)
        if pairs and q: reasoning = equation_reasoning(pairs, q, answer)
        else: reasoning = f"Transform → {answer}"
    else: reasoning = f"Result: {answer}"
    return f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"

print("Data functions loaded!")

# Cell 4: Generate All Training Data

# Auto-discover train.csv
df = None
csv_matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
if csv_matches:
    df = pd.read_csv(csv_matches[0])
    print(f"Loaded {len(df)} rows from {csv_matches[0]}")
else:
    # Fallback for local dev
    for p in ["data/raw/train.csv", "../data/raw/train.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"Loaded {len(df)} rows from {p}")
            break

if df is None:
    raise FileNotFoundError(
        "Cannot find train.csv!\n"
        "→ Click 'Add Input' → 'Competition' → search 'nvidia-nemotron-model-reasoning-challenge'"
    )

all_data = []

# 1. Process existing problems with reasoning chains
print("Processing existing problems...")
for _, row in df.iterrows():
    response = generate_reasoning_chain(row["prompt"], str(row["answer"]))
    all_data.append({"messages": [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": response},
    ]})
print(f"  {len(all_data)} existing problems processed")

# 2. Synthetic Roman numerals
rng = random.Random(42)
N = CONFIG["synthetic_per_easy_category"]
print(f"Generating {N} synthetic Roman numeral problems...")
for _ in range(N):
    target = rng.randint(1, 3999)
    ex_nums = rng.sample(range(1, 3999), rng.randint(3, 6))
    ex_text = "\n".join(f"{n} -> {int_to_roman(n)}" for n in ex_nums)
    prompt = ("In Alice's Wonderland, numbers are secretly converted into a different "
              f"numeral system. Some examples are given below:\n{ex_text}\n"
              f"Now, write the number {target} in the Wonderland numeral system.")
    answer = int_to_roman(target)
    reasoning, _ = int_to_roman_reasoning(target)
    all_data.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
    ]})

# 3. Synthetic gravity
print(f"Generating {N} synthetic gravity problems...")
for _ in range(N):
    g = round(rng.uniform(5.0, 30.0), 4)
    examples = [(round(rng.uniform(0.5, 5.0), 2), 0) for _ in range(rng.randint(3, 7))]
    examples = [(t, round(0.5*g*t**2, 2)) for t,_ in examples]
    qt = round(rng.uniform(0.5, 5.0), 2)
    ans = f"{round(0.5*g*qt**2, 2):.2f}"
    ex_text = "\n".join(f"For t = {t}s, distance = {d} m" for t,d in examples)
    prompt = ("In Alice's Wonderland, the gravitational constant has been secretly changed. "
              f"Here are some example observations:\n{ex_text}\n"
              f"Now, determine the falling distance for t = {qt}s given d = 0.5*g*t^2.")
    reasoning = gravity_reasoning(examples, qt, ans)
    all_data.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{ans}}}"},
    ]})

# 4. Synthetic unit conversion
print(f"Generating {N} synthetic unit conversion problems...")
for _ in range(N):
    ratio = round(rng.uniform(0.3, 3.0), 6)
    examples = [(round(rng.uniform(1.0, 100.0), 2), 0) for _ in range(rng.randint(3, 7))]
    examples = [(inp, round(inp*ratio, 2)) for inp,_ in examples]
    q = round(rng.uniform(1.0, 100.0), 2)
    ans = f"{round(q*ratio, 2):.2f}"
    ex_text = "\n".join(f"{inp} m becomes {out}" for inp,out in examples)
    prompt = ("In Alice's Wonderland, a secret unit conversion is applied to measurements. "
              f"For example:\n{ex_text}\nNow, convert the following measurement: {q} m")
    reasoning = unit_conversion_reasoning(examples, q, ans)
    all_data.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{ans}}}"},
    ]})

# 5. Synthetic encryption
N_ENC = CONFIG["synthetic_encryption"]
WORDS = ["alice","queen","king","rabbit","hatter","cat","turtle","mouse",
         "dragon","wizard","princess","knight","castle","garden","mirror",
         "forest","palace","tower","bridge","river","mountain","valley",
         "secret","magical","mysterious","golden","silver","ancient",
         "discovers","creates","imagines","watches","reads","follows",
         "chases","draws","builds","finds","opens","closes",
         "the","in","on","under","through","near","behind","above",
         "book","door","key","map","scroll","gem","crown","sword",
         "bird","fish","star","moon","sun","cloud","tree","flower",
         "wise","brave","clever","swift","strong","gentle","fierce"]
print(f"Generating {N_ENC} synthetic encryption problems...")
for _ in range(N_ENC):
    alphabet = list(string.ascii_lowercase)
    shuffled = alphabet.copy()
    rng.shuffle(shuffled)
    cipher = dict(zip(alphabet, shuffled))
    def encrypt(text): return "".join(cipher.get(c, c) for c in text)
    examples = []
    for _ in range(rng.randint(3, 6)):
        sentence = " ".join(rng.sample(WORDS, rng.randint(2, 6)))
        examples.append((encrypt(sentence), sentence))
    qs = " ".join(rng.sample(WORDS, rng.randint(2, 4)))
    eq = encrypt(qs)
    ex_text = "\n".join(f"{enc} -> {dec}" for enc, dec in examples)
    prompt = ("In Alice's Wonderland, secret encryption rules are used on text. "
              f"Here are some examples:\n{ex_text}\nNow, decrypt the following text: {eq}")
    reasoning = encryption_reasoning(examples, eq, qs)
    all_data.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{qs}}}"},
    ]})

random.seed(42)
random.shuffle(all_data)
print(f"\nTotal training examples: {len(all_data)}")

# Free DataFrame — no longer needed, saves ~50-100 MB of RSS
del df
gc.collect()

# Cell 5: Load Model & Tokenizer
import sys, os, types, importlib

# ======================================================================
# ENSURE MAMBA-SSM IS AVAILABLE
# ======================================================================
def _inject_cuda_mocks():
    for mod_name in ["selective_scan_cuda", "causal_conv1d_cuda"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

try:
    import mamba_ssm
    print(f"mamba-ssm {mamba_ssm.__version__} available")
except ImportError:
    print("mamba-ssm not found, installing...")
    os.environ["MAMBA_SKIP_CUDA_BUILD"] = "TRUE"
    os.environ["CAUSAL_CONV1D_SKIP_CUDA_BUILD"] = "TRUE"
    _inject_cuda_mocks()

    # Strategy 1: sys.path injection from offline sources
    _installed = False
    # Always search all of /kaggle/input for mamba-ssm source directories
    if True:
        for root, dirs, files in os.walk("/kaggle/input"):
            for d in dirs:
                full = os.path.join(root, d)
                if "mamba_ssm" in d and os.path.exists(os.path.join(full, "setup.py")):
                    sys.path.insert(0, full)
                elif "causal_conv1d" in d and os.path.exists(os.path.join(full, "setup.py")):
                    sys.path.insert(0, full)
            if "config.json" in files:
                dirs.clear()
        try:
            importlib.invalidate_caches()
            from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
            _installed = True
            print("  mamba-ssm installed via sys.path injection")
        except Exception as e:
            print(f"  sys.path injection failed: {e}")

    # Strategy 2: pip install (requires internet)
    if not _installed:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "causal-conv1d>=1.4.0", "mamba-ssm>=2.0.0"],
                       env={**os.environ, "MAMBA_SKIP_CUDA_BUILD": "TRUE",
                            "CAUSAL_CONV1D_SKIP_CUDA_BUILD": "TRUE"},
                       timeout=600)
        try:
            importlib.invalidate_caches()
            _inject_cuda_mocks()
            from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
            print("  mamba-ssm installed via pip")
        except Exception as e:
            raise RuntimeError(f"Cannot install mamba-ssm: {e}")

# ======================================================================
# FIX BITSANDBYTES (triton.ops compatibility)
# ======================================================================
if "triton.ops" not in sys.modules:
    sys.modules["triton.ops"] = types.ModuleType("triton.ops")
if "triton.ops.matmul_perf_model" not in sys.modules:
    _perf = types.ModuleType("triton.ops.matmul_perf_model")
    _perf.early_config_prune = lambda *a, **k: []
    _perf.estimate_matmul_time = lambda *a, **k: 0
    sys.modules["triton.ops.matmul_perf_model"] = _perf

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes {bnb.__version__} loaded")
except Exception as e:
    print(f"bitsandbytes: {e}")

# ======================================================================
# DETECT GPU + LOAD MODEL
# ======================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer

n_gpus = torch.cuda.device_count()
vram_per_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
vram_total = vram_per_gpu * n_gpus
USE_4BIT = vram_total < 40  # QLoRA only if total VRAM < 40 GiB
print(f"\nVRAM: {n_gpus}x {vram_per_gpu:.1f} GiB = {vram_total:.1f} GiB total")
print(f"Mode: {'QLoRA 4-bit' if USE_4BIT else 'BF16 full precision'}")

# Free training data memory during model load
_saved_all_data = all_data
del all_data
gc.collect()

# Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Model loading
gc.collect()
torch.cuda.empty_cache()

if USE_4BIT:
    from transformers import BitsAndBytesConfig
    # QLoRA 4-bit for low-VRAM GPUs (T4x2 etc.)
    print("Loading model in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
    )
    CONFIG["bf16"] = True
    CONFIG["fp16"] = False
    CONFIG["optim"] = "paged_adamw_8bit"
else:
    # BF16 for large-VRAM GPUs (RTX Pro 6000 etc.)
    print("Loading model in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )

model.config.use_cache = False

# Disable CUDA fast path (mocked causal_conv1d_cuda has no real kernels)
for _mod_name, _mod in sys.modules.items():
    if 'modeling_nemotron' in _mod_name and hasattr(_mod, 'is_fast_path_available'):
        _mod.is_fast_path_available = False

# Restore training data
all_data = _saved_all_data
del _saved_all_data
gc.collect()

print(f"\n✓ Model loaded on {n_gpus} GPUs")
for _i in range(n_gpus):
    _alloc = torch.cuda.memory_allocated(_i) / 1024**3
    print(f"  GPU {_i}: {_alloc:.2f} GiB allocated")


# Cell 6: Apply LoRA
from peft import LoraConfig, get_peft_model, TaskType

if USE_4BIT:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
else:
    model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["target_modules"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Ensure all LoRA params are BF16 (prevents GradScaler issues)
if USE_4BIT:
    _cast_count = 0
    for _n, _p in model.named_parameters():
        if _p.requires_grad and _p.dtype != torch.bfloat16:
            _p.data = _p.data.to(torch.bfloat16)
            _cast_count += 1
    if _cast_count:
        print(f"  Cast {_cast_count} LoRA params -> bfloat16")

model.print_trainable_parameters()
print(f"LoRA rank={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}")
print(f"Mode: {'QLoRA 4-bit' if USE_4BIT else 'BF16'}")

# Cell 7: Format & Create Dataset
from datasets import Dataset

def format_for_training(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = Dataset.from_list(all_data)
dataset = dataset.map(format_for_training)

print(f"Sample:\n{dataset[0]['text'][:400]}...")

split = dataset.train_test_split(test_size=0.02, seed=42)
train_ds, val_ds = split["train"], split["test"]
print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")

# Cell 8: Train!
from trl import SFTTrainer, SFTConfig

# warmup_steps ~ 5% of total steps
# total_steps = ceil(len(train_ds) / (batch * grad_accum)) * epochs
_steps_per_epoch = max(1, len(train_ds) // (CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]))
_total_steps = _steps_per_epoch * CONFIG["num_train_epochs"]
_warmup_steps = max(10, int(0.05 * _total_steps))
print(f"Steps: {_steps_per_epoch}/epoch x {CONFIG['num_train_epochs']} epochs = {_total_steps} total, {_warmup_steps} warmup")

sft_config = SFTConfig(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=_warmup_steps,
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    fp16=CONFIG["fp16"],
    bf16=CONFIG["bf16"],
    gradient_checkpointing=CONFIG["gradient_checkpointing"],
    optim=CONFIG["optim"],
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    max_length=CONFIG["max_seq_length"],
    dataset_text_field="text",
    packing=False,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

# Free PyTorch CUDA memory cache before training.
# During model loading + LoRA setup, PyTorch accumulates reserved-but-freed
# memory (~600+ MiB on T4). Calling empty_cache() returns those pages to CUDA
# so the backward pass can allocate gradient buffers.
gc.collect()
for _gi in range(torch.cuda.device_count()):
    with torch.cuda.device(_gi):
        torch.cuda.empty_cache()
for _gi in range(torch.cuda.device_count()):
    _free = (torch.cuda.get_device_properties(_gi).total_memory - torch.cuda.memory_allocated(_gi)) / 1024**3
    print(f"GPU {_gi} free after cache flush: {_free:.2f} GiB")

print(f"Training: {CONFIG['num_train_epochs']} epochs, "
      f"batch {CONFIG['per_device_train_batch_size']}x{CONFIG['gradient_accumulation_steps']}, "
      f"lr {CONFIG['learning_rate']}")

trainer.train()
print("Training complete!")

# Cell 9: Save Adapter & Package Submission
import zipfile

adapter_dir = CONFIG["output_dir"] + "/final"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print("Adapter files:")
for f in sorted(os.listdir(adapter_dir)):
    size = os.path.getsize(os.path.join(adapter_dir, f))
    print(f"  {f}: {size/1024:.1f} KB")

assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json")), "Missing adapter_config.json!"

# Package submission
with zipfile.ZipFile(CONFIG["submission_path"], "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(adapter_dir):
        fp = os.path.join(adapter_dir, f)
        if os.path.isfile(fp): zf.write(fp, f)

print(f"\nSubmission: {CONFIG['submission_path']}")
print(f"Size: {os.path.getsize(CONFIG['submission_path'])/1024/1024:.1f} MB")
print("\n" + "="*50)
print("Download submission.zip and submit to competition!")
print("="*50)

# Cell 10: Quick Validation
model.eval()

test_problems = [
    ("In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
     "Some examples are given below:\n5 -> V\n10 -> X\n50 -> L\n"
     "Now, write the number 42 in the Wonderland numeral system.", "XLII"),
    ("In Alice's Wonderland, the gravitational constant has been secretly changed. "
     "Here are some example observations:\n"
     "For t = 2.0s, distance = 20.0 m\nFor t = 3.0s, distance = 45.0 m\n"
     "Now, determine the falling distance for t = 4.0s given d = 0.5*g*t^2.", "80.00"),
    ("In Alice's Wonderland, a secret unit conversion is applied to measurements. "
     "For example:\n10.0 m becomes 5.0\n20.0 m becomes 10.0\n"
     "Now, convert the following measurement: 30.0 m", "15.00"),
]

print("Quick validation:")
for i, (prob, expected) in enumerate(test_problems):
    msgs = [{"role": "user", "content": prob}]
    inputs = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
    resp = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    boxed = re.search(r'\\boxed\{([^}]+)\}', resp)
    ans = boxed.group(1) if boxed else "NO_BOXED"
    status = "✓" if ans.strip() == expected else "✗"
    print(f"  [{status}] Q{i+1}: expected={expected}, got={ans}")
    print(f"       Response: {resp[:200]}...")
