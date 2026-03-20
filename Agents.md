# Workspace Agent Instructions

## Context
This is the **NVIDIA Nemotron Model Reasoning Challenge** workspace for the Kaggle competition.
Full details in `Docs/KAGGLE_NEMOTRON_COMPETITION_HANDOFF.md`.

## Rules
- Always activate the venv (`.venv\Scripts\Activate.ps1`) before running Python commands
- Use `d:\AI\Nemotron_Challange\.venv\Scripts\python.exe` as the Python interpreter
- Do NOT modify anything in `d:\AI\World-Agent` (the trading workspace)
- Log experiments in `Docs/experiment_log.md`
- Save trained adapters in `adapters/` with version prefixes (v001_, v002_, etc.)
- Package submissions via `src/utils.py:package_submission()`

## Current Phase
**Phase 1: Baseline** — Explore dataset, understand benchmark, get zero-shot baseline score.

## Key Constraints
- LoRA max rank: 32
- Eval: vLLM, temperature=0.0, max_tokens=7680
- Answers must be in `\boxed{}` format
- Model: 30B params hybrid (Mamba-2 + MoE + GQA) — needs careful LoRA target selection

## Infrastructure & Cloud Tracking
- **Environment**: GCP VM `nemotron-trainer` (g2-standard-4, 1x L4 GPU, 500GB disk).
- **Project**: gen-lang-client-0867915969 | **Zone**: us-east4-a
- **Connection**: `gcloud compute ssh nemotron-trainer --zone="us-east4-a" --tunnel-through-iap`
- **Base Software**: Ubuntu 22.04, Python 3.10, NVIDIA Driver 550, CUDA 12.4
- **State**: Provisioned, configuring driver and uploading workspace...
- **Security Check**: Only stateless `--command="..."` SSH executions used. No hanging sessions.
