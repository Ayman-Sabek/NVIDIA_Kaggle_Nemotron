# Nemotron Reasoning Challenge

**Competition:** [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

## Goal

Improve reasoning accuracy of `Nemotron-3-Nano-30B-A3B-BF16` by submitting LoRA adapters (max rank 32).

## Quick Start

```powershell
# Activate environment
.venv\Scripts\Activate.ps1

# Download data (requires Kaggle API key)
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw

# Run evaluation tests
python src/evaluate.py
```

## Project Structure

```
├── data/raw/         # Competition data
├── data/processed/   # Cleaned training data  
├── data/synthetic/   # Generated synthetic data
├── notebooks/        # Jupyter notebooks (EDA, training, submission)
├── src/              # Python modules (evaluate, data_prep, utils)
├── configs/          # LoRA training configs
├── adapters/         # Trained LoRA checkpoints
├── submissions/      # Packaged submission.zip files
└── Docs/             # Handoff docs, solution writeup
```

## Key Deadlines

| Date | Event |
|------|-------|
| April 9, 2026 | Midpoint prize cutoff ($5K + DGX Spark) |
| April 16, 2026 | Methodology writeup due (if midpoint winner) |
| June 8, 2026 | Entry deadline |
| June 15, 2026 | Final submission deadline |

## Submission Format

- LoRA adapter with `adapter_config.json`
- Packaged as `submission.zip`
- Answers in `\boxed{}` format
- Graded: exact match OR numerical tolerance 1e-2
