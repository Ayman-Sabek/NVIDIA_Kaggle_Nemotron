# Kaggle Notebook Setup Guide

## Quick Start: Upload & Run on Kaggle

### Step 1: Create Kaggle Notebook

1. Go to <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge>
2. Click **"Code"** → **"New Notebook"**

### Step 2: Add Inputs

Click **"Add Input"** and add:

- **Competition Data**: `nvidia-nemotron-model-reasoning-challenge` (train.csv, test.csv)
- **Model**: Search for `metric/nemotron-3-nano-30b-a3b-bf16` → select "transformers" framework

### Step 3: Configure Accelerator

- Click **Settings** (gear icon)
- Set **Accelerator** to **GPU T4 x2** (or whatever highest is available)
- Set **Internet** to **ON** (needed for pip installs)
- Set **Persistence** to **Files only**

### Step 4: Copy Notebook

- Upload `kaggle_lora_train_v1.ipynb` OR
- Copy cells from `kaggle_lora_train_v1.py` into notebook cells

### Step 5: Run All Cells

- Click **"Run All"**
- Training takes ~2-6 hours depending on GPU
- Monitor loss curve — should decrease steadily

### Step 6: Download & Submit

- Download `submission.zip` from the notebook output
- Go to competition page → **"Submit Predictions"**
- Upload `submission.zip`

## If VRAM Issues (OOM)

Try these adjustments in order:

1. Reduce `max_seq_length` from 4096 → 2048
2. Reduce `lora_r` from 16 → 8
3. Remove `gate_proj`, `up_proj`, `down_proj` from targets (keep only attention)
4. Reduce `gradient_accumulation_steps` from 8 → 4
5. If all else fails, use Google Cloud G4 VM instead

## If Training is Too Slow

1. Reduce `num_train_epochs` from 2 → 1
2. Reduce training data: set `synthetic_per_easy_category` to 500
3. Disable `packing` (set to False)

## Model Path on Kaggle

The model will be available at:

```
/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1/
```

## Competition Data Path on Kaggle

```
/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv
/kaggle/input/nvidia-nemotron-model-reasoning-challenge/test.csv
```
