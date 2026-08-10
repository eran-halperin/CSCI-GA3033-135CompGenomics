---
layout: default
title: NYU HPC Guide
---

# NYU HPC Guide

All compute-intensive course project experiments must run on the **NYU HPC Cloud Bursting platform** (the instructional GPU platform provided by NYU IT — distinct from the Torch cluster, which is for research use only). Available resources include **A100 (40 GB)** and **L4** GPUs, with shared file systems for course data.

> **Note:** The platform is being updated ahead of the fall semester and is expected to be ready by mid-August. Detailed connection instructions and documentation will be shared once accounts are provisioned — this page will be updated accordingly. The Slurm example below is illustrative; exact hostnames, module names, and paths will be confirmed once documentation arrives.

---

## Getting an Account

Student and TA accounts are requested in bulk by the instructor after the semester begins. If you don't yet have access, check with the instructor. See NYU IT's [account request documentation](https://services.rt.nyu.edu/docs/hpc/getting_started/HPC_Accounts/getting_and_renewing_an_account/) for background on the process.

> **Shared storage caveat:** The platform's shared file systems are **not optimized for very large datasets**. Plan your data management strategy accordingly (e.g., subsampling, staging data closer to compute, or using compressed formats) — details will be refined once full documentation is available.

---

## Sample Slurm Script

Save the following as `submit_job.sh` and submit with `sbatch submit_job.sh`. **Illustrative example** — module names, paths, and `$SCRATCH` conventions will be confirmed once the platform's documentation is shared.

```bash
#!/bin/bash
#SBATCH --job-name=compgen_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=32GB                    # Request 32 GB RAM
#SBATCH --time=12:00:00              # Max wall time (HH:MM:SS)
#SBATCH --output=%x_%j.out           # stdout log: jobname_jobid.out
#SBATCH --error=%x_%j.err            # stderr log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<NetID>@nyu.edu

# --- Environment setup ---
module purge
module load python/3.11
module load cuda/12.1

source ~/.venv/compgen/bin/activate   # or: conda activate compgen

# --- Run training ---
python train.py \
    --data_dir "$SCRATCH/data/" \
    --output_dir "$SCRATCH/checkpoints/run_${SLURM_JOB_ID}" \
    --epochs 50 \
    --batch_size 64
```

Submit and monitor your job:

```bash
sbatch submit_job.sh          # submit
squeue -u $USER               # check queue
scancel <JOBID>               # cancel a job
sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,State   # job accounting
```

---

## PyTorch Boilerplate (for deep learning projects)

The snippet below handles device detection (CUDA → MPS → CPU) and enables **mixed-precision training** via `torch.cuda.amp` for faster, memory-efficient GPU utilization. Skip this section if your project uses R or other tools.

```python
import torch
from torch.cuda.amp import GradScaler, autocast

# ── Device detection ────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (local dev only — switch to CUDA on the HPC platform)")
    else:
        device = torch.device("cpu")
        print("Warning: no GPU found, falling back to CPU")
    return device


# ── Training loop with mixed precision ──────────────────────────────────────
def train(model, loader, optimizer, criterion, epochs: int = 10):
    device = get_device()
    model = model.to(device)

    # GradScaler is a no-op on CPU/MPS — safe to always instantiate
    scaler = GradScaler(enabled=device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Mixed-precision forward pass
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}]  loss: {avg_loss:.4f}")


# ── Checkpoint helpers ───────────────────────────────────────────────────────
import os

def save_checkpoint(model, optimizer, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path: str) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Resumed from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]
```

---

## Recommended Environments

**Python / PyTorch (deep learning projects)**
```bash
python -m venv "$SCRATCH/.venv/compgen"
source "$SCRATCH/.venv/compgen/bin/activate"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate wandb einops
```

**Python / Bioinformatics (single-cell, epigenomics, microbiome)**
```bash
pip install scanpy anndata pydeseq2 scikit-learn pandas numpy
```

**R (statistical genomics, GWAS, PRS)**
```bash
module load r/4.3
# Then inside R:
# install.packages(c("MASS", "lme4", "glmnet", "ggplot2"))
# BiocManager::install(c("limma", "minfi", "DESeq2"))
```

---

## Useful Links

- [NYU HPC Documentation](https://sites.google.com/nyu.edu/nyu-hpc)
- [Slurm Cheat Sheet](https://slurm.schedmd.com/pdfs/summary.pdf)
- [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)
