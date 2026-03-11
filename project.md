---
layout: default
title: Final Research Paper
---

# Final Research Paper

## Overview

> **There is NO homework in this course.** The sole deliverable is the Final Research Paper, which accounts for 90% of your grade (70% paper + 20% peer review).

Students will conduct original research at the intersection of machine learning and genomics, produce a conference-style paper, and participate in a double-blind peer-review process during Week 14.

---

## Paper Requirements

| Requirement | Details |
|-------------|---------|
| **Format** | NeurIPS / ICML style, 8 pages max (+ unlimited references) |
| **Scope** | Novel application or methodological contribution in computational genomics or genomic ML |
| **Code** | Public GitHub repository with reproducible experiments |
| **Compute** | All training **must** run on the **NYU Torch HPC** using **PyTorch** |

### Topic Ideas (non-exhaustive)

- Fine-tuning a genomic foundation model (e.g., Enformer, Evo, Nucleotide Transformer) on a downstream task
- Novel architecture for sequence-to-function prediction
- Self-supervised pre-training on single-cell or epigenomic data
- Improving polygenic risk score prediction with deep learning
- Multi-modal integration of genomic + clinical data

---

## Compute Requirements

All model training **must** be performed on the **NYU Torch HPC cluster** using **PyTorch**. See the [HPC Guide](hpc_guide.md) for Slurm submission scripts and boilerplate code.

- Use `torch.cuda.amp` for mixed-precision training
- Log experiments with Weights & Biases or TensorBoard
- Save checkpoints to your Torch scratch space (`$SCRATCH`)

---

## Timeline

| Milestone | Due Date |
|-----------|----------|
| Topic proposal (1 page) | Week 4 |
| Related work & methods outline | Week 7 |
| Draft paper submission (for peer review) | Week 13 |
| Peer reviews submitted | Week 14 (before conference) |
| Final paper (camera-ready) | Week 14 + 1 week |

---

## Double-Blind Peer Review

The Week 14 session is modeled on a real machine learning conference:

1. **Anonymization** — Remove all author names and affiliations from your draft before submission. The instructor will assign papers to reviewers.
2. **Each student reviews 2 papers** — Reviews must follow the provided rubric (novelty, soundness, clarity, significance).
3. **Reviews are also anonymous** — Reviewers do not know whose paper they are reviewing, and authors do not know who reviewed them.
4. **Rebuttal period** — Authors have 48 hours to respond to reviews before the final paper is due.
5. **Presentation** — All students present their work (10 min talk + 5 min Q&A) during the Week 14 conference session.

### Review Rubric

| Criterion | Points |
|-----------|--------|
| Novelty & Originality | 25 |
| Technical Soundness | 25 |
| Clarity & Writing Quality | 25 |
| Experimental Rigor & Reproducibility | 25 |
| **Total** | **100** |
