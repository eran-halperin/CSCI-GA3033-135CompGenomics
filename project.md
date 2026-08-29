---
layout: default
title: Project & Presentation
---

# Project & Presentation

## Overview

> **There are no exams and no homework in this course.** You choose **one** of two tracks — a hands-on project or a topic presentation. The grade is **80%** presentation (project or topic) + **20%** participation in the class forum.

This course is about gaining real familiarity with machine learning and statistical genomics. See the [Syllabus](/) for the full assignments and policies.

---

## Track 1 — Hands-on Project

Start from the data and code of a published paper, then benchmark it and push it further with your own improvements. Projects are guided by the teaching staff.

You pick from a list of topics provided by the teaching staff. Examples (non-exhaustive):

- **Tumor detection (MRD)** — finding tumor signal in cell-free DNA.
- **Methylation foundation models** — using transformer-based models (the same family as ChatGPT) to predict patients' health conditions from methylation data.

### Deliverables

| Deliverable | Details |
|-------------|---------|
| **Short summary** | A brief write-up of the methodology, analysis, and results |
| **Presentation** | A talk on the results in class toward the end of the semester |
| **Code** | Public GitHub repository with reproducible experiments |

### Compute

Compute-intensive experiments should run on the **NYU HPC Cloud Bursting platform** (A100 40GB / L4 GPUs). See the [HPC Guide](hpc_guide.md) for Slurm submission scripts and environment setup — this page will be updated as NYU IT finalizes documentation ahead of the fall semester.

- Use the tool appropriate for your project: PyTorch, R, scikit-learn, or domain-specific tools (e.g., PLINK, GATK, Scanpy)
- Log experiments and results for reproducibility (Weights & Biases, TensorBoard, or equivalent)
- Save outputs and checkpoints to shared scratch storage (`$SCRATCH`) — note that shared storage is not optimized for very large datasets, so plan your data management accordingly

---

## Track 2 — Topic Presentation

Prefer breadth over building? Instead of a project, present a topic to the class covering a couple of computational approaches from one of the areas we study.

- Topics and papers are assigned early in the semester.
- Presentations take place near the end of the semester.

---

## Class Forum Participation

Throughout the semester, everyone takes part in the class forum — posting questions and answering classmates'. This accounts for **20%** of the grade.

---

## Use of AI

Using AI tools is allowed and encouraged, both for coding and for preparing your presentation.
