---
layout: default
title: Project & Presentation
---

# Project & Presentation

## Overview

> **There are no exams and no homework in this course.** Each student chooses **one** of two options — a hands-on project or a topic presentation. The grade is **80%** presentation (project or topic) + **20%** participation in the class forum.

The objective is to provide familiarity with machine learning and statistical genomics. See the [Syllabus](/) for the full assignments and policies.

---

## Option 1 — Hands-on Project

Students choose a project where they use existing data and code from an earlier paper for benchmarking and for developing improved approaches. These are guided projects with the teaching staff.

Projects are chosen from a list of topics provided by the teaching staff. Examples (non-exhaustive):

- **Tumor detection (MRD)** — finding tumor signals in cell-free DNA samples.
- **Methylation foundational models** — using deep learning (transformer) approaches, specifically foundational models similar to ChatGPT, to predict health conditions of patients.

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

## Option 2 — Topic Presentation

Students can opt out of the hands-on project and instead present to the class a topic that covers a couple of computational approaches in one of the topics covered in the course.

- The list of topics and papers will be given early in the semester.
- Student presentations take place toward the end of the semester.

---

## Class Forum Participation

Throughout the semester, students are expected to send questions as well as answer other students' questions in the class forum. This accounts for **20%** of the grade.

---

## AI Policy

The usage of AI is allowed, both for coding and to prepare the presentations.
