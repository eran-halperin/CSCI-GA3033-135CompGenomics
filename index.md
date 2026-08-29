---
layout: default
title: Syllabus
---

# Computational Genomics (Fall 2026)

**CSCI-GA 3033-135 | New York University**

| | |
|---|---|
| **Instructor** | Eran Halperin, PhD |
| **Office** | TBD |

---

## Course Description

How do we find the genes behind a disease? Can a blood test catch a tumor before any symptoms appear? And can the same kind of "foundation model" that powers ChatGPT and Claude Code learn to read the genome?

This course bridges classical statistical genomics and modern deep learning. We start from first principles, including maximum likelihood estimation, the EM algorithm, and hidden Markov models, and build up to the large-scale genomic foundation models now reshaping the field. The goal is a working understanding of how machine learning is used in genomics today: broad enough to see the whole landscape, deep enough to build things yourself.

We go deep on four themes:

- **Connecting genes to disease.** How do we identify genes associated with a disease, how do we infer a patient's genetic ancestry from their DNA, and why do these two problems turn out to be deeply linked?
- **Foundation models for genomics.** The breakthroughs behind ChatGPT and Claude come from models that "understand" human language. The computational genomics community is now adapting the same ideas to "understand" DNA, RNA, and methylation data. We study several of these models and what they can do.
- **Early cancer detection.** When a tumor grows in the kidney, fragments of it circulate in the blood. A simple blood draw lets us "read" those fragments and ask whether a cancer is emerging, evolving, or turning aggressive. We cover the computational methods that make this possible.
- **Deconvolution.** Biological samples are mixtures. A blood sample contains many different cell types. Deconvolution methods take a measurement of the whole mixture and recover the signal from each component. We study the machine learning behind them.

Throughout, we connect the computational methods back to the underlying science and to real applications in biology and medicine, for example how inferring a patient's genetic ancestry can help pinpoint the genes that drive their disease.

**No biology background is required.** This is a computational course focused on methodology, and we cover the biological concepts we need as we go.

---

## Course Units

The course is built from the units below. They are **taught in a flexible order and often interleaved**. For example, cancer genomics follows naturally once we have covered EM, and the deconvolution unit comes near the end since the course projects do not depend on it. What stays fixed is the arc: build the statistical foundations first, then use them to read modern genomic data.

### Unit 1. Introduction to Genomics

A fast, friendly tour of genomics: the handful of biological ideas this course actually needs, and a map of where the semester is headed. No prior biology is required, and we build the background here so everyone starts on the same page.

### Unit 2. Statistical Genomics: Estimating Genomic Parameters

The classical backbone of the field: recovering hidden genomic quantities from noisy data. We work through ancestry inference, haplotype phasing, and variant calling, and the methods that power them:

- Maximum likelihood estimation
- Expectation-Maximization
- Hidden Markov models
- Dimensionality reduction approaches, and their connection to MLE

### Unit 3. Cancer Genomics: Early Detection

A focused look at one of the highest-stakes applications in the field. When a tumor sheds DNA into the bloodstream, a simple blood draw can reveal whether a cancer is present, growing, or evolving. We cover cell-free DNA and minimal residual disease (MRD) detection, and take a first look at separating tumor signal from healthy background, a mixture problem that Unit 6 develops in full.

### Unit 4. From Genes to Disease

How genetic variation is linked to disease, and why that question is inseparable from population structure. Genome-wide association studies, confounding by ancestry, heritability, linear mixed models, and polygenic risk scores. Here we close the loop from Unit 2: the same ancestry and structure estimates that look like nuisance parameters are exactly what make disease-gene discovery possible.

### Unit 5. Foundation Models in Genomics

The ideas behind modern language models, now trained on DNA. We cover the deep learning architectures and pretraining strategies used in genomic foundation models, with a focus on methylation and DNA sequence data.

### Unit 6. Deconvolution Methods in Genomics

Real genomic samples are mixtures: many cell types in a methylation or RNA sample, many microbial genomes in a microbiome sample. Building on maximum likelihood and EM, we study methods that pull these mixtures apart, including tensor deconvolution, non-negative matrix factorization, and maximum likelihood approaches, with applications across methylation and microbiome data.

---

## Course Assignments and Policies

This course is about gaining real familiarity with machine learning and statistical genomics, so there are no exams. Instead, you choose **one** of two tracks (you don't do both):

- **Hands-on project.** Start from the data and code of a published paper, then benchmark it and push it further with your own improvements. Projects are guided by the teaching staff, and you'll produce a short write-up and present your results near the end of the semester. You pick from a list of topics, for example:
  - **Tumor detection (MRD):** finding tumor signal in cell-free DNA.
  - **Methylation foundation models:** using transformer-based models (the same family as ChatGPT) to predict patients' health conditions from methylation data.
- **Topic presentation.** Prefer breadth over building? Instead of a project, present a topic to the class covering a couple of computational approaches from one of the areas we study. Topics and papers are assigned early in the semester, and presentations take place near the end.

Everyone also takes part in the class forum throughout the semester, posting questions and answering classmates'.

**Use of AI.** Using AI tools is allowed and encouraged, both for coding and for preparing your presentation.

### Grade

| Component | Weight |
|-----------|--------|
| Presentation (hands-on project or topic) | 80% |
| Participation in the class forum | 20% |

There is **no homework** and there are **no exams** in this course. See the [Project & Presentation](project.md) page for details.

---

## Prerequisites

Python programming experience. Previous coursework in probability, statistics, and linear Algebra. Familiarity with machine learning fundamentals is helpful but not required. 
