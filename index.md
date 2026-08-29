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

This course will bridge classical statistical genomics and modern deep learning applications in genomics — we will go over topics from maximum likelihood estimation through large-scale genomic foundational models. The objective of the course is to provide an overview of applications of machine learning in genomics, going deep into a few specific subjects, specifically:

- **Connecting genes to disease** — how do we find genes that are related to disease, how do we infer the ancestry of a patient from their genetic content, and how are these two topics connected.
- **Foundational models in genomics** — recent advances in foundational models led to incredible advances in AI (e.g., ChatGPT, Claude Code, etc.). These models "understand" human language. In recent years similar approaches have been adopted by the computational genomics community, trying to "understand" genomic data. We will go over a few of these models and see their applications.
- **Early cancer detection** — when cancer develops in the kidney, some of the "leftovers" of the tumor are circulating in the blood. Using a blood test, we can "read" these and try to find out whether cancer is developing, changing, or becoming more aggressive. We will learn the computational approaches that are used to detect these changes.
- **Deconvolution models** — biological samples are often heterogeneous and composed of many different types of tissues — for example, a blood sample is composed of many different cell types. We will learn about machine learning methods that de-convolute the data: the input is the read of the data from the heterogeneous sample, and the output is the information for each of the components (in the above case, cell types).

We will connect the computational methods to actual science and applications in biology and medicine. For example, we will show how computationally finding the genetic ancestry of a patient can help find genes that are related to that disease.

No background in biology is needed — this is a computational class and the focus is on the computational methodology; we will go over the biological concepts in class.

The following is a **tentative** syllabus - we may deviate slightly from this list of topics based on the students' interests, the projects, or the instructor's choice.

---

## Tentative Schedule

| Topic | Key Methods / Papers |
|------|-------|----------------------|
| **Intro: The Multi-Omic Landscape** | DNA, RNA, Methylation and the Microbiome |
| **Probabilistic Foundations I** | Maximum Likelihood Estimation (MLE) |
| **Probabilistic Foundations II** | Optimization and the EM Algorithm - Haplotype Phasing |
| **Epigenomics & Deconvolution** | CIBERSORT, UNICO, Tensor Composition Analysis |
| **Microbiome & Metagenomics** | Source Tracking, Time Series Data, Foundational Models |
| **Genomic Foundational Models I** | Transformers: Attention, BERT, DNABERT |
| **Genomic Foundational Models II** | Foundational Models in methylation and microbiome data |
| **Population Structure I** | PCA and Eigenstrat for Ancestry Inference, sparse PCA in methylation |
| **Population Structure II** | HMMs and Autoencoders |
| **GWAS & Linear Mixed Models** | Linear Mixed Models and heritability estimates |
| **Polygenic Risk Scores** | PRS calculations from summary data |



---

## Course Assignments and Policies

The objective of the class is to provide familiarity with machine learning and statistical genomics. There will be no exams. Instead, students choose **one** of the following two options (you do not need to do both):

**Hands-on project.** Students choose a project where they use existing data and code from an earlier paper for benchmarking and for developing improved approaches. These are guided projects with the teaching staff and include a short summary and a presentation of the results toward the end of the semester. Students choose the project from a list of topics — examples of potential projects:

- **Tumor detection (MRD)** — finding tumor signals in cell-free DNA samples.
- **Methylation foundational models** — using deep learning (transformer) approaches, specifically foundational models similar to ChatGPT, to predict health conditions of patients.

**Topic presentation.** Students can opt out of the hands-on project and instead present to the class a topic that covers a couple of computational approaches in one of the topics that we cover in the class. The list of topics and papers will be given early in the semester, and the student presentations will take place toward the end of the semester.

In addition, students are expected to send questions as well as answer other students' questions in the class forum during the semester.

**AI policy.** The usage of AI is allowed, both for coding and to prepare the presentations.

### Grade components

| Component | Weight |
|-----------|--------|
| Topic presentation / hands-on project presentation | 80% |
| Participation in the class forum | 20% |

There is **no homework** and there are **no exams** in this course. See the [Project & Presentation](project.md) page for details.

---

## Prerequisites

Python programming experience. Previous coursework in probability, statistics, and linear Algebra. Familiarity with machine learning fundamentals is helpful but not required. 
