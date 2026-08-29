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
| **Office hours** | TBD (posted on Brightspace) |
| **Meeting time and location** | See Albert and Brightspace |
| **Teaching assistant** | TBD (posted on Brightspace within the first week of classes) |
| **Semester** | Fall 2026 |

> **Note:** Further course material (lecture slides, readings, assignments, and announcements) will be published on Brightspace.

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

**Learning outcomes.** By the end of the course, students will be able to formulate genomic questions as statistical estimation problems and apply machine learning and statistical methods to answer them, including for the genetic basis of disease (GWAS, heritability, mixed models, polygenic risk scores), genomic foundation models, and deconvolution of DNA, methylation, RNA, and microbiome data. Students will also be able to critically read current research papers and communicate a computational project or topic in writing and orally.

---

## Course Units

The course is built from the units below. They are **taught in a flexible order and often interleaved**.

### Unit 1. Introduction to Genomics

A fast, friendly tour of genomics: the handful of biological ideas this course actually needs, and a map of where the semester is headed. No prior biology is required, and we build the background here so everyone starts on the same page.

### Unit 2. Statistical Genomics: Estimating Genomic Parameters

The classical backbone of the field: recovering hidden genomic quantities from noisy data. We work through ancestry inference, haplotype phasing, and variant calling, and the methods that power them:

- Maximum likelihood estimation
- Expectation-Maximization
- Hidden Markov models
- Dimensionality reduction approaches, and their connection to MLE

### Unit 3. Cancer Genomics: Early Detection

A focused look at one of the highest-stakes applications. When a tumor sheds DNA into the bloodstream, a simple blood draw can reveal whether a cancer is present, growing, or evolving. We cover cell-free DNA and minimal residual disease (MRD) detection, and take a first look at separating tumor signal from healthy background, a mixture problem that Unit 6 develops in full.

### Unit 4. From Genes to Disease

How genetic variation is linked to disease, and why that question is inseparable from population structure. Genome-wide association studies, confounding by ancestry, heritability, linear mixed models, and polygenic risk scores.

### Unit 5. Foundation Models in Genomics

The ideas behind modern language models, now trained on DNA. We cover the deep learning architectures and pretraining strategies used in genomic foundation models, with a focus on methylation and DNA sequence data.

### Unit 6. Deconvolution Methods in Genomics

Real genomic samples are mixtures: many cell types in a methylation or RNA sample, many microbial genomes in a microbiome sample. We will study advanced approaches for deconvolution, including tensor deconvolution, non-negative matrix factorization, and maximum likelihood approaches, with applications across methylation, RNA, and microbiome data.

---

## Course Assignments and Policies

This course is about gaining real familiarity with machine learning and statistical genomics, so there are no exams. Instead, you choose **one** of two tracks (you don't do both):

- **Hands-on project.** Start from the data and code of a published paper, then benchmark it and push it further with your own improvements. Projects are guided by the teaching staff, and you'll produce a short write-up and present your results near the end of the semester. You pick from a list of topics, for example:
  - **Tumor detection (MRD):** finding tumor signal in cell-free DNA.
  - **Methylation foundation models:** using transformer-based models (the same family as ChatGPT) to predict patients' health conditions from methylation data.
- **Topic presentation.** Prefer breadth over building? Instead of a project, present a topic to the class covering a couple of computational approaches from one of the areas we study. Topics and papers are assigned early in the semester, and presentations take place near the end.

Everyone also takes part in the class forum throughout the semester, posting questions and answering classmates'.

**Use of AI.** Generative AI tools are permitted and encouraged. See [Course Policies](#course-policies) for the full policy.

### Grade

| Component | Weight |
|-----------|--------|
| Presentation (hands-on project or topic) | 80% |
| Participation in the class forum | 20% |

There is **no homework** and there are **no exams** in this course. See the [Project & Presentation](project.md) page for details.

### Late work

The graded work is the end-of-semester presentation, a short written summary for the project track, and ongoing participation in the class forum.

- Presentations happen on scheduled dates near the end of the semester. If you have a conflict, let the instructor know as early as you can and an alternative slot will be arranged.
- Written project summary: if you need more time, ask before the deadline. Reasonable extensions are granted without penalty.
- Forum participation is assessed over the whole semester, so there are no individual deadlines. Contribute steadily rather than all at once.

---

## Prerequisites

Python programming experience. Previous coursework in probability, statistics, and linear Algebra. Familiarity with machine learning fundamentals is helpful but not required.

---

## Course Materials

There is no required textbook. Lecture notes are posted on Brightspace before and after each class for review, along with the publications relevant to each lecture. No software setup is required. Students taking the hands-on project track are given access to Google Cloud Platform (GCP) as part of the course, and should have a GitHub account for their project code.

---

## Course Policies

### Academic integrity

All students are expected to follow the NYU Academic Integrity Policy. This includes not presenting others' words, code, figures, or ideas as your own, not fabricating results, and citing every source, dataset, and repository you use or adapt. Suspected violations are handled through NYU's academic integrity process and may result in a failing grade for the assignment or the course. If you are unsure whether something is allowed, ask the instructor first.

### Collaboration

Projects and topic presentations are encouraged to be done in teams of two. Both members should contribute, and the written summary should note who did what. Discussing course concepts and helping classmates on the class forum is encouraged.

### Generative AI

Generative AI tools (ChatGPT, Claude, Copilot, and similar) are permitted and encouraged in this course.

- **Allowed uses:** brainstorming project and presentation ideas, writing and debugging code, searching and summarizing literature, and drafting and editing text and slides.
- **Your responsibility:** you are fully responsible for everything you submit. Verify all claims, code, and citations, since AI output is often confidently wrong, and do not submit text or code you could not explain or reproduce yourself.
- **Disclosure:** in your project summary or presentation, include a brief note on which AI tools you used and for what. You do not need to log individual prompts.
- **Misuse:** fabricated results or references, or presenting AI output as your own understanding when it is not, is handled through the NYU academic integrity process.

### Academic accommodations

Academic accommodations are available to any student with a chronic, psychological, visual, mobility, or learning disability, or who is deaf or hard of hearing. Students must register with the Moses Center for Student Accessibility (212-998-4980, mosescsa@nyu.edu). Please contact the Moses Center as early in the semester as possible, and let the instructor know so that approved accommodations can be arranged in a timely way.

### Religious observance

NYU respects students' religious observances. If you cannot attend a class, meet a deadline, or give a presentation on its scheduled date because of a religious holiday or observance, notify the instructor in advance and reasonable alternative arrangements (a make-up or an extension) will be made without penalty. See NYU's religious observance policy for details; questions may be directed to religiousaccommodations@nyu.edu.

### Mental health and wellness

Your wellbeing comes first. If stress, anxiety, low mood, or other challenges are affecting your ability to engage with the course, please reach out. NYU's Wellness Exchange offers confidential medical and counseling support 24/7 at 212-443-9999 (wellness.exchange@nyu.edu) and through the NYU Wellness Exchange app. You are also welcome to talk with the instructor about coursework-related stress. In an emergency, call 911 or NYU Campus Safety at 212-998-2222; the Suicide and Crisis Lifeline is 988.
