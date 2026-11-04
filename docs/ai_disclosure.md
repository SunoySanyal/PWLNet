# Declaration of Generative AI Use

**Manuscript:** *PWLNet: A Framework for Linearized Neural Surrogates in
Nonlinear Process Optimization*

**Authors:** Sunoy Sanyal<sup>a,b</sup>, Canberk Demir<sup>b,c</sup>,
Metin Türkay<sup>b</sup>, Erdal Aydın<sup>ad,\*</sup>

<sup>a</sup> Department of Industrial and Systems Engineering, Rensselaer Polytechnic Institute,
110 8th St, Troy, NY, 12180, USA
<sup>b</sup> Department of Industrial Engineering, Koç University,
Rumelifeneri Yolu, Istanbul, 34450, Türkiye
<sup>c</sup> Department of Chemistry, Koç University, Rumelifeneri Yolu,
Istanbul, 34450, Türkiye
<sup>d</sup> Department of Chemical and Metallurgical Engineering, Aalto University, Otakaari 1,
Espoo, 02150, Finland

**\*Corresponding author:** Erdal Aydın, erdal.aydin@aalto.fi

**Accompanying software:** `PWLNet` — https://github.com/SunoySanyal/PWLNet

**Submitted to:** Elsevier

**Date of this declaration:** <submission date>

---

## 1. Summary

Two generative-AI systems were used during this work, for two clearly
separated purposes:

| Tool | Provider | Used for | Period |
|------|----------|----------|--------|
| ChatGPT (V 5) | OpenAI | Language, grammar, and structural editing of author-written manuscript text | June–July 2026 |
| Claude / Claude Code (Fable 5; Opus 4.8) | Anthropic | Programming assistance during development of the accompanying `PWLNet` software, notebooks, and figure-generation scripts | June–July 2026 |

Neither tool generated the scientific content, the research design, the
numerical results, or the conclusions of this work. Both were used under
continuous author supervision, and the authors take full responsibility for
the entire content of the manuscript and of the released software.

---

## 2. Use of ChatGPT in the writing process

Generative AI was used **only as a language and structuring aid on text that
the authors had already conceived and written or dictated themselves.** The
workflow was as follows:

1. The author described, in detail and in their own words, the technical
   content to be conveyed in each section — the scientific claims, the
   methodology, the results, and the interpretation — either by typing draft
   prose or by explaining the intended content conversationally to the tool.
2. The tool was asked to return the same information in a better-organised
   form. Its contributions were limited to:
   - correcting grammar, spelling, punctuation, and article/preposition usage
     (the authors are not native English speakers, and this was the principal
     motivation for using the tool);
   - improving sentence structure, register, and readability;
   - reorganising the order of sentences and paragraphs for logical flow;
   - proposing a small number of connective or transitional sentences to link
     passages that the author had supplied;
   - removing redundant or repetitive sentences and shortening passages to
     meet length constraints.
3. Every suggestion returned by the tool was read, checked for technical
   accuracy, and either accepted, edited, or rejected by the authors. Text
   that altered the meaning of the author-supplied content was rejected or
   rewritten.
4. Passages written entirely by hand by the authors were additionally passed
   through the tool for grammar correction only.

### What ChatGPT was *not* used for

The authors confirm that ChatGPT was **not** used to:

- generate scientific ideas, hypotheses, research questions, or the study
  design;
- derive, state, or verify any mathematical formulation, proof, or model
  equation — including the ReLU big-M, exact tanh, piecewise-linear tanh
  big-M/SOS2, and SOSX branch-and-bound formulations presented in the paper;
- produce, simulate, transform, fabricate, or manipulate any data;
- run, interpret, or draw conclusions from any numerical experiment;
- produce any figure, table, or numerical value reported in the manuscript;
- generate, select, or verify references and citations; every reference was
  located, read, and cited by the authors;
- write the abstract, highlights, conclusions, or any claim of novelty from
  scratch without prior author-supplied content;
- review the manuscript in place of a human author or reviewer.

---

## 3. Use of Claude Code in software development

The `PWLNet` software that produces the results reported in the manuscript was
developed with the assistance of Claude Code (Anthropic), an agentic
command-line coding assistant operated interactively by the authors in their
own development environment. This section is included because the software is
released publicly alongside the manuscript and is used to generate the
reported results, so the authors consider its provenance part of the
reproducibility record referred to in the Data Availability statement.

### Nature of the assistance

The assistant was used in an author-directed, session-by-session manner. In
each session the authors specified the algorithm, the mathematical
formulation, the intended interface, or the defect to be fixed; the assistant
proposed source-code changes; and the authors inspected, executed, tested,
corrected, and committed them under their own name. Typical assisted tasks
were:

- implementation and refactoring of library modules from author-specified
  designs — for example the mixed-integer formulations of the piecewise-linear
  tanh activations, the training back-ends, the optimization-aware pruning and
  compression utilities, the SOSX branch-and-bound components, interval bound
  propagation, and the Pyomo/OMLT and solver interfaces;
- translation of author-derived equations and formulations into working code;
- writing and repairing unit tests, benchmark harnesses, and the plotting and
  figure-generation scripts used for the manuscript figures;
- debugging — locating the causes of numerical, plotting, and interface
  defects identified by the authors;
- routine software engineering: docstrings, API documentation, README content,
  notebook restructuring, and code style;
- drafting commit messages and internal development notes.

### Extent recorded in the version history

At the time of writing, the public repository contains 99 commits made between
11 June 2026 and 8 July 2026 by two human authors. Eight of these commits
carry an explicit `Co-Authored-By: Claude` trailer, recording sessions in which
a substantial part of the code in that commit was drafted with the assistant:
three on 12 June 2026 (Claude Fable 5) and five between 13 and 17 June 2026
(Claude Opus 4.8). That trailer marks only a subset of the assisted work: the
assistant was also used, without the trailer being recorded, in later sessions
covering notebook repairs, regeneration of the manuscript figures in vector
format, and documentation. This declaration is therefore intended to cover the
whole development period rather than only the annotated commits.

### Author control, verification, and responsibility

- No AI-generated code was accepted without being read by an author.
- No result reported in the manuscript was taken from the assistant. Every
  reported number, table, and figure was produced by executing the authors'
  own code in the authors' own environment.
- Model-implementation changes that affect the physics were validated by the
  authors against the published reference sources rather than against the
  assistant's output; for example, the corrections to the hydroformylation
  reaction kinetics and catalyst loading were accepted only after the
  simulated trajectories were checked against the corresponding figures of the
  original literature model (Kiedorf et al., 2014; Hentschel et al., 2015).
- Numerical-optimization results were cross-checked against independent
  references before being reported — literature solutions and independently
  computed reference optima, such as the multiple-shooting hydroformylation
  reference solution and the refined direct-shooting Van de Vusse reference
  against which the surrogate controllers are benchmarked.
- Formulation equivalences reported in the paper were verified numerically by
  the authors rather than asserted — for example the SOSX-versus-big-M
  cross-checks and the exact replay of MILP plans through the NumPy cell.
- The assistant was not permitted to define the research questions, choose the
  benchmark problems, decide the experimental protocol, or interpret the
  results.
- The full source code, notebooks, datasets, trained models, and
  figure-generation scripts are publicly released, so that every reported
  result can be reproduced and independently audited.

---

## 4. Authorship, responsibility, and data confidentiality

The authors reviewed and edited all AI-assisted text and all AI-assisted code,
and take full responsibility for the entire content of the manuscript and of
the released software, including scientific accuracy, originality, the
integrity of the data and results, and the correctness of all references. No
generative-AI tool is listed as an author or co-author, and no AI tool appears
in the CRediT Author Statement, in accordance with COPE, ICMJE, and Elsevier
policy, because such tools cannot take responsibility for the work or approve
the final version.

The authors further confirm that no confidential, third-party, or unpublished
data belonging to others was entered into either tool, and that the
AI-assisted text was checked for unattributed overlap with existing
publications.

---
## 5. The section in the manuscript has been added


