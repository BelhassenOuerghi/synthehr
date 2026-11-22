# A compliance framework for synthetic-EHR release: Pillar 3 of a four-pillar programme

*Part of [SynthEHR](../../README.md), an independent research programme by Belhassen Ouerghi. [All research stages](../README.md).*

## 1. Overview

This folder is Pillar 3, the ethics and compliance pillar, of my research
programme on synthetic electronic health records. Stages 0 to 2 built and measured
the artefact: a CTGAN baseline
([`research/stage0-synthesis`](../stage0-synthesis)),
a differentially private generator
([`research/stage1-privacy`](../stage1-privacy)),
and a fairness-mitigated one
([`research/stage2-fairness`](../stage2-fairness)).
This stage answers the question a data controller actually has to answer: what
evidence makes such a release defensibly compliant, and can that evidence be produced
automatically rather than written by hand?

The contribution is deliberately not a data protection impact assessment for one
dataset, which would be administration. It is a reusable, executable compliance
framework, comprising an evidence compiler that generates an impact assessment, a
data card, a model card, a residual-risk register and a regulatory mapping from the
pipeline's own manifests and metrics, together with a programmatic release gate that
decides, and can refuse, the promotion of a candidate dataset. Applying the same
framework to a second dataset is its planned reusability test.

## 2. Why compliance is the third pillar

Privacy produced a formal budget; bias mitigation produced fairness evidence. Neither,
by itself, permits a release. Under UK GDPR a controller must assess and document
residual identifiability, by means reasonably likely to be used (Recital 26),
demonstrate accountability (Article 5(2)), and, for AI touching access to healthcare,
meet the EU AI Act's data-governance and technical-documentation duties. These are
evidentiary obligations, and the standard failure mode in practice is documentation
drift: compliance documents are written once, by hand, from a snapshot of the
pipeline; the pipeline is re-run; and the document silently ceases to describe the
data.

This stage attacks that failure mode directly. Every number in the evidence package is
injected from the artefacts that generated the data, so the documents cannot drift.
Re-running the pipeline recompiles them.

It is worth being precise about what "third" means here. Ethical *practice* has
governed the work from the first commit: the data use agreement, the rule that no data
is ever committed, the decision to treat un-privatised synthetic data with the same
caution as real data, and the refusal to release anything before the privacy pillar.
Ethics was never deferred. What is third is the ethical *evidence package*, and it is
third by dependency, because it consumes what Pillars 1 and 2 produce. An impact
assessment with no privacy budget and no disparity metrics is an empty form.

## 3. The chosen technique: evidence as code

Three design commitments, argued in [`doc/research-paper.md`](doc/research-paper.md).

**An evidence compiler, not templates filled in by hand** (`src/build_evidence.py`).
Templates in `doc/templates/` carry placeholder slots, which the compiler fills from
the Stage 0 to 2 manifests and metrics. It fails loudly if any placeholder is left
unfilled, so an evidence document with a blank where a number belongs is a build
failure rather than a to-do item. Compliance artefacts become a build product of the
pipeline, versioned alongside the code.

**A release gate that can say no** (`src/release_gate.py`). Seven technical checks
decide internal promotion: the formal budget, the empirical attack results, a
near-duplicate bound, the presence of fairness evidence, honest disclosure of utility
including negative results, and provenance. Negative-control tests
(`src/test_gate.py`) prove that the gate genuinely fails non-compliant candidates and
refuses to promote them. A gate that cannot fail is theatre.

**Human gates that code must never close.** External release additionally requires a
review of the data use agreement and a practitioner or governance review. The pipeline
holds these permanently open and reports the external decision as held. Automating a
legal judgement would be the single most dangerous thing this framework could do, so it
is structurally prevented rather than merely discouraged.

Stage 3 changes the process, not the distribution. The promoted datasets are
byte-identical copies of the Stage-2 release candidate, and the checksums prove it.
That is the honest scientific position: compliance is an evidentiary and procedural
layer, and to pretend it altered the data would be a fabrication.

## 4. What the framework produces

| Artefact | Standard followed | Generated from |
|---|---|---|
| [`doc/evidence/DPIA.md`](doc/evidence/DPIA.md) | UK ICO impact assessment, Steps 1 to 7 | manifests, metrics, attack results |
| [`doc/evidence/data-card.md`](doc/evidence/data-card.md) | Datasheets for Datasets (Gebru et al., 2021) | generation manifests |
| [`doc/evidence/model-card.md`](doc/evidence/model-card.md) | Model Cards (Mitchell et al., 2019) | evaluation metrics |
| [`doc/evidence/risk-register.md`](doc/evidence/risk-register.md) | residual-risk register, with owners and status | documented gaps in the guarantee |
| [`doc/evidence/regulatory-mapping.md`](doc/evidence/regulatory-mapping.md) | UK and EU GDPR, ICO guidance, EU AI Act | evidence-against-obligation table |
| `results/release_decision.json` | machine-readable audit record | the gate |

## 5. Results

The release decision for the Stage-2 candidate:

| Gate | Verdict | Evidence |
|---|---|---|
| Formal privacy, both tabular tasks | Pass | Budget spent within tolerance of the target |
| Empirical privacy, both tabular tasks | Pass | Attack AUC within the chance band, no near-duplicates |
| Fairness evidence | Pass | Representation audit and disparity metrics present for the released arm |
| Utility disclosure, including negatives | Pass | Weak results published rather than suppressed |
| Provenance | Pass | Upstream manifests and candidate files verified |
| **Internal promotion** | **Pass** | Three datasets promoted byte-identically, with checksums |
| **External release** | **Held** | The `dua` and `stakeholder` human gates remain open |

The negative controls pass: with the budget ceiling tightened below what the candidate
spent, the gate fails it and promotes nothing; with the chance band narrowed so that
the real attack results fall outside it, the empirical-privacy check fails and
promotes nothing; and the real candidate passes internally while external release
stays held. The gate is falsifiable.

**The finding that matters most is uncomfortable.** Everything this framework certifies
is procedural. It proves the release is well-evidenced and within budget. It does not
make the release useful, and the very same evidence package records that the compliant
dataset is scientifically weak, with a readmission AUC near chance and a negative
length-of-stay R². A dataset can be perfectly compliant and scientifically worthless.
This stage's tooling is what makes that fact visible instead of hidden. Restoring
utility is the realism pillar's job; the compliance layer's contribution is that it
will not permit a low-utility release to be dressed up as a good one.

## 6. Reproducibility

```bash
PY=python3   # any Python with requirements.txt installed

make evidence PY=$PY   # compile the impact assessment, cards, risk register and mapping
make gate     PY=$PY   # run the release gate; promote on a technical pass
make test     PY=$PY   # negative controls: prove the gate can fail
make all      PY=$PY   # all three; seconds, since no model is trained in this stage
```

Paths, budgets and gate thresholds live in [`config.yaml`](config.yaml). To reuse the
framework on another dataset, point the configuration at that dataset's manifests and
metrics and re-run. No template editing and no document rewriting are required.

## 7. Limitations

* **Not legal advice.** This is a research demonstration of a compliance methodology.
  It authorises no disclosure, and the contractual layer is a blocking human gate
  precisely because code must not adjudicate it.
* **Not yet validated by practitioners.** The `stakeholder` gate is open. Review by an
  NHS or ICB information-governance officer, or by a data protection officer, is the
  step that would turn a proposed framework into a validated one, and it is scheduled.
* **Compliance is neither utility nor safety.** The gate certifies evidence and budgets,
  not clinical value. The residual risks in the register, notably row-level rather than
  patient-level adjacency and the non-private preprocessing statistics, remain open and
  are disclosed rather than resolved.

## 8. Data use and ethics

PhysioNet Credentialed Health Data Use Agreement, MIMIC-IV v3.1. No real or synthetic
patient-level data is committed to this repository. The promoted datasets live outside
it and are not externally released; the framework itself holds them. The generator and
the models trained on its output are research artefacts, not medical devices, and must
not inform clinical care.

---

### References

Information Commissioner's Office. *Data Protection Impact Assessments*;
*Anonymisation, pseudonymisation and privacy-enhancing technologies*; *Guidance on AI
and data protection*.

Regulation (EU) 2024/1689 (the AI Act), Articles 6, 10, 11, 13 and 15.

UK GDPR, Articles 5, 25, 32 and 35, and Recital 26.

Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). *Datasheets for Datasets.*
Communications of the ACM.

Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). *Model Cards for Model Reporting.*
ACM FAccT.

Johnson, A., et al. *MIMIC-IV* v3.1, PhysioNet (2024).
