# Compliance as a Build Product: an Executable Evidence Framework and Release Gate for Synthetic Electronic Health Records

Belhassen Ouerghi. Independent research, part of [SynthEHR](../../../README.md).

Working paper. Stage 3 (ethics and compliance pillar) of the programme *Advanced Data
Generation for Healthcare Predictive Modelling: Ensuring Privacy, High Realism, Bias
Mitigation and Ethical Compliance.*

---

## Abstract

A synthetic health dataset with a formal privacy bound and a fairness audit still
cannot lawfully leave a hospital. What is missing is evidence: the documented,
defensible assessment that UK GDPR (Articles 5(2), 25 and 35, and Recital 26), the
ICO's anonymisation guidance, and the EU AI Act's data-governance and
technical-documentation duties actually require. This paper contributes a reusable,
executable compliance framework for synthetic-EHR release, and demonstrates it on the
artefact produced by Stages 0 to 2 of this programme.

Two mechanisms carry the contribution. An evidence compiler generates an impact
assessment, a data card, a model card, a residual-risk register and a regulatory
mapping by injecting figures from the pipeline's own manifests and metrics, so that
the documents cannot drift from the data they describe, which is the standard failure
mode of hand-written compliance. A programmatic release gate then decides the
promotion of a candidate dataset against seven technical criteria, while external
release is held behind human gates that code is structurally forbidden to close.
Negative controls prove that the gate can fail, because a gate that cannot say no is
theatre. Applied to the Stage-2 candidate, the framework promotes it internally and
holds it externally. Notably, its own evidence package records that the compliant
release is scientifically weak, which is exactly the fact a compliance layer must
surface rather than obscure.

---

## 1. The problem: the gap between a guarantee and a release

Stages 1 and 2 delivered a differentially private, fairness-mitigated generator with a
formal budget and an audited subgroup profile. A controller asked to release its
output must nevertheless answer questions that no privacy budget answers. Has the
residual identifiability risk been assessed by means reasonably likely to be used?
Where is the record that fairness was examined? What are the known shortcomings? Who
signed off, on what evidence, and can that evidence be reproduced?

These are evidentiary and procedural duties, and in practice they fail in a
characteristic way, which is documentation drift. An impact assessment is written once,
by hand, from a snapshot of the pipeline. The pipeline is then re-run, and the document
silently ceases to describe the data. The compliance artefact becomes a fiction that is
nonetheless relied upon.

The position taken in this stage is that compliance evidence should be a build product
of the same pipeline that builds the data. If the numbers in the assessment are
injected from the manifests that generated the release, drift becomes structurally
impossible, and regeneration refreshes the evidence automatically.

## 2. Related practice, and what is missing

Datasheets for Datasets (Gebru et al., 2021) and Model Cards (Mitchell et al., 2019)
standardise what to document. ICO guidance and the AI Act specify which obligations
must be evidenced. Neither literature addresses how the documents stay true, and
neither supplies a decision procedure that can refuse a release. Meanwhile the
synthetic-health-data literature reports privacy and utility metrics but stops at the
paper, leaving the release decision itself implicit. This stage supplies both missing
pieces, and tests them adversarially.

## 3. Framework design

### 3.1 The evidence compiler

`src/build_evidence.py` reads the generation manifests and metric files from Stages 0
to 2, extracts around thirty audit-relevant values (the spent budget and noise
multiplier per released dataset, attack AUCs, near-duplicate shares, utility against
the real-data ceiling, subgroup representation before and after mitigation, and the
generator's hyper-parameters), and renders five templates. The compiler raises an
error if any placeholder is left unfilled: an evidence document with a blank where a
number belongs is treated as a build failure rather than as a to-do item. The package
is version-controlled next to the code, and `results/evidence_values.json` records
exactly what was injected.

### 3.2 The release gate

`src/release_gate.py` implements a two-tier decision.

Internal promotion, meaning whether the candidate may advance to the Stage-3 dataset
generation inside the credentialed environment, is decided by seven technical checks:
the formal budget per released dataset; membership-inference attack AUCs inside a
chance band; a near-duplicate share below threshold; the presence of fairness evidence
for the released arm; honest disclosure of utility, including negative results; and
provenance of the upstream manifests and candidate files. On a pass, the datasets are
copied byte-identically, checksummed, and accompanied by a provenance manifest.

External release, meaning whether the data may leave the environment, additionally
requires human gates: a review of the data use agreement, and a practitioner or
governance review. The code holds these permanently open and reports the external
decision as held. Automating a legal or clinical-governance judgement would be the most
dangerous thing this framework could do, so it is made structurally impossible rather
than merely discouraged.

Two design choices deserve emphasis. Honest disclosure is itself a gate criterion, so a
candidate whose negative results were suppressed would fail. And Stage 3 changes the
process, not the distribution: the promoted files are byte-identical to the Stage-2
candidate, and the checksums prove it. Any other choice would fabricate a data-level
effect that compliance does not have.

### 3.3 Falsifiability

A compliance gate that always passes is theatre. Three negative controls are therefore
run as tests (`src/test_gate.py`). Tightening the budget ceiling below what the
candidate spent must make the gate fail it, and must promote nothing. Narrowing the
chance band so that the real attack results fall outside it must make the
empirical-privacy check fail. And the real candidate must pass internally while external
release stays held and all human gates remain open. All three hold.

## 4. Regulatory mapping

Each artefact is mapped to the obligation it discharges. The impact assessment
addresses Article 35. Data minimisation addresses Article 5(1)(c). The privacy bound
together with the attack results addresses Articles 5(1)(f) and 32, and the ICO's
motivated-intruder expectation. Privacy by design addresses Article 25, since the
mechanism sits inside the generator and the gate blocks non-compliant candidates.
Regenerable evidence addresses accountability under Article 5(2). The residual-risk
register addresses the identifiability test of Recital 26.

For the AI Act the position is argued explicitly. The generator is not itself a
high-risk system, but a downstream model trained on its output for clinical or triage
use would be, since that touches access to healthcare. This is why the model card
prohibits clinical use: it keeps the artefact outside the high-risk regime by design,
while still supplying the evidence that a would-be deployer would need under Articles
10, 11 and 13. The full table is in `doc/evidence/regulatory-mapping.md`.

## 5. Results

Applied to the Stage-2 release candidate, all seven technical checks pass: the formal
budget is within its target for both tabular datasets; the empirical attacks fall inside
the chance band with no near-duplicate records; fairness evidence is present; utility is
disclosed including negative results; and provenance is verified. Three datasets are
promoted byte-identically to the Stage-3 folder, with checksums and a provenance
manifest. External release is held, because both human gates remain open.

The negative controls pass. The over-budget configuration fails and promotes nothing.
The narrowed-band configuration fails and promotes nothing. The real candidate passes
internally while external release remains held. The gate is falsifiable.

### 5.1 The uncomfortable finding, surfaced by the framework itself

Everything certified here is procedural. The same evidence package that records a clean
privacy bound and a rebalanced subgroup profile also records that the compliant release
is scientifically weak: readmission discrimination near chance, and a negative
length-of-stay R².

A synthetic dataset can be perfectly compliant and useless. That sentence is this
stage's contribution to the programme's argument as much as the tooling is. A compliance
framework reporting only privacy and fairness would certify this release and say nothing
about its worthlessness. A framework that forces joint disclosure of utility makes the
trade-off visible, and hands the problem to the realism pillar, where it belongs.

### 5.2 An unplanned test of the central claim

The framework's central claim is that compliance documents cannot drift, because they
are compiled from the artefacts rather than typed by hand. Shortly after this stage was
completed, that claim faced real drift. A data-leakage defect was found in one of the
upstream splits, the ladder was re-run on corrected splits with multiple seeds, and one
of Stage 2's headline findings was retracted. Several of the numbers that appear in the
impact assessment, the model card and the risk register changed as a result.

No document in `doc/evidence/` was edited by hand. Recompiling the package regenerated
them from the new artefacts. This is a stronger test of the anti-drift claim than the
one designed for it, and the framework passed it.

## 6. Contribution and limits

The contribution has three parts. Evidence as code, meaning compliance documents as a
build product, immune to drift, regenerable, and reusable on a new dataset by
re-pointing a configuration file. A falsifiable, two-tier release gate whose human gates
code cannot close. And a worked, honest regulatory mapping for synthetic health data,
including an explicit position on where the generator itself sits under the AI Act.

The limits are stated plainly. This is a methodology demonstration and not legal advice,
and it authorises nothing. The stakeholder gate is open: validation by a practitioner,
whether an NHS or ICB information-governance officer or a data protection officer, is
the step that would turn a proposed framework into a validated one, and it is scheduled.
The framework certifies evidence and budgets, not clinical value or absolute safety.
The residual risks in the register, notably row-level rather than patient-level
adjacency, non-private preprocessing statistics, and the run-to-run instability of the
generator under differential privacy, remain open by design and are disclosed rather
than resolved. Reusability is claimed here, and will be tested when the framework is
applied to a second dataset.

## 7. Relation to the ladder

The realism pillar must now do what compliance cannot, which is to restore utility at
the fixed budget so that the compliant release is also a useful one. Its success
criterion is already written into this stage's evidence package: the gap between the
promoted release and the non-private baseline, and between both and the real-data
ceiling. When that stage completes, this framework re-runs unchanged and re-issues the
evidence package for the new generation. That automatic re-issue is the practical proof
of the claim that compliance can be a build product.

---

### References

Information Commissioner's Office. *Data Protection Impact Assessments*;
*Anonymisation, pseudonymisation and privacy-enhancing technologies*; *Guidance on AI
and data protection*.

Regulation (EU) 2024/1689 (the AI Act), Articles 6, 10, 11, 13 and 15, and Annexes III
and IV.

UK GDPR, Articles 5, 25, 32 and 35, and Recital 26.

Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). *Datasheets for Datasets.*
Communications of the ACM.

Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). *Model Cards for Model Reporting.*
ACM FAccT.

Abadi, M., et al. (2016). *Deep Learning with Differential Privacy.* ACM CCS.

Johnson, A., Bulgarelli, L., Pollard, T., et al. (2024). *MIMIC-IV* (v3.1). PhysioNet.
