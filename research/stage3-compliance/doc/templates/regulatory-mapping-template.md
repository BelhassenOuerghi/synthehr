# Regulatory mapping: evidence against obligation

*Compiled {{GEN_DATE}}. This document maps each artefact the programme produces to
the obligation it discharges under UK GDPR, the ICO's anonymisation and AI guidance,
and the EU AI Act. The claim made here is methodological, namely that this is the
evidence a controller would need and that the mechanism produces it automatically. It
is not a legal opinion.*

## A. UK and EU GDPR

| Obligation | Evidence produced | Artefact |
|---|---|---|
| Article 35, impact assessment for high-risk processing | A full ICO-structured assessment, compiled from the pipeline | `DPIA.md` |
| Article 5(1)(c), data minimisation | The generator consumes model-ready features only, with no identifiers, free text or dates | Data card, composition section |
| Articles 5(1)(f) and 32, security of processing | A formal budget of {{EPS_READM}} at delta {{DELTA}}, together with empirical attacks at chance ({{MIA_RD_DCR}}) | Model card; assessment, Step 4 |
| Article 25, data protection by design | Privacy is applied at the generator, through a DP-SGD critic, rather than bolted on afterwards, and the release gate blocks non-compliant candidates | `release_gate.py`, `release_decision.json` |
| Recital 26, the identifiability test | Risk assessed by means reasonably likely to be used: a formal bound, attack results, and openly stated residual channels (RR-1 to RR-4) | Risk register |
| Article 5(2), accountability | Every figure in the evidence package is injected from the artefacts that generated the data, so the package is regenerable and version-controlled | `build_evidence.py`, `evidence_values.json` |

## B. ICO guidance

| Expectation | Evidence | Artefact |
|---|---|---|
| Anonymisation: identify the residual re-identification risk rather than assert that data are anonymous | Residual risks enumerated with severity and status, and the limits of adjacency stated (RR-1) | Risk register |
| Anonymisation: motivated-intruder testing | A membership-inference suite, comprising a distance-to-closest-record attack and a nearest-neighbour classifier, run against the actual release | Model card; `release_decision.json` |
| AI and data protection: fairness must be assessed, not assumed | Subgroup representation and disparity metrics for the released arm, with honest disclosure that the parity metrics are partly vacuous at this level of utility | Assessment, Step 4 (R4); model card |
| AI: explain the trade-offs made | The utility cost of privacy and of fairness measured per stage against fixed baselines | The stage working papers |

## C. EU AI Act (Regulation 2024/1689)

| Provision | Position and evidence |
|---|---|
| Article 6 and Annex III, classification | The generator is not itself a high-risk AI system. A model trained on its output for clinical or triage use would be, since that touches access to healthcare. The model card therefore prohibits clinical use, which keeps this artefact outside the high-risk regime by design. |
| Article 10, data governance | Should a downstream deployer proceed, this package supplies the evidence required by Article 10(2) to (5): provenance, preparation steps, examination for bias, and identified gaps and shortcomings, the last of these in the risk register. |
| Article 11 and Annex IV, technical documentation | The data card, model card, impact assessment, risk register and reproducible pipeline together constitute the documentation set. |
| Article 13, transparency to deployers | Utility limits, including negative results such as a length-of-stay R² of {{UTIL_REL_LOS}}, are stated plainly rather than presented as success. |
| Article 15, accuracy and robustness | Accuracy is characterised against a real-data ceiling. Robustness is limited by the run-to-run instability of the DP generator, which is disclosed and quantified rather than hidden. |

## D. Statement of limits

This mapping is a research demonstration of a compliance methodology. It is not legal
advice, it authorises nothing, and the contractual layer, being the data-use-agreement
constraints on redistributing derived data, is treated as a blocking human gate rather
than a box the pipeline can tick.
