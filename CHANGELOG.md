# Changelog

## 1.0.0 (22 November 2026)

First public release of the tool and the research behind it.

* Generators: CTGAN, DP-CTGAN (with fair training and fair sampling), TabDDPM,
  DP-TabDDPM with record- or patient-level privacy, Poisson-AR for count series.
  All numerically identical to the research implementations (tested).
* Privacy: Rényi accountant for the sampled Gaussian mechanism, composition of
  pooled releases, membership-inference attacks with low-false-positive rates.
* Evaluation: train-on-synthetic utility against the real ceiling; realism
  (detection, precision/recall, density/coverage, marginals, correlations,
  label rate) against real held-out rows; fairness with a degeneracy check.
* Compliance: a release gate that can refuse (including a utility floor) and
  an evidence pack compiled from the measurements.
* Command line: `synthehr demo | fit | sample | evaluate | gate`.

Fixed relative to the research code: DP-CTGAN's fair variant now derives its
subgroup shares from the one accounted release of the category counts (the
research version drew a second, uncharged noisy copy; true epsilon 9.9922
against 9.9902 charged). Trained models no longer keep training rows.
