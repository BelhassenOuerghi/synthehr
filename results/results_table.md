# Pillar 4 (realism): results

Seeds [42, 43, 44, 45, 46], patient-disjoint splits identical to the evidence repository. Mean ± 95% CI over seeds. ε is in Stage 1's convention (ε_ar: the same runs under add/remove adjacency).

## los: utility, realism and privacy

| Arm | R² | detection AUC ↓ | precision ↑ | recall ↑ | coverage ↑ | target-assoc. err ↓ | DCR attack | TPR@1%FPR | ε | ε_ar |
|---|---|---|---|---|---|---|---|---|---|---|
| TRTR | 0.186 ± 0.001 | 0.511 ± 0.005 | 0.894 ± 0.007 | 0.892 ± 0.004 | 0.965 ± 0.005 | 0.006 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | – | – |
| ctgan | 0.017 ± 0.032 | 1.000 ± 0.000 | 0.592 ± 0.048 | 0.774 ± 0.052 | 0.605 ± 0.007 | 0.049 ± 0.016 | 0.500 ± 0.005 | 0.010 ± 0.002 | – | – |
| ddpm | 0.135 ± 0.005 | 0.639 ± 0.002 | 0.864 ± 0.011 | 0.881 ± 0.007 | 0.940 ± 0.008 | 0.011 ± 0.004 | 0.500 ± 0.006 | 0.009 ± 0.001 | – | – |
| ddpm_S | 0.128 ± 0.004 | 0.689 ± 0.014 | 0.841 ± 0.019 | 0.882 ± 0.006 | 0.909 ± 0.005 | 0.028 ± 0.022 | 0.500 ± 0.005 | 0.009 ± 0.001 | – | – |
| ddpm_L | 0.155 ± 0.000 | 0.599 ± 0.008 | 0.903 ± 0.007 | 0.868 ± 0.008 | 0.961 ± 0.006 | 0.007 ± 0.001 | 0.502 ± 0.006 | 0.010 ± 0.001 | – | – |
| dpctgan | -0.339 ± 0.329 | 1.000 ± 0.000 | 0.042 ± 0.051 | 0.014 ± 0.022 | 0.009 ± 0.011 | 0.112 ± 0.020 | 0.497 ± 0.007 | 0.011 ± 0.003 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_r1 | -1.358 ± 1.791 | 1.000 ± 0.000 | 0.142 ± 0.239 | 0.011 ± 0.010 | 0.020 ± 0.017 | 0.154 ± 0.080 | 0.499 ± 0.006 | 0.009 ± 0.003 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_r2 | -0.161 ± 0.279 | 1.000 ± 0.000 | 0.111 ± 0.192 | 0.005 ± 0.007 | 0.018 ± 0.029 | 0.147 ± 0.049 | 0.497 ± 0.003 | 0.010 ± 0.001 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_r3 | -1.125 ± 2.940 | 1.000 ± 0.000 | 0.160 ± 0.064 | 0.013 ± 0.009 | 0.017 ± 0.009 | 0.132 ± 0.061 | 0.499 ± 0.006 | 0.010 ± 0.001 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_ens4 | -0.284 ± 0.307 | 1.000 ± 0.000 | 0.112 ± 0.102 | 0.035 ± 0.025 | 0.031 ± 0.025 | 0.216 ± 0.091 | 0.497 ± 0.005 | 0.009 ± 0.002 | 22.75 ± 0.00 | 7.62 ± 0.00 |
| dpctgan_s1 | -0.210 ± 0.268 | 1.000 ± 0.000 | 0.145 ± 0.365 | 0.001 ± 0.002 | 0.003 ± 0.006 | 0.178 ± 0.108 | 0.498 ± 0.006 | 0.010 ± 0.001 | 4.65 ± 0.00 | 2.05 ± 0.00 |
| dpctgan_ens4split | -0.756 ± 0.939 | 1.000 ± 0.000 | 0.070 ± 0.075 | 0.014 ± 0.016 | 0.009 ± 0.004 | 0.332 ± 0.091 | 0.497 ± 0.006 | 0.010 ± 0.002 | 9.99 ± 0.00 | 4.24 ± 0.00 |
| dpctgan_rake | -0.443 ± 0.478 | 1.000 ± 0.000 | 0.043 ± 0.049 | 0.011 ± 0.011 | 0.009 ± 0.010 | 0.113 ± 0.022 | 0.497 ± 0.006 | 0.010 ± 0.002 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpddpm | -0.840 ± 1.113 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.994 ± 0.011 | 0.001 ± 0.001 | 0.181 ± 0.058 | 0.499 ± 0.006 | 0.010 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_r1 | -0.156 ± 0.245 | 1.000 ± 0.000 | 0.001 ± 0.001 | 0.989 ± 0.017 | 0.003 ± 0.004 | 0.160 ± 0.046 | 0.500 ± 0.007 | 0.009 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_r2 | -0.188 ± 0.172 | 1.000 ± 0.001 | 0.001 ± 0.001 | 0.993 ± 0.012 | 0.002 ± 0.002 | 0.169 ± 0.119 | 0.500 ± 0.007 | 0.010 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_r3 | -0.154 ± 0.187 | 0.999 ± 0.001 | 0.001 ± 0.002 | 0.990 ± 0.017 | 0.002 ± 0.005 | 0.164 ± 0.094 | 0.500 ± 0.006 | 0.010 ± 0.002 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_ens4 | -0.080 ± 0.062 | 1.000 ± 0.000 | 0.001 ± 0.001 | 0.993 ± 0.006 | 0.002 ± 0.001 | 0.145 ± 0.045 | 0.500 ± 0.006 | 0.009 ± 0.001 | 23.47 ± 0.00 | 9.42 ± 0.00 |
| dpddpm_s1 | -7.599 ± 4.239 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.716 ± 0.438 | 0.000 ± 0.000 | 0.120 ± 0.052 | 0.502 ± 0.004 | 0.010 ± 0.002 | 4.63 ± 0.00 | 2.18 ± 0.00 |
| dpddpm_ens4split | -4.889 ± 3.193 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.952 ± 0.132 | 0.000 ± 0.000 | 0.092 ± 0.038 | 0.502 ± 0.005 | 0.011 ± 0.001 | 10.00 ± 0.00 | 4.55 ± 0.00 |
| dpddpm_rake | -0.371 ± 0.446 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.999 ± 0.001 | 0.001 ± 0.001 | 0.179 ± 0.051 | 0.500 ± 0.005 | 0.011 ± 0.003 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_patient | -0.466 ± 0.265 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.148 ± 0.041 | 0.499 ± 0.006 | 0.010 ± 0.004 | 10.00 ± 0.00 | 4.41 ± 0.00 |
| dpddpm_S | 0.028 ± 0.028 | 0.990 ± 0.007 | 0.060 ± 0.056 | 0.925 ± 0.018 | 0.142 ± 0.124 | 0.182 ± 0.079 | 0.501 ± 0.007 | 0.010 ± 0.002 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_L | -10.195 ± 3.914 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.882 ± 0.322 | 0.000 ± 0.000 | 0.156 ± 0.055 | 0.500 ± 0.008 | 0.010 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |

**Noise floors (R²).** DP-CTGAN: mean |replicate diff| 0.942, max 5.058, MDE(n=5) 2.335, within-seed sd 1.426; DP-TabDDPM: mean |replicate diff| 0.712, max 2.105, MDE(n=5) 1.437, within-seed sd 0.527

**Spread (Brown-Forsythe).**

| Comparison | sd A | sd B | n A | n B | p |
|---|---|---|---|---|---|
| dpctgan single runs vs dpctgan_ens4 | 1.383 | 0.247 | 20 | 5 | 0.421 |
| dpctgan single runs vs dpctgan_ens4split | 1.383 | 0.756 | 20 | 5 | 0.837 |
| ctgan: non-private vs DP (seed sd) | 0.026 | 0.265 | 5 | 5 | 0.036 |
| dpddpm single runs vs dpddpm_ens4 | 0.525 | 0.050 | 20 | 5 | 0.322 |
| dpddpm single runs vs dpddpm_ens4split | 0.525 | 2.572 | 20 | 5 | 0.000 |
| ddpm: non-private vs DP (seed sd) | 0.004 | 0.896 | 5 | 5 | 0.138 |

**Reproduction of the evidence repository** (same seed, same recipe): ctgan: max |Δ| = 0.0868 over 5 seeds; dpctgan: max |Δ| = 2.6219 over 5 seeds; TRTR: max |Δ| = 0.0000 over 5 seeds

## readmission: utility, realism and privacy

| Arm | ROC-AUC | detection AUC ↓ | precision ↑ | recall ↑ | coverage ↑ | target-assoc. err ↓ | DCR attack | TPR@1%FPR | ε | ε_ar |
|---|---|---|---|---|---|---|---|---|---|---|
| TRTR | 0.778 ± 0.007 | 0.516 ± 0.006 | 0.861 ± 0.010 | 0.862 ± 0.014 | 0.966 ± 0.005 | 0.008 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | – | – |
| ctgan | 0.673 ± 0.019 | 1.000 ± 0.000 | 0.067 ± 0.019 | 0.663 ± 0.017 | 0.075 ± 0.031 | 0.062 ± 0.010 | 0.501 ± 0.005 | 0.011 ± 0.002 | – | – |
| ddpm | 0.741 ± 0.007 | 0.950 ± 0.010 | 0.184 ± 0.058 | 0.906 ± 0.015 | 0.456 ± 0.073 | 0.114 ± 0.027 | 0.502 ± 0.006 | 0.010 ± 0.002 | – | – |
| ddpm_S | 0.641 ± 0.014 | 1.000 ± 0.000 | 0.004 ± 0.011 | 1.000 ± 0.000 | 0.000 ± 0.001 | 0.178 ± 0.081 | 0.501 ± 0.007 | 0.011 ± 0.003 | – | – |
| ddpm_L | 0.751 ± 0.007 | 0.767 ± 0.020 | 0.750 ± 0.026 | 0.853 ± 0.007 | 0.896 ± 0.014 | 0.040 ± 0.022 | 0.501 ± 0.005 | 0.010 ± 0.001 | – | – |
| dpctgan | 0.492 ± 0.056 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.089 ± 0.012 | 0.501 ± 0.006 | 0.009 ± 0.002 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_r1 | 0.505 ± 0.071 | 1.000 ± 0.000 | 0.000 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.093 ± 0.012 | 0.501 ± 0.004 | 0.011 ± 0.002 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_r2 | 0.507 ± 0.057 | 1.000 ± 0.000 | 0.000 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.090 ± 0.009 | 0.500 ± 0.006 | 0.009 ± 0.002 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_r3 | 0.498 ± 0.066 | 1.000 ± 0.000 | 0.014 ± 0.039 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.093 ± 0.007 | 0.500 ± 0.005 | 0.010 ± 0.001 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpctgan_ens4 | 0.464 ± 0.047 | 1.000 ± 0.000 | 0.004 ± 0.010 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.064 ± 0.007 | 0.501 ± 0.006 | 0.011 ± 0.002 | 22.75 ± 0.00 | 7.62 ± 0.00 |
| dpctgan_s1 | 0.505 ± 0.089 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.098 ± 0.009 | 0.501 ± 0.005 | 0.011 ± 0.002 | 4.65 ± 0.00 | 2.05 ± 0.00 |
| dpctgan_ens4split | 0.491 ± 0.039 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.129 ± 0.099 | 0.501 ± 0.006 | 0.011 ± 0.002 | 9.99 ± 0.00 | 4.24 ± 0.00 |
| dpctgan_rake | 0.504 ± 0.056 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.076 ± 0.009 | 0.501 ± 0.007 | 0.010 ± 0.002 | 9.99 ± 0.00 | 3.57 ± 0.00 |
| dpddpm | 0.516 ± 0.039 | 1.000 ± 0.000 | 0.003 ± 0.009 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.077 ± 0.019 | 0.500 ± 0.005 | 0.010 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_r1 | 0.539 ± 0.063 | 1.000 ± 0.000 | 0.018 ± 0.049 | 1.000 ± 0.001 | 0.000 ± 0.000 | 0.116 ± 0.029 | 0.500 ± 0.010 | 0.010 ± 0.003 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_r2 | 0.506 ± 0.099 | 1.000 ± 0.000 | 0.006 ± 0.015 | 1.000 ± 0.001 | 0.000 ± 0.000 | 0.095 ± 0.034 | 0.500 ± 0.007 | 0.009 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_r3 | 0.501 ± 0.096 | 1.000 ± 0.000 | 0.015 ± 0.042 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.070 ± 0.034 | 0.499 ± 0.009 | 0.009 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_ens4 | 0.523 ± 0.084 | 1.000 ± 0.000 | 0.011 ± 0.029 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.084 ± 0.016 | 0.497 ± 0.007 | 0.010 ± 0.001 | 23.47 ± 0.00 | 9.42 ± 0.00 |
| dpddpm_s1 | 0.489 ± 0.056 | 1.000 ± 0.000 | 0.004 ± 0.011 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.083 ± 0.039 | 0.497 ± 0.006 | 0.010 ± 0.002 | 4.63 ± 0.00 | 2.18 ± 0.00 |
| dpddpm_ens4split | 0.516 ± 0.028 | 1.000 ± 0.000 | 0.005 ± 0.012 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.080 ± 0.025 | 0.497 ± 0.008 | 0.011 ± 0.002 | 10.00 ± 0.00 | 4.55 ± 0.00 |
| dpddpm_rake | 0.533 ± 0.047 | 1.000 ± 0.000 | 0.011 ± 0.031 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.080 ± 0.013 | 0.501 ± 0.008 | 0.009 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_patient | 0.518 ± 0.092 | 1.000 ± 0.000 | 0.015 ± 0.042 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.098 ± 0.042 | 0.499 ± 0.009 | 0.009 ± 0.001 | 10.00 ± 0.00 | 4.42 ± 0.00 |
| dpddpm_S | 0.538 ± 0.067 | 1.000 ± 0.000 | 0.013 ± 0.035 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.070 ± 0.019 | 0.500 ± 0.008 | 0.010 ± 0.001 | 10.00 ± 0.00 | 4.38 ± 0.00 |
| dpddpm_L | 0.504 ± 0.057 | 1.000 ± 0.000 | 0.001 ± 0.002 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.104 ± 0.016 | 0.503 ± 0.014 | 0.010 ± 0.002 | 10.00 ± 0.00 | 4.38 ± 0.00 |

**Noise floors (ROC-AUC).** DP-CTGAN: mean |replicate diff| 0.049, max 0.169, MDE(n=5) 0.114, within-seed sd 0.048; DP-TabDDPM: mean |replicate diff| 0.059, max 0.107, MDE(n=5) 0.120, within-seed sd 0.060

**Spread (Brown-Forsythe).**

| Comparison | sd A | sd B | n A | n B | p |
|---|---|---|---|---|---|
| dpctgan single runs vs dpctgan_ens4 | 0.047 | 0.038 | 20 | 5 | 0.558 |
| dpctgan single runs vs dpctgan_ens4split | 0.047 | 0.031 | 20 | 5 | 0.228 |
| ctgan: non-private vs DP (seed sd) | 0.016 | 0.045 | 5 | 5 | 0.234 |
| dpddpm single runs vs dpddpm_ens4 | 0.060 | 0.068 | 20 | 5 | 0.864 |
| dpddpm single runs vs dpddpm_ens4split | 0.060 | 0.022 | 20 | 5 | 0.058 |
| ddpm: non-private vs DP (seed sd) | 0.005 | 0.031 | 5 | 5 | 0.077 |

**Reproduction of the evidence repository** (same seed, same recipe): ctgan: max |Δ| = 0.0432 over 5 seeds; dpctgan: max |Δ| = 0.0609 over 5 seeds; TRTR: max |Δ| = 0.0000 over 5 seeds

**Fairness at restored utility** (readmission, insurance):

| Arm | label rate (real ≈ 0.139) | AUC gap | DP diff | EO diff | degenerate cells | Medicare share |
|---|---|---|---|---|---|---|
| TRTR | 0.139 ± 0.001 | 0.075 ± 0.033 | 0.309 ± 0.032 | 0.290 ± 0.029 | 0/35 | 0.542 ± 0.000 |
| ctgan | 0.199 ± 0.041 | 0.102 ± 0.058 | 0.272 ± 0.072 | 0.327 ± 0.089 | 0/35 | 0.504 ± 0.104 |
| ddpm | 0.375 ± 0.104 | 0.084 ± 0.029 | 0.059 ± 0.072 | 0.120 ± 0.093 | 19/35 | 0.379 ± 0.158 |
| ddpm_S | 0.573 ± 0.160 | 0.093 ± 0.060 | 0.074 ± 0.118 | 0.116 ± 0.161 | 27/35 | 0.347 ± 0.180 |
| ddpm_L | 0.155 ± 0.011 | 0.070 ± 0.033 | 0.302 ± 0.030 | 0.326 ± 0.088 | 0/35 | 0.526 ± 0.015 |
| dpctgan | 0.087 ± 0.012 | 0.121 ± 0.043 | 0.156 ± 0.066 | 0.176 ± 0.060 | 1/35 | 0.552 ± 0.389 |
| dpctgan_rake | 0.048 ± 0.020 | 0.108 ± 0.049 | 0.157 ± 0.124 | 0.180 ± 0.124 | 6/35 | 0.199 ± 0.002 |
| dpddpm | 0.114 ± 0.138 | 0.086 ± 0.016 | 0.001 ± 0.002 | 0.001 ± 0.002 | 35/35 | 0.592 ± 0.206 |
| dpddpm_rake | 0.097 ± 0.084 | 0.095 ± 0.042 | 0.000 ± 0.000 | 0.000 ± 0.000 | 35/35 | 0.297 ± 0.056 |
| dpddpm_S | 0.099 ± 0.073 | 0.148 ± 0.070 | 0.000 ± 0.001 | 0.001 ± 0.001 | 35/35 | 0.568 ± 0.141 |
| dpddpm_L | 0.349 ± 0.356 | 0.110 ± 0.035 | 0.003 ± 0.008 | 0.003 ± 0.009 | 35/35 | 0.251 ± 0.126 |

## Pre-registered primary comparisons (paired t-test, Holm)

| Task | Comparison | Δ | p | p (Holm) | noise floor | verdict |
|---|---|---|---|---|---|---|
| los | RQ1 realism without privacy: TabDDPM vs CTGAN | +0.118 | 0.0004 | 0.0050 | – | SIGNIFICANT (survives Holm) |
| los | RQ2 realism under privacy: DP-TabDDPM vs DP-CTGAN | -0.501 | 0.2842 | 1.0000 | 0.942 | within noise |
| los | RQ3b split-budget pool vs single run (CTGAN) | -0.417 | 0.2549 | 1.0000 | 0.942 | within noise |
| los | RQ3b split-budget pool vs single run (TabDDPM) | -4.049 | 0.0312 | 0.3116 | 0.712 | significant at p < 0.05, not after Holm |
| los | RQ5 cost of fair resampling (TabDDPM) | +0.469 | 0.1555 | 1.0000 | 0.712 | within noise |
| los | RQ6 patient-level vs stay-level DP (TabDDPM) | +0.374 | 0.3913 | 1.0000 | 0.712 | within noise |
| los | RQ4 capacity x privacy interaction: (L-S) without DP minus (L-S) with DP | +10.250 | 0.0019 | 0.0225 | 0.712 | SIGNIFICANT (survives Holm) |
| readmission | RQ1 realism without privacy: TabDDPM vs CTGAN | +0.068 | 0.0002 | 0.0028 | – | SIGNIFICANT (survives Holm) |
| readmission | RQ2 realism under privacy: DP-TabDDPM vs DP-CTGAN | +0.024 | 0.4329 | 1.0000 | 0.059 | within noise |
| readmission | RQ3b split-budget pool vs single run (CTGAN) | -0.001 | 0.9580 | 1.0000 | 0.049 | within noise |
| readmission | RQ3b split-budget pool vs single run (TabDDPM) | -0.001 | 0.9570 | 1.0000 | 0.059 | within noise |
| readmission | RQ5 cost of fair resampling (TabDDPM) | +0.017 | 0.3168 | 1.0000 | 0.059 | within noise |
| readmission | RQ6 patient-level vs stay-level DP (TabDDPM) | +0.002 | 0.9559 | 1.0000 | 0.059 | within noise |
| readmission | RQ4 capacity x privacy interaction: (L-S) without DP minus (L-S) with DP | +0.144 | 0.0071 | 0.0779 | 0.059 | significant at p < 0.05, not after Holm |

## Secondary comparisons (paired t-test, unadjusted)

| Task | Comparison | metric | Δ | p |
|---|---|---|---|---|
| los | realism (detection_auc): TabDDPM vs CTGAN | detection_auc | -0.361 | 0.0000 |
| los | realism (detection_auc): DP-TabDDPM vs DP-CTGAN | detection_auc | -0.000 | 0.0239 |
| los | realism (precision): TabDDPM vs CTGAN | precision | +0.272 | 0.0001 |
| los | realism (precision): DP-TabDDPM vs DP-CTGAN | precision | -0.041 | 0.0870 |
| los | realism (recall): TabDDPM vs CTGAN | recall | +0.107 | 0.0038 |
| los | realism (recall): DP-TabDDPM vs DP-CTGAN | recall | +0.981 | 0.0000 |
| los | realism (target_assoc_error): TabDDPM vs CTGAN | target_assoc_error | -0.038 | 0.0014 |
| los | realism (target_assoc_error): DP-TabDDPM vs DP-CTGAN | target_assoc_error | +0.069 | 0.0181 |
| los | cost of synthesis (TRTR vs TabDDPM) | r2 | +0.050 | 0.0000 |
| los | cost of privacy (TabDDPM vs DP-TabDDPM) | r2 | +0.975 | 0.0712 |
| los | cost of privacy (CTGAN vs DP-CTGAN) | r2 | +0.356 | 0.0370 |
| los | RQ3a full-budget pool vs single run (CTGAN) | r2 | +0.055 | 0.7165 |
| los | RQ3a full-budget pool vs single run (TabDDPM) | r2 | +0.760 | 0.1342 |
| los | cost of fair resampling (CTGAN) | r2 | -0.104 | 0.1475 |
| los | capacity S vs M, non-private | r2 | -0.007 | 0.0711 |
| los | capacity S vs M, DP | r2 | +0.868 | 0.0914 |
| los | capacity L vs M, non-private | r2 | +0.019 | 0.0005 |
| los | capacity L vs M, DP | r2 | -9.355 | 0.0029 |
| readmission | realism (detection_auc): TabDDPM vs CTGAN | detection_auc | -0.050 | 0.0001 |
| readmission | realism (detection_auc): DP-TabDDPM vs DP-CTGAN | detection_auc | +0.000 | 0.3742 |
| readmission | realism (precision): TabDDPM vs CTGAN | precision | +0.117 | 0.0089 |
| readmission | realism (precision): DP-TabDDPM vs DP-CTGAN | precision | +0.003 | 0.3872 |
| readmission | realism (recall): TabDDPM vs CTGAN | recall | +0.242 | 0.0000 |
| readmission | realism (recall): DP-TabDDPM vs DP-CTGAN | recall | +1.000 | 0.0000 |
| readmission | realism (target_assoc_error): TabDDPM vs CTGAN | target_assoc_error | +0.052 | 0.0049 |
| readmission | realism (target_assoc_error): DP-TabDDPM vs DP-CTGAN | target_assoc_error | -0.012 | 0.1922 |
| readmission | cost of synthesis (TRTR vs TabDDPM) | roc_auc | +0.037 | 0.0000 |
| readmission | cost of privacy (TabDDPM vs DP-TabDDPM) | roc_auc | +0.225 | 0.0001 |
| readmission | cost of privacy (CTGAN vs DP-CTGAN) | roc_auc | +0.181 | 0.0002 |
| readmission | RQ3a full-budget pool vs single run (CTGAN) | roc_auc | -0.028 | 0.1397 |
| readmission | RQ3a full-budget pool vs single run (TabDDPM) | roc_auc | +0.007 | 0.8172 |
| readmission | cost of fair resampling (CTGAN) | roc_auc | +0.012 | 0.0042 |
| readmission | capacity S vs M, non-private | roc_auc | -0.100 | 0.0001 |
| readmission | capacity S vs M, DP | roc_auc | +0.022 | 0.4023 |
| readmission | capacity L vs M, non-private | roc_auc | +0.010 | 0.0008 |
| readmission | capacity L vs M, DP | roc_auc | -0.013 | 0.3386 |
