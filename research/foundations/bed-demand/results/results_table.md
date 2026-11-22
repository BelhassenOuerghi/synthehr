# Replication results: rolling 14-day-ahead backtest

## Target: admissions

- window: 2140-01-01 to 2149-12-31 (3653 days); mean 3.28, variance 3.21, **Var/Mean = 0.980** (marginal)
- conditional (Pearson) dispersion under Poisson-AR: **0.978**
- GP-AR dispersion parameter: lambda = -0.011 (implied conditional Var/Mean = 0.978)
- selected AR window W: GP-AR = 1, Poisson-AR = 1

| model | MAE | RMSE | 95% cov. | 50% cov. | 95% width | heldout ll/day (joint) | one-step ll | PIT MACE |
|---|---|---|---|---|---|---|---|---|
| gp_ar | 1.445 | 1.822 | 0.975 | 0.604 | 6.96 | -1.9951 | -1.9943 | 0.0118 |
| poisson_ar | 1.445 | 1.838 | 0.967 | 0.602 | 7.09 | -1.9949 | -1.9942 | 0.0143 |
| seasonal_naive | 2.058 | 2.465 | 0.967 | 0.604 | 7.97 | n/a | n/a | n/a |
| moving_average | 1.613 | 1.965 | 0.970 | 0.585 | 7.10 | n/a | n/a | n/a |

## Target: census

- window: 2140-01-01 to 2149-12-31 (3653 days); mean 11.59, variance 12.42, **Var/Mean = 1.071** (marginal)
- conditional (Pearson) dispersion under Poisson-AR: **0.430**
- GP-AR dispersion parameter: lambda = -0.505 (implied conditional Var/Mean = 0.442)
- selected AR window W: GP-AR = 5, Poisson-AR = 5

| model | MAE | RMSE | 95% cov. | 50% cov. | 95% width | heldout ll/day (joint) | one-step ll | PIT MACE |
|---|---|---|---|---|---|---|---|---|
| gp_ar | 2.849 | 3.555 | 0.956 | 0.530 | 12.66 | -2.4213 | -2.2984 | 0.0113 |
| poisson_ar | 3.029 | 3.625 | 1.000 | 0.687 | 18.86 | -2.6040 | -2.4060 | 0.0407 |
| seasonal_naive | 3.457 | 4.364 | 0.940 | 0.566 | 16.62 | n/a | n/a | n/a |
| moving_average | 3.055 | 3.788 | 0.959 | 0.558 | 14.60 | n/a | n/a | n/a |
