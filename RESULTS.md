# Publication results summary

The authoritative compact result summaries in this release are derived from the fixed MovieLens 1M publication protocol.

## Primary full-catalog NDCG@10

| Rank | Model | Mean | Sample SD |
|---:|---|---:|---:|
| 1 | LogisticPCA | 0.588825 | 0.001977 |
| 2 | BernoulliMF | 0.302618 | 0.000704 |
| 3 | p-emb | 0.290837 | 0.000550 |
| 4 | PoissonMF | 0.283416 | 0.001596 |
| 5 | b-emb | 0.283123 | 0.000422 |
| 6 | HPF | 0.266846 | 0.004070 |
| 7 | PCA/SVD | 0.162711 | 0.000403 |
| 8 | ap-emb | 0.131045 | 0.000188 |
| 9 | Popularity | 0.129276 | 0.000000 |
| 10 | GaussianMF | 0.077506 | 0.000547 |
| 11 | ng-emb | 0.076744 | 0.000281 |
| 12 | g-emb | 0.068306 | 0.000793 |
| 13 | Random | 0.005087 | 0.000280 |

## Preference-aware NDCG@10 (rating >= 4)

| Rank | Model | Mean | Sample SD |
|---:|---|---:|---:|
| 1 | LogisticPCA | 0.499410 | 0.000899 |
| 2 | BernoulliMF | 0.265526 | 0.000072 |
| 3 | p-emb | 0.262535 | 0.000424 |
| 4 | b-emb | 0.250906 | 0.000319 |
| 5 | PoissonMF | 0.243128 | 0.000928 |
| 6 | HPF | 0.237518 | 0.003465 |
| 7 | PCA/SVD | 0.169246 | 0.000638 |
| 8 | ap-emb | 0.119740 | 0.000107 |
| 9 | Popularity | 0.118698 | 0.000000 |
| 10 | GaussianMF | 0.081940 | 0.000481 |
| 11 | ng-emb | 0.076541 | 0.000339 |
| 12 | g-emb | 0.073758 | 0.000668 |
| 13 | Random | 0.003286 | 0.000114 |

The only rank exchange between the primary and preference-aware orderings is b-emb / PoissonMF.

## Gaussian EF-EMB validation-only normalization study

| Model | Normalization | NDCG@10 | RMSE | MAE |
|---|---|---:|---:|---:|
| g-emb | raw | 0.024038 | 1.014885 | 0.787349 |
| g-emb | sqrt | 0.063807 | 0.909133 | 0.715927 |
| g-emb | mean | 0.060606 | 0.958674 | 0.765777 |
| ng-emb | raw | 0.000000 | 10.857162 | 7.956222 |
| ng-emb | sqrt | 0.053875 | 1.166518 | 0.910279 |
| ng-emb | mean | 0.071496 | 1.034235 | 0.831691 |

This ablation uses validation data only and does not alter the final-test model definitions.

For complete multi-metric and per-seed summaries, see `results/step60-aggregation-v1.1/`.
