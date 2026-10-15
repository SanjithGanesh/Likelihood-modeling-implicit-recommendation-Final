# Paper 2 — Step 61 Figure Captions

These captions are factual draft captions generated from the frozen Step 61 figure contract. Final journal-specific wording may be shortened later without changing the underlying results.

## F01 — Primary full-catalog ranking performance

Primary full-catalog NDCG@10 on the frozen MovieLens 1M test set. Points show the mean across seeds 2026, 2027, and 2028; error bars show the sample standard deviation across those three frozen seeds.

## F02 — Primary NDCG across ranking cutoffs

Primary NDCG at K = 5, 10, and 20 under the frozen full-catalog test evaluation. Error bars show sample standard deviation across three frozen seeds.

## F03 — Primary Recall across ranking cutoffs

Primary Recall at K = 5, 10, and 20 under the frozen full-catalog test evaluation. Error bars show sample standard deviation across three frozen seeds.

## F04 — Primary metric profile at K=10

Primary mean ranking-metric profile at K = 10 for all thirteen models. Values are descriptive means across the three frozen seeds.

## F05 — Primary versus preference-aware NDCG@10

NDCG@10 under the primary all-observed relevance definition and the separate rating >= 4 preference-aware sensitivity analysis. Error bars show sample standard deviation across three frozen seeds. The two evaluations use different relevance definitions and are compared descriptively.

## F06 — User-level preference-aware NDCG@10

Distribution across 5,851 eligible users of per-user preference-aware mean NDCG@10, where rating >= 4 defines relevance. The sensitivity analysis is secondary and does not replace the primary results.

## F07 — Gaussian EF-EMB ranking stability by context normalization

Validation NDCG@10 for g-emb and ng-emb across raw, square-root, and mean context normalization in the prespecified Gaussian EF-EMB stability ablation. Error bars show sample standard deviation across three frozen seeds; no test metrics are used in this ablation.

## F08 — Gaussian EF-EMB RMSE by context normalization

Validation RMSE for g-emb and ng-emb across context-normalization choices in the prespecified Gaussian EF-EMB stability ablation. The y-axis is logarithmic. Error bars show sample standard deviation across three frozen seeds.

## F09 — Gaussian EF-EMB context-norm scale by normalization

99th percentile of positive-training context-vector norm across context normalizations for g-emb and ng-emb. Points summarize the three stability seeds; the y-axis is logarithmic.

## F10 — Maximum pre-clipping gradient norm by normalization

Maximum pre-clipping gradient norm observed over training epochs in the Gaussian EF-EMB stability ablation. Points summarize the three stability seeds; the y-axis is logarithmic.

## F11 — MovieLens 1M rating distribution

Distribution of explicit ratings in the frozen MovieLens 1M dataset used for the publication benchmark.

## F12 — MovieLens 1M user-history length distribution

Distribution of MovieLens 1M interaction-history length by user. The x-axis is logarithmic to expose the heavy upper tail.

## F13 — ECDF of MovieLens 1M user-history lengths

Empirical cumulative distribution of MovieLens 1M user interaction-history length. The x-axis is logarithmic.

## F14 — Canonical MovieLens 1M split composition

Interaction counts in the frozen canonical MovieLens 1M train, validation, and test partitions.

## F15 — Relevant items per eligible user in preference sensitivity

Distribution of relevant test items per eligible user in the rating >= 4 preference-aware sensitivity analysis.

## F16 — Test-set rating distribution and preference threshold

Rating distribution in the frozen MovieLens 1M test partition. Ratings 4 and 5 define relevant items in the secondary preference-aware sensitivity analysis.

## S01 — Primary NDCG@10 by individual frozen seed

Primary test NDCG@10 for each individual frozen seed, shown without additional statistical inference.

## S02 — Preference-aware NDCG@10 by individual frozen seed

Preference-aware test NDCG@10 for each individual frozen seed in the secondary rating >= 4 sensitivity analysis.

## S03 — Model rank shift under preference-aware relevance

Descriptive model-rank shift from the primary all-observed evaluation to the separate preference-aware sensitivity analysis.

## S04 — Final-test Gaussian-family RMSE

Final-test RMSE for GaussianMF, g-emb, and ng-emb. Points show means across three frozen seeds and error bars show sample standard deviation.

## S05 — Final-test Gaussian-family MAE

Final-test MAE for GaussianMF, g-emb, and ng-emb. Points show means across three frozen seeds and error bars show sample standard deviation.

## S06 — Maximum validation prediction magnitude by normalization

Maximum absolute validation prediction magnitude across context normalizations in the Gaussian EF-EMB stability ablation. The y-axis is logarithmic.

## S07 — Maximum training context norm by normalization

Maximum positive-training context-vector norm across context normalizations in the Gaussian EF-EMB stability ablation. The y-axis is logarithmic.

## S08 — Candidate-set size distribution in preference sensitivity

Distribution of legal full-catalog candidate-set sizes across eligible users in the preference-aware sensitivity analysis.

## S09 — Primary Precision across ranking cutoffs

Primary Precision at K = 5, 10, and 20. Error bars show sample standard deviation across three frozen seeds.

## S10 — Primary MAP across ranking cutoffs

Primary MAP at K = 5, 10, and 20. Error bars show sample standard deviation across three frozen seeds.

## S11 — Primary MRR across ranking cutoffs

Primary MRR at K = 5, 10, and 20. Error bars show sample standard deviation across three frozen seeds.

## S12 — Primary Hit Rate across ranking cutoffs

Primary Hit Rate at K = 5, 10, and 20. Error bars show sample standard deviation across three frozen seeds.
