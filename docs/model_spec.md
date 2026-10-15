# Paper 2 — Frozen Model Specification
## Status
Specification revision: v1.2.
This revision preserves the frozen Ground Truth v1 data/evaluation contract while incorporating B91 (PoissonMF zero-sampling objective), B92 (ng-emb softplus consistency), and B93 (LogisticPCA literature alignment).
This document defines the mathematical meaning and common experimental rules for every model used in Paper 2.
The purpose of this specification is to freeze model semantics before implementation so that:
- model names correspond to the mathematics actually implemented;
- all models use the same canonical data split;
- validation and test information cannot leak into training;
- comparison protocols remain consistent across model families;
- differences between models are scientifically interpretable;
- implementation changes cannot silently alter the intended model.
No model implementation should be considered valid unless it matches this specification.
The development benchmark uses the frozen `ml-latest-small` Ground Truth v1 artifacts.
The publication benchmark will use MovieLens 1M after the development pipeline passes Gate B.
---
# 1. Common notation
Let
\[
\mathcal{U}
\]
denote the set of users and
\[
\mathcal{I}
\]
denote the set of items.
Let
\[
R
=
\{(u,i,r_{ui})\}
\]
denote the observed rating interactions.
For the primary implicit-ranking task, every observed rating is treated as a positive interaction regardless of rating value.
Define
\[
y_{ui}
=
\begin{cases}
1, & (u,i)\text{ is observed},\\
0, & (u,i)\text{ is genuinely unobserved}.
\end{cases}
\]
The primary implicit task therefore models interaction occurrence rather than explicit preference strength.
A secondary preference-aware sensitivity analysis may instead define positives using
\[
r_{ui} \geq 4.0.
\]
This sensitivity analysis remains separate from the primary all-observed interaction task.
---
# 2. Canonical split
All models must use the same frozen split.
For development:
- dataset: `ml-latest-small`
- artifact version: `v1`
- train rows: 81,200
- validation rows: 9,818
- test rows: 9,818
- users: 610
- items: 9,724
- split seed: 2026
- train-item coverage enforced: yes
Models are forbidden from generating an independent train/validation/test split.
The canonical development split fingerprint is:
```text
cb9f49a2342f6d4567c885bcdf495fab31116a7779961061622526d81d131cc9
```
---
# 3. Information boundaries
## 3.1 Training information
Model fitting may use only the training partition except where an explicitly frozen final retraining policy later permits train-plus-validation retraining.
Validation interactions must not contribute positive training signal or influence:
- parameter fitting;
- popularity statistics;
- feature construction;
- target construction;
- training-derived interaction-frequency statistics;
- hyperparameter selection other than validation evaluation.
Test interactions must never contribute positive training signal or model-selection information.
The identities of held-out validation/test interactions may be used only for leakage prevention.
For example, the shared negative sampler may know that a held-out pair is observed so that it is not incorrectly treated as a negative.
That knowledge must not be used to increase the held-out item's score during training.
---
## 3.2 Negative and zero handling
Whenever a training objective requires explicit negative examples, a negative pair for user \(u\) must satisfy
\[
(u,j)
\notin
R_{\mathrm{all}},
\]
where
\[
R_{\mathrm{all}}
=
R_{\mathrm{train}}
\cup
R_{\mathrm{validation}}
\cup
R_{\mathrm{test}}.
\]
Therefore, a validation or test positive can never be used as a training negative.
The same principle applies to models that operate on an implicit binary/count matrix.
Define a training-observation mask
\[
w_{ui}
=
\begin{cases}
0,
&
(u,i)
\in
R_{\mathrm{validation}}
\cup
R_{\mathrm{test}},
\\
1,
&
\text{otherwise}.
\end{cases}
\]
Held-out positive interactions are therefore absent from the training objective rather than being treated as zero observations.
The candidate item universe is restricted to the canonical training catalog.
Any popularity-based training statistic must be calculated from training interactions only.
---
# 4. Common ranking evaluation
The primary comparison across model families is full-catalog ranking.
For validation user \(u\),
\[
\mathcal{C}^{\mathrm{val}}_u
=
\mathcal{I}_{\mathrm{train}}
\setminus
\mathcal{I}^{\mathrm{train}}_u.
\]
Validation positives remain inside the candidate set.
The test partition is not consulted while validation metrics are calculated.
For final test evaluation,
\[
\mathcal{C}^{\mathrm{test}}_u
=
\mathcal{I}_{\mathrm{train}}
\setminus
\left(
\mathcal{I}^{\mathrm{train}}_u
\cup
\mathcal{I}^{\mathrm{validation}}_u
\right).
\]
Test positives remain inside the candidate set.
Every model produces a real-valued ranking score
\[
s_{ui}.
\]
Items are ranked by descending score.
Score ties are resolved deterministically using ascending item ID.
Primary ranking metrics are:
- Precision@K
- Recall@K
- HitRate@K
- NDCG@K
- MRR@K
- MAP@K
with
\[
K\in\{5,10,20\}.
\]
The primary validation-selection metric is
\[
\mathrm{NDCG@10}.
\]
---
# 5. Native family diagnostics
Full-catalog ranking is the common comparison layer.
Family-native diagnostics are supplementary.
## Bernoulli-family models
May report Bernoulli log-likelihood or binary cross-entropy on an explicitly defined evaluation set.
## Poisson-family models
May report Poisson likelihood diagnostics for the binary count-occurrence target.
## Gaussian-family models
May report explicit-rating prediction metrics:
\[
\mathrm{RMSE}
\]
and
\[
\mathrm{MAE}.
\]
Native likelihood values must not be directly compared across different likelihood families as though they represented the same quantity.
---
# 6. Latent-factor notation
For factorization models, let
\[
\mathbf{p}_u\in\mathbb{R}^{K}
\]
denote a user latent representation and
\[
\mathbf{q}_i\in\mathbb{R}^{K}
\]
denote an item latent representation.
The generic interaction term is
\[
\mathbf{p}_u^\top\mathbf{q}_i.
\]
When biases are present,
\[
\eta_{ui}
=
\mu
+
b_u
+
b_i
+
\mathbf{p}_u^\top\mathbf{q}_i.
\]
The exact interpretation of \(\eta_{ui}\) depends on the model family.
---
# 7. Random seeds
Model initialization, minibatch ordering, negative sampling, context subsampling, and other stochastic operations must use explicit deterministic random-number streams.
The global development seed is
\[
2026.
\]
Different stochastic mechanisms must use independently derived seeds.
Changing one random process must not unintentionally change unrelated processes.
Final publication experiments use the frozen final-seed set in the publication configuration.
---
# 8. Hyperparameter selection
Hyperparameters are selected using validation data only.
During tuning,
\[
\text{test access}
=
\text{forbidden}.
\]
Final test evaluation occurs only after:
1. model semantics are frozen;
2. the tuning space is frozen;
3. validation-based model selection is completed;
4. the selected hyperparameters are frozen.
---
# 9. Model inventory
The corrected project contains the following models.
## Universal baselines
1. Random
2. Popularity
## Linear decomposition baseline
3. PCA/SVD
## Matrix-factorization models
4. BernoulliMF
5. PoissonMF
6. GaussianMF
## Literature-based baselines
7. LogisticPCA
8. HPF
## Exponential-family embedding models
9. b-emb
10. p-emb
11. ap-emb
12. g-emb
13. ng-emb
The original terminology is retained only when the implementation corresponds to the stated mathematics.
---
# 10. Required semantic distinctions
## BernoulliMF versus LogisticPCA
BernoulliMF is an implicit user-item factorization model using positive interactions and explicitly sampled legal negatives.
LogisticPCA models a low-rank Bernoulli natural-parameter matrix with an item intercept.
They must not be aliases.
## PoissonMF versus HPF
PoissonMF is an optimization-based Poisson factorization with real-valued latent factors and an exponential link.
HPF is a hierarchical Bayesian Gamma-Poisson model.
They must not be aliases.
## p-emb versus ap-emb
These are mathematically different models.
p-emb uses a multiplicative Poisson mean induced by an exponentiated natural parameter.
ap-emb uses nonnegative embeddings and an additive Poisson mean.
An unused `additive` flag does not constitute a separate model.
## g-emb versus ng-emb
g-emb uses unconstrained real-valued target and context embeddings.
ng-emb uses nonnegative target and context embeddings obtained from unconstrained raw parameters through the frozen softplus transformation.
The same softplus transformation must be used identically during training and inference.
---
# 11. Model-output contract
Every recommender model exposes the conceptual interface
```text
fit(train_data)
score(user_id, candidate_items)
save(...)
load(...)
```
For candidate set \(\mathcal{C}_u\),
\[
|\mathbf{s}_u|
=
|\mathcal{C}_u|.
\]
Scores must contain no:
- NaN;
- positive infinity;
- negative infinity.
Higher score always means a higher ranking position.
---
# 12. Scientific interpretation rule
The corrected project does not assume in advance which model family will win.
Results determine the conclusions.
The implementation must never be modified merely to reproduce claims from the original report.
The study evaluates:
- likelihood/objective alignment;
- representation architecture;
- implicit versus explicit targets;
- context aggregation;
- Gaussian EF-EMB stability;
- sensitivity to context normalization;
- preference-aware relevance sensitivity.
---
# 13. General optimization rules
Trainable gradient-based models are implemented with PyTorch unless a model-specific inference method requires another algorithm.
All training procedures must:
- use deterministic initialization from derived seeds;
- keep losses and parameters finite;
- record the optimization configuration;
- use validation rather than test performance for selection;
- hard-fail on NaN or infinite outputs.
Learning rate, latent dimension, regularization, epoch count, and similar quantities are explicit configuration values.
---
# 14. Random baseline
For each pair \((u,i)\),
\[
s_{ui}=z_{ui},
\qquad
z_{ui}\sim\mathrm{Uniform}(0,1).
\]
The score is generated deterministically from:
- experiment seed;
- user ID;
- item ID;
- model namespace.
The same user-item pair under the same seed must always receive the same score.
Random has no fitting stage.
---
# 15. Popularity baseline
For item \(i\), define its training frequency
\[
c_i
=
\sum_u
\mathbb{1}
[(u,i)\in R_{\mathrm{train}}].
\]
The ranking score is
\[
s_{ui}=c_i.
\]
No validation or test interaction contributes to \(c_i\).
Popularity is calculated once per canonical split.
---
# 16. PCA/SVD baseline
For user \(u\), calculate the training-only mean rating
\[
\bar r_u
=
\frac{1}{|R_u^{\mathrm{train}}|}
\sum_{i:(u,i)\in R_{\mathrm{train}}}
r_{ui}.
\]
Construct a centered matrix \(X\):
\[
X_{ui}
=
r_{ui}-\bar r_u
\]
for observed training entries and
\[
X_{ui}=0
\]
for missing entries.
For latent rank \(K\),
\[
X
\approx
U_K\Sigma_KV_K^\top.
\]
The reconstructed prediction is
\[
\hat r_{ui}
=
\bar r_u
+
(U_K\Sigma_KV_K^\top)_{ui}.
\]
The ranking score is
\[
s_{ui}=\hat r_{ui}.
\]
PCA/SVD may additionally report RMSE and MAE.
---
# 17. BernoulliMF
## 17.1 Model
BernoulliMF uses:
- global intercept \(\mu\);
- user bias \(b_u\);
- item bias \(b_i\);
- user factors \(\mathbf p_u\);
- item factors \(\mathbf q_i\).
Define
\[
\eta_{ui}
=
\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i.
\]
Then
\[
p_{ui}
=
\sigma(\eta_{ui})
\]
and
\[
y_{ui}
\sim
\mathrm{Bernoulli}(p_{ui}).
\]
---
## 17.2 Training
Training positives are canonical training interactions.
Sampled negatives must satisfy
\[
(u,j)\notin R_{\mathrm{all}}.
\]
For positive set \(P\) and sampled-negative set \(N_s\),
\[
\mathcal L_{\mathrm{BMF}}
=
-
\frac{1}{|P|}
\sum_{(u,i)\in P}
\log p_{ui}
-
\lambda_{\mathrm{neg}}
\frac{1}{|N_s|}
\sum_{(u,j)\in N_s}
\log(1-p_{uj})
+
\mathcal R.
\]
Regularization is
\[
\mathcal R
=
\lambda_f
\left(
\sum_u\|\mathbf p_u\|_2^2+
\sum_i\|\mathbf q_i\|_2^2
\right)
+
\lambda_b
\left(
\sum_u b_u^2+
\sum_i b_i^2
\right).
\]
Implementation uses numerically stable logits-based binary cross-entropy.
---
## 17.3 Ranking score
\[
s_{ui}
=
\eta_{ui}.
\]
Ranking by logit is equivalent to ranking by probability.
---
# 18. PoissonMF
## 18.1 Target
The primary target is binary interaction occurrence interpreted as a count:
\[

y_{ui}\in\{0,1\}.

\]
It is not the MovieLens star rating.
Training positives are exactly the canonical training interactions.
Held-out validation/test positives are masked and never treated as zero counts.
---
## 18.2 Model
PoissonMF uses:
- global intercept \(\mu\);
- user bias \(b_u\);
- item bias \(b_i\);
- user factors \(\mathbf p_u\);
- item factors \(\mathbf q_i\).
Define the log-rate
\[

\eta_{ui}

=

\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i.

\]
The conceptual Poisson rate is
\[

\lambda_{ui}

=

\exp(\eta_{ui}).

\]
A documented finite log-rate guard may be applied immediately before exponentiation for numerical safety.
Any such guard must use the same configured bounds whenever a Poisson rate is evaluated during training or diagnostics, and clipping activity must be observable in run diagnostics rather than silently changing across runs.
---
## 18.3 Legal training cells
For user \(u\), define the genuine-zero set inside the canonical training catalog as
\[

Z_u

=

\left\{

j\in\mathcal I_{\mathrm{train}}
:

(u,j)\notin R_{\mathrm{all}}

\right\}.

\]
Therefore the conceptual Poisson training domain consists of:
- every canonical training positive \((u,i)\in R_{\mathrm{train}}\), with \(y_{ui}=1\); and
- every legal zero \((u,j)\) with \(j\in Z_u\), with \(y_{uj}=0\).
Validation/test positives are in neither category.
The primary PoissonMF model must never use a held-out observed interaction as a zero-count training example.
---
## 18.4 Full conceptual objective
For one legal training cell,
\[

\ell_{ui}

=

\lambda_{ui}

-

y_{ui}\log\lambda_{ui}.

\]
The constant \(\log(y_{ui}!)\) is omitted because \(y_{ui}\in\{0,1\}\).
Define
\[

M

=

|R_{\mathrm{train}}|
+
\sum_u |Z_u|.

\]
The primary conceptual data objective is
\[

\mathcal L_{\mathrm{PMF}}^{\mathrm{full}}

=

\frac{1}{M}
\left[

\sum_{(u,i)\in R_{\mathrm{train}}}
\left(
\lambda_{ui}-\log\lambda_{ui}
\right)

+

\sum_u
\sum_{j\in Z_u}
\lambda_{uj}

\right]

+

\mathcal R_{\mathrm{PMF}}.

\]
This is the objective that the corrected PoissonMF implementation represents.
The primary model does not replace the full-zero term with an arbitrary unweighted sampled-negative surrogate.
---
## 18.5 Sparse unbiased zero estimator
Enumerating every zero cell at every update is unnecessary.
All positive training interactions remain exact.
Legal zero cells may be subsampled through the shared leakage-safe negative sampler.
For user \(u\), suppose \(m_u>0\) zero items are sampled uniformly with replacement:
\[

j_1,\ldots,j_{m_u}

\sim

\mathrm{Uniform}(Z_u).

\]
Then the user's full zero-loss sum
\[

\sum_{j\in Z_u}\lambda_{uj}

\]
is estimated by
\[

\widehat S_u^{(0)}

=

\frac{|Z_u|}{m_u}
\sum_{s=1}^{m_u}
\lambda_{u j_s}.

\]
Therefore,
\[

\mathbb E
\left[
\widehat S_u^{(0)}
\right]

=

\sum_{j\in Z_u}\lambda_{uj}.

\]
If a nonuniform proposal distribution \(q_u(j)\) is used instead, the unbiased estimator is
\[

\widehat S_u^{(0)}

=

\frac{1}{m_u}
\sum_{s=1}^{m_u}
\frac{
\lambda_{u j_s}
}{
q_u(j_s)
}.

\]
The primary implementation may use a configurable number of zero draws per training positive for computational efficiency, but the resulting zero term must be rescaled so that it remains an unbiased estimator of the frozen full-zero objective.
A configuration that intentionally downweights zeros defines a separate named ablation and must not silently replace primary PoissonMF.
---
## 18.6 Regularization
The regularizer is
\[

\mathcal R_{\mathrm{PMF}}

=

\lambda_f
\left(
\sum_u\|\mathbf p_u\|_2^2
+
\sum_i\|\mathbf q_i\|_2^2
\right)

+

\lambda_b
\left(
\sum_u b_u^2
+
\sum_i b_i^2
\right).

\]
The global intercept is not L2-regularized unless a separately documented configuration explicitly states otherwise.
---
## 18.7 Numerical-stability requirements
The implementation must:
- keep every evaluated Poisson rate finite and strictly positive;
- use the same rate calculation in training and diagnostics;
- hard-fail on NaN or infinite loss, rate, parameter, score, or gradient after stabilization;
- record any configured log-rate guard and gradient-clipping threshold in run metadata.
The numerical guard is a stability mechanism, not a license to change the statistical objective between configurations without documentation.
---
## 18.8 Ranking score
The ranking score is the unconstrained log-rate
\[

s_{ui}

=

\eta_{ui}.

\]
Because the conceptual exponential link is monotonic, ranking by log-rate is equivalent to ranking by the conceptual Poisson rate.
---
# 19. GaussianMF
GaussianMF uses the original MovieLens rating target.
The prediction is
\[
\hat r_{ui}
=
\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i.
\]
The model assumes
\[
r_{ui}
\sim
\mathcal N(\hat r_{ui},\sigma^2).
\]
For fixed variance, training minimizes
\[
\mathcal L_{\mathrm{GMF}}
=
\frac{1}{|R_{\mathrm{train}}|}
\sum_{(u,i,r_{ui})\in R_{\mathrm{train}}}
(r_{ui}-\hat r_{ui})^2
+
\mathcal R.
\]
The global mean is initialized using training ratings only:
\[
\mu_0
=
\frac{1}{|R_{\mathrm{train}}|}
\sum r_{ui}.
\]
The ranking score is
\[
s_{ui}
=
\hat r_{ui}.
\]
Native diagnostics are RMSE and MAE.
---
# 20. LogisticPCA
## 20.1 Binary matrix
Define
\[
X\in\{0,1\}^{|\mathcal U|\times|\mathcal I|}.
\]
Training positives have
\[
X_{ui}=1.
\]
Genuinely unobserved pairs have
\[
X_{ui}=0.
\]
Validation/test positives are masked using \(W\) and do not contribute as zeros.
---
## 20.2 Natural-parameter matrix
\[
X_{ui}
\sim
\mathrm{Bernoulli}
\left(
\sigma(\Theta_{ui})
\right).
\]
The low-rank natural-parameter matrix is
\[
\Theta
=
\mathbf1\boldsymbol\mu^\top
+
AB^\top.
\]
Therefore,
\[
\Theta_{ui}
=
\mu_i
+
\mathbf a_u^\top\mathbf b_i.
\]
---
## 20.3 Low-rank LogisticPCA parameterization

Paper 2 uses the matrix-factorization formulation of Logistic PCA associated
with Schein, Saul, and Ungar (2003).

The Bernoulli natural-parameter matrix is

\[
\Theta
=
\mathbf 1\boldsymbol\mu^\top
+
AB^\top.
\]

Here:

- \(A\in\mathbb R^{|\mathcal U|\times K}\) contains user/observation coefficients;
- \(B\in\mathbb R^{|\mathcal I|\times K}\) contains item/dimension basis vectors;
- \(\boldsymbol\mu\in\mathbb R^{|\mathcal I|}\) is the item intercept vector.

The factorization \(AB^\top\) is not unique under invertible changes of latent
coordinates. This identifiability ambiguity does not alter the natural-parameter
matrix \(\Theta\) or the resulting ranking scores.

The primary LogisticPCA implementation therefore does not impose an additional
orthogonality or centering constraint on \(A\).

Any QR/SVD operation used internally for numerical conditioning must preserve
the represented matrix \(AB^\top\); it must not silently define a different
statistical model.
---
## 20.4 Objective

\[
\mathcal L_{\mathrm{LPCA}}
=
-
\frac{1}{\sum_{u,i}w_{ui}}
\sum_{u,i}
w_{ui}
\left[
X_{ui}\log p_{ui}
+
(1-X_{ui})\log(1-p_{ui})
\right],
\]

where

\[
p_{ui}
=
\sigma(\Theta_{ui}).
\]

The primary LogisticPCA baseline does not apply an L2 penalty to \(A\),
\(B\), or \(\boldsymbol\mu\).

This preserves the likelihood-based Schein, Saul, and Ungar (2003)
low-rank Bernoulli formulation without introducing scale-dependent
asymmetric regularization into the non-identifiable factorization
\(AB^\top\).

Implementation uses numerically stable logits-based binary
cross-entropy.

The mask \(W\) is a Paper 2 benchmark adaptation required to prevent
held-out validation/test positives from being incorrectly treated as
binary zeros. It changes the set of likelihood terms that are observed,
but does not change the LogisticPCA natural-parameter model
\(\Theta=\mathbf1\boldsymbol\mu^\top+AB^\top\).
---
## 20.5 Ranking
\[
s_{ui}
=
\Theta_{ui}.
\]
LogisticPCA uses all legal binary zeros under the mask rather than the BernoulliMF sampled-negative objective.
---
# 21. Hierarchical Poisson Factorization
## 21.1 Gamma notation
\[
X\sim\mathrm{Gamma}(\alpha,\rho)
\]
uses shape \(\alpha\) and rate \(\rho\).
Therefore,
\[
\mathbb E[X]
=
\frac{\alpha}{\rho}.
\]
---
## 21.2 User hierarchy
For user \(u\),
\[
\xi_u
\sim
\mathrm{Gamma}
\left(
a_0,
\frac{a_0}{b_0}
\right).
\]
For component \(k\),
\[
\theta_{uk}
\sim
\mathrm{Gamma}(a,\xi_u).
\]
---
## 21.3 Item hierarchy
For item \(i\),
\[
\eta_i
\sim
\mathrm{Gamma}
\left(
c_0,
\frac{c_0}{d_0}
\right).
\]
For component \(k\),
\[
\beta_{ik}
\sim
\mathrm{Gamma}(c,\eta_i).
\]
---
## 21.4 Observation model
\[
y_{ui}
\sim
\mathrm{Poisson}
\left(
\boldsymbol\theta_u^\top
\boldsymbol\beta_i
\right).
\]
The rate is
\[
\lambda_{ui}
=
\sum_{k=1}^{K}
\theta_{uk}\beta_{ik}.
\]
Validation/test positives are masked from the training likelihood.
HPF uses no synthetic negative sampler.
---
## 21.5 Default priors
The default literature-aligned settings are
\[
a_0=a=c_0=c=0.3
\]
and
\[
b_0=d_0=1.0.
\]
Latent dimension \(K\) remains tunable.
---
## 21.6 Variational approximation
Use Gamma mean-field factors:
\[
q(\theta_{uk})
=
\mathrm{Gamma}
(
\gamma^{\mathrm{shp}}_{uk},
\gamma^{\mathrm{rte}}_{uk}
),
\]
\[
q(\beta_{ik})
=
\mathrm{Gamma}
(
\lambda^{\mathrm{shp}}_{ik},
\lambda^{\mathrm{rte}}_{ik}
),
\]
\[
q(\xi_u)
=
\mathrm{Gamma}
(
\kappa^{\mathrm{shp}}_u,
\kappa^{\mathrm{rte}}_u
),
\]
and
\[
q(\eta_i)
=
\mathrm{Gamma}
(
\tau^{\mathrm{shp}}_i,
\tau^{\mathrm{rte}}_i
).
\]
For Gamma shape-rate parameters,
\[
\mathbb E[\log X]
=
\psi(\alpha)-\log\rho.
\]
---
## 21.7 Latent-count allocation
For positive interaction \((u,i)\),
\[
\phi_{uik}
\propto
\exp
\left(
\mathbb E[\log\theta_{uk}]
+
\mathbb E[\log\beta_{ik}]
\right).
\]
The vector is normalized across \(k\) using log-sum-exp.
---
## 21.8 User-factor updates
\[
\gamma^{\mathrm{shp}}_{uk}
=
a
+
\sum_{i:(u,i)\in R_{\mathrm{train}}}
y_{ui}\phi_{uik},
\]
\[
\gamma^{\mathrm{rte}}_{uk}
=
\mathbb E[\xi_u]
+
\sum_{i:w_{ui}=1}
\mathbb E[\beta_{ik}].
\]
---
## 21.9 User-activity updates
\[
\kappa^{\mathrm{shp}}_u
=
a_0+Ka,
\]
\[
\kappa^{\mathrm{rte}}_u
=
\frac{a_0}{b_0}
+
\sum_k
\mathbb E[\theta_{uk}].
\]
---
## 21.10 Item-factor updates
\[
\lambda^{\mathrm{shp}}_{ik}
=
c
+
\sum_{u:(u,i)\in R_{\mathrm{train}}}
y_{ui}\phi_{uik},
\]
\[
\lambda^{\mathrm{rte}}_{ik}
=
\mathbb E[\eta_i]
+
\sum_{u:w_{ui}=1}
\mathbb E[\theta_{uk}].
\]
---
## 21.11 Item-popularity updates
\[
\tau^{\mathrm{shp}}_i
=
c_0+Kc,
\]
\[
\tau^{\mathrm{rte}}_i
=
\frac{c_0}{d_0}
+
\sum_k
\mathbb E[\beta_{ik}].
\]
All Gamma parameters must remain finite and strictly positive.
---
## 21.12 Ranking score
\[
s_{ui}
=
\sum_{k=1}^{K}
\mathbb E[\theta_{uk}]
\mathbb E[\beta_{ik}].
\]
---
# 22. Shared EF-EMB framework
This section freezes the common representation used by:
- b-emb;
- p-emb;
- ap-emb;
- g-emb;
- ng-emb.
---
## 22.1 Target and context embeddings
Each item \(i\) has two conceptually distinct latent vectors.
The target embedding is
\[
\mathbf t_i\in\mathbb R^K.
\]
The context embedding is
\[
\mathbf c_i\in\mathbb R^K.
\]
The two embedding tables must remain distinct parameters unless a separately named ablation explicitly ties them.
---
## 22.2 Training history
For user \(u\), define the canonical training history
\[
H_u
=
\{
j:
(u,j)\in R_{\mathrm{train}}
\}.
\]
Only canonical training interactions may appear in \(H_u\).
Validation and test interactions must never be inserted into the context during ordinary model development.
---
## 22.3 Binary context values
The primary Paper 2 EF-EMB architecture uses interaction presence as its context value.
For
\[
j\in H_u,
\]
define
\[
x^{\mathrm{ctx}}_{uj}=1.
\]
Thus each training-history item contributes its context embedding once.
Explicit rating magnitude is not used as a context weight in the primary EF-EMB comparison.
Gaussian variants still use the explicit star rating as their prediction target.
Any future rating-weighted-context experiment must be separately named and reported as an ablation.
---
## 22.4 Positive-target context
For a positive training target \(i\in H_u\),
\[
C^{+}_{ui}
=
H_u\setminus\{i\}.
\]
Therefore,
\[
i\notin C^{+}_{ui}.
\]
The target item is never permitted to predict itself.
This rule applies to all five EF-EMB variants.
---
## 22.5 Unobserved-target context
For a legal unobserved target \(j\),
\[
(u,j)\notin R_{\mathrm{all}},
\]
define
\[
C^{-}_{uj}
=
H_u.
\]
Because \(j\notin H_u\), target exclusion occurs automatically.
---
## 22.6 Evaluation context
For a validation or test candidate item \(i\), the model context is the history available to the fitted model.
Under the standard train-only model,
\[
C^{\mathrm{eval}}_{ui}
=
H_u.
\]
Validation/test interactions are not inserted into this context.
If a later final-evaluation policy explicitly refits the model on train-plus-validation, that refitted model may construct its history from its new fitting data.
Such a policy must be frozen separately before final publication runs.
---
## 22.7 Raw context aggregation
Given context set \(C\),
\[
\mathbf h_{\mathrm{raw}}(C)
=
\sum_{j\in C}
\mathbf c_j.
\]
This is the raw-sum context representation.
---
## 22.8 Context normalization family
Define
\[
\mathbf h_q(C)
=
\frac{
\sum_{j\in C}\mathbf c_j
}{
|C|^q
}
\]
for nonempty \(C\).
The three frozen normalization modes are:
### Raw
\[
q=0.
\]
Therefore,
\[
\mathbf h_{\mathrm{raw}}
=
\sum_{j\in C}\mathbf c_j.
\]
### Square-root
\[
q=\frac12.
\]
Therefore,
\[
\mathbf h_{\mathrm{sqrt}}
=
\frac{1}{\sqrt{|C|}}
\sum_{j\in C}\mathbf c_j.
\]
### Mean
\[
q=1.
\]
Therefore,
\[
\mathbf h_{\mathrm{mean}}
=
\frac{1}{|C|}
\sum_{j\in C}\mathbf c_j.
\]
The primary corrected EF-EMB configuration uses
```text
context_normalization: mean
```
for all five EF-EMB models.
Raw and square-root normalization are reserved for the controlled context-normalization/stability ablation.
---
## 22.9 Empty context
If
\[
|C|=0,
\]
define
\[
\mathbf h(C)
=
\mathbf0.
\]
The production datasets are expected to provide nonempty contexts after filtering, but the behavior remains explicitly defined for synthetic tests.
---
## 22.10 Context ordering
Each user's training history is stored as an immutable sorted sequence of canonical item indices.
Set membership and aggregation must not depend on:
- Pandas row ordering;
- Python hash ordering;
- file-system ordering.
---
## 22.11 Context cap
The primary model uses
```text
context_pair_cap: null
```
and
```text
max_context_items: null
```
meaning that the complete legal training history is used.
There is no hidden 2,000-pair cap in the corrected primary implementation.
The previous explicit pair-materialization mechanism is replaced by exact aggregated contexts.
If a future experiment specifies a finite context length \(L\), it must be recorded in the configuration.
For deterministic capped contexts:
1. start from the user's sorted training history;
2. construct a deterministic per-user priority ordering using a derived context seed;
3. for a target item, exclude the target first;
4. take the first \(L\) legal items.
The cap must never be silently applied.
---
## 22.12 Efficient exact aggregation
The corrected implementation must not explicitly materialize every target-context pair.
For each user, calculate a cached context sum
\[
\mathbf S_u
=
\sum_{j\in H_u}
\mathbf c_j.
\]
For a positive target \(i\),
\[
\mathbf S^{(+)}_{ui}
=
\mathbf S_u-\mathbf c_i.
\]
For a legal unobserved or evaluation target,
\[
\mathbf S^{(-)}_{ui}
=
\mathbf S_u.
\]
The appropriate normalization denominator is then applied.
This produces the exact context representation without \(O(|H_u|^2)\) pair materialization.
---
## 22.13 Shared item intercept
Where a model uses an item intercept, denote it
\[
b_i.
\]
EF-EMB does not use a learned user-bias table.
User-specific information is represented through the aggregated context.
This preserves the architectural distinction between:
- user-factor MF;
- history/context-based EF-EMB.
---
# 23. Bernoulli EF-EMB — b-emb
## 23.1 Purpose
b-emb models implicit interaction occurrence using a Bernoulli conditional distribution and a context-derived user representation.
---
## 23.2 Natural parameter
Using the selected normalized context
\[
\mathbf h_{ui},
\]
define
\[
\eta^{(b)}_{ui}
=
b_i
+
\mathbf t_i^\top
\mathbf h_{ui}.
\]
---
## 23.3 Bernoulli probability
\[
p_{ui}
=
\sigma
\left(
\eta^{(b)}_{ui}
\right).
\]
Thus,
\[
y_{ui}
\sim
\mathrm{Bernoulli}(p_{ui}).
\]
---
## 23.4 Training positives
For each canonical positive interaction,
\[
(u,i)\in R_{\mathrm{train}},
\]
the target is
\[
y_{ui}=1
\]
and the context is
\[
H_u\setminus\{i\}.
\]
---
## 23.5 Training negatives
b-emb uses the shared leakage-safe negative sampler.
Every sampled negative satisfies
\[
(u,j)\notin R_{\mathrm{all}}.
\]
Its target is
\[
y_{uj}=0
\]
and its context is
\[
H_u.
\]
The negative-sampling ratio is an explicit configuration value.
---
## 23.6 Objective
For positive examples \(P\) and sampled negatives \(N_s\),
\[
\mathcal L_{b}
=
-
\frac1{|P|}
\sum_{(u,i)\in P}
\log
\sigma(\eta^{(b)}_{ui})
\]
\[
-
\lambda_{\mathrm{neg}}
\frac1{|N_s|}
\sum_{(u,j)\in N_s}
\log
\sigma(-\eta^{(b)}_{uj})
+
\mathcal R_b.
\]
The regularizer is
\[
\mathcal R_b
=
\lambda_t
\sum_i
\|\mathbf t_i\|_2^2
+
\lambda_c
\sum_i
\|\mathbf c_i\|_2^2
+
\lambda_b
\sum_i
b_i^2.
\]
Implementation uses logits-based BCE.
---
## 23.7 Ranking score
\[
s_{ui}
=
\eta^{(b)}_{ui}.
\]
Ranking by logit is equivalent to ranking by Bernoulli probability.
---
# 24. Poisson EF-EMB — p-emb
## 24.1 Purpose
p-emb models implicit interaction occurrence as a Poisson count.
For the primary Paper 2 task,
\[
y_{ui}\in\{0,1\}.
\]
The model uses unconstrained real-valued target and context embeddings.
---
## 24.2 Log-rate
Define
\[
\eta^{(p)}_{ui}
=
b_i
+
\mathbf t_i^\top
\mathbf h_{ui}.
\]
The Poisson rate is
\[
\lambda^{(p)}_{ui}
=
\exp
\left(
\eta^{(p)}_{ui}
\right).
\]
Thus context contributions act multiplicatively on the Poisson mean.
A documented finite log-rate clamp may be used before exponentiation for numerical stability.
The exact same scoring transformation must be used during training and inference.
---
## 24.3 Conditional distribution
\[
y_{ui}
\sim
\mathrm{Poisson}
\left(
\lambda^{(p)}_{ui}
\right).
\]
For a positive,
\[
y_{ui}=1.
\]
For a genuine unobserved zero,
\[
y_{ui}=0.
\]
Held-out validation/test positives never contribute as zeros.
---
## 24.4 Full conceptual objective
Ignoring the constant term,
\[
\ell^{(p)}_{ui}
=
\lambda^{(p)}_{ui}
-
y_{ui}
\log
\lambda^{(p)}_{ui}.
\]
The conceptual training objective is the masked Poisson negative log-likelihood over all legal training cells:
\[
\mathcal L_p^{\mathrm{full}}
=
\frac{
\sum_{u,i}
w_{ui}
\ell^{(p)}_{ui}
}{
\sum_{u,i}w_{ui}
}
+
\mathcal R_p.
\]
Regularization is
\[
\mathcal R_p
=
\lambda_t
\sum_i\|\mathbf t_i\|_2^2
+
\lambda_c
\sum_i\|\mathbf c_i\|_2^2
+
\lambda_b
\sum_i b_i^2.
\]
---
## 24.5 Sparse stochastic approximation
Computing every zero cell at every update is unnecessary.
All positive training interactions are retained.
Legal zeros may be subsampled.
For user \(u\), let
\[
Z_u
=
\{
j:
(u,j)\notin R_{\mathrm{all}}
\}.
\]
If \(m_u\) zero items are sampled uniformly with replacement,
\[
j_1,\ldots,j_{m_u}
\sim
\mathrm{Uniform}(Z_u),
\]
the zero-loss sum may be estimated as
\[
\widehat L^{(0)}_u
=
\frac{|Z_u|}{m_u}
\sum_{s=1}^{m_u}
\ell^{(p)}_{u j_s}.
\]
This preserves the intended full-zero objective in expectation.
If a nonuniform zero proposal \(q_u(j)\) is used, inverse-probability weighting must be applied.
The primary implementation must not silently treat an unweighted sampled-zero objective as if it were the full Poisson likelihood.
---
## 24.6 Zero weighting
The primary p-emb model uses
\[
w_0=1.
\]
A downweighted-zero p-emb experiment would be a separate named ablation such as `p-emb-dw`.
It must not silently replace the primary p-emb definition.
---
## 24.7 Ranking score
\[
s_{ui}
=
\eta^{(p)}_{ui}.
\]
Because exponential is monotonic, ranking by log-rate is equivalent to ranking by the Poisson rate.
---
# 25. Additive Poisson EF-EMB — ap-emb
## 25.1 Purpose
ap-emb is not p-emb with a different label.
It uses a genuinely additive Poisson mean and nonnegative embeddings.
This is the required mathematical distinction between `p-emb` and `ap-emb`.
---
## 25.2 Nonnegative target embeddings
Introduce unconstrained log-parameters
\[
\widetilde{\mathbf t}_i
\in
\mathbb R^K.
\]
Define the actual target embedding by
\[
\mathbf t_i^{+}
=
\exp
\left(
\widetilde{\mathbf t}_i
\right).
\]
Therefore,
\[
\mathbf t_i^{+}>0
\]
elementwise.
---
## 25.3 Nonnegative context embeddings
Introduce
\[
\widetilde{\mathbf c}_i
\in
\mathbb R^K.
\]
Define
\[
\mathbf c_i^{+}
=
\exp
\left(
\widetilde{\mathbf c}_i
\right).
\]
Therefore,
\[
\mathbf c_i^{+}>0
\]
elementwise.
The selected context normalization is then applied to these nonnegative context vectors.
---
## 25.4 Positive item baseline
The item baseline rate is
\[
\delta_i
=
\exp(b_i).
\]
Therefore,
\[
\delta_i>0.
\]
---
## 25.5 Additive rate
Define
\[
\lambda^{(ap)}_{ui}
=
\delta_i
+
(\mathbf t_i^{+})^\top
\mathbf h^{+}_{ui}.
\]
Hence
\[
\lambda^{(ap)}_{ui}>0.
\]
Its Poisson natural parameter is
\[
\eta^{(ap)}_{ui}
=
\log
\lambda^{(ap)}_{ui}.
\]
This is fundamentally different from p-emb.
For p-emb,
\[
\lambda^{(p)}_{ui}
=
\exp
\left(
b_i
+
\mathbf t_i^\top\mathbf h_{ui}
\right).
\]
For ap-emb,
\[
\lambda^{(ap)}_{ui}
=
\exp(b_i)
+
(\mathbf t_i^{+})^\top
\mathbf h^{+}_{ui}.
\]
---
## 25.6 Interpretation
p-emb allows target-context inner products to be positive or negative.
Therefore a context item may increase or decrease the rate multiplicatively.
ap-emb uses nonnegative embeddings.
Its context contribution to the mean is additive and nonnegative.
Consequently, ap-emb cannot express negative target-context associations through its latent interaction term.
---
## 25.7 Objective
For
\[
y_{ui}\in\{0,1\},
\]
the Poisson negative log-likelihood is
\[
\ell^{(ap)}_{ui}
=
\lambda^{(ap)}_{ui}
-
y_{ui}
\log
\lambda^{(ap)}_{ui}.
\]
As with p-emb, the conceptual objective includes all legal masked training cells.
Legal zeros may be subsampled only with the appropriate rescaling/inverse-probability correction.
---
## 25.8 Log-space regularization
Because the positive embeddings are exponential transforms of unconstrained log-parameters, regularization is applied in log space:
\[
\mathcal R_{ap}
=
\lambda_t
\sum_i
\|
\widetilde{\mathbf t}_i
\|_2^2
+
\lambda_c
\sum_i
\|
\widetilde{\mathbf c}_i
\|_2^2
+
\lambda_b
\sum_i b_i^2.
\]
---
## 25.9 Ranking score
The ranking score is
\[
s_{ui}
=
\log
\lambda^{(ap)}_{ui}.
\]
Because logarithm is strictly increasing, this ranks identically to the additive Poisson rate.
---
## 25.10 Mandatory p-emb/ap-emb test
Given identical data and ordinary nondegenerate parameters, the implementation must demonstrate that the two variants compute different rate functions.
A Boolean configuration field that does not modify the mathematics is a failure.
---
# 26. Gaussian EF-EMB — g-emb
## 26.1 Purpose
g-emb uses explicit MovieLens star ratings as a real-valued Gaussian prediction target.
Its user representation is generated from the user's binary training-history context.
---
## 26.2 Parameters
g-emb uses:
- target embeddings \(\mathbf t_i\);
- context embeddings \(\mathbf c_i\);
- item bias \(b_i\);
- global training mean \(\mu\).
The embeddings are unconstrained real-valued vectors.
---
## 26.3 Prediction
For training target \((u,i)\),
\[
\hat r^{(g)}_{ui}
=
\mu
+
b_i
+
\mathbf t_i^\top
\mathbf h_{ui}.
\]
The training context is
\[
H_u\setminus\{i\}.
\]
The target item never appears inside its own context.
---
## 26.4 Gaussian likelihood
\[
r_{ui}
\sim
\mathcal N
\left(
\hat r^{(g)}_{ui},
\sigma^2
\right).
\]
With fixed variance, the training objective is
\[
\mathcal L_g
=
\frac1{|R_{\mathrm{train}}|}
\sum_{(u,i,r_{ui})\in R_{\mathrm{train}}}
\left(
r_{ui}
-
\hat r^{(g)}_{ui}
\right)^2
+
\mathcal R_g.
\]
Regularization is
\[
\mathcal R_g
=
\lambda_t
\sum_i\|\mathbf t_i\|_2^2
+
\lambda_c
\sum_i\|\mathbf c_i\|_2^2
+
\lambda_b
\sum_i b_i^2.
\]
---
## 26.5 Global mean
\[
\mu
=
\frac1{|R_{\mathrm{train}}|}
\sum_{(u,i,r_{ui})\in R_{\mathrm{train}}}
r_{ui}.
\]
Validation/test ratings do not contribute.
---
## 26.6 Negative sampling
g-emb does not use negative sampling.
It trains only on canonical observed training ratings.
---
## 26.7 Ranking score
For an evaluation candidate,
\[
s_{ui}
=
\hat r^{(g)}_{ui}.
\]
The evaluation context is the user's training history.
---
## 26.8 Native diagnostics
g-emb reports:
\[
\mathrm{RMSE}
\]
and
\[
\mathrm{MAE}.
\]
---
# 27. Nonnegative Gaussian EF-EMB — ng-emb
## 27.1 Purpose
ng-emb is the nonnegative counterpart of g-emb.
Its target and context embeddings are produced from unconstrained raw parameters by a frozen softplus transformation.
The transformed embeddings used during fitting must be the same transformed embeddings used during inference.
Raw unconstrained parameters must never be substituted directly during scoring.
---
## 27.2 Raw parameters and positivity transform
Introduce unconstrained raw parameters
\[

\widetilde{\mathbf t}_i
\in
\mathbb R^K

\]
and
\[

\widetilde{\mathbf c}_i
\in
\mathbb R^K.

\]
Define the frozen elementwise positivity transformation
\[

\phi_{+}(x)

=

\operatorname{softplus}(x)

=

\log(1+\exp x).

\]
The actual target embedding is
\[

\mathbf t_i^{+}

=

\phi_{+}
\left(
\widetilde{\mathbf t}_i
\right),

\]
and the actual context embedding is
\[

\mathbf c_i^{+}

=

\phi_{+}
\left(
\widetilde{\mathbf c}_i
\right).

\]
For finite raw parameters, both transformed embedding tables are strictly positive elementwise.
The implementation must use a numerically stable softplus primitive rather than computing \(\log(1+\exp x)\) naively.
---
## 27.3 Nonnegative context
The context vector is built from transformed context embeddings:
\[

\mathbf h^{+}_{ui}

=

\mathrm{Normalize}
\left(
\{
\mathbf c_j^{+}
:
 j\in C_{ui}
\}
\right).

\]
The positivity transformation occurs before aggregation and before the configured raw/square-root/mean context normalization.
For a positive training target \(i\), the target item remains excluded from its own context exactly as specified in the shared EF-EMB framework.
---
## 27.4 Prediction
\[

\hat r^{(ng)}_{ui}

=

\mu
+
 b_i
+
(\mathbf t_i^{+})^\top
\mathbf h^{+}_{ui}.

\]
The global mean \(\mu\) is calculated from canonical training ratings only.
The item bias \(b_i\) and global mean remain unconstrained real-valued quantities.
---
## 27.5 Training objective
\[

\mathcal L_{ng}

=

\frac1{|R_{\mathrm{train}}|}
\sum_{(u,i,r_{ui})\in R_{\mathrm{train}}}
\left(
 r_{ui}
-
\hat r^{(ng)}_{ui}
\right)^2
+
\mathcal R_{ng}.

\]
Regularization is applied to the unconstrained raw embedding parameters:
\[

\mathcal R_{ng}

=

\lambda_t
\sum_i
\|\widetilde{\mathbf t}_i\|_2^2
+
\lambda_c
\sum_i
\|\widetilde{\mathbf c}_i\|_2^2
+
\lambda_b
\sum_i b_i^2.

\]
---
## 27.6 Training/inference consistency
The following identity is mandatory in every code path.
During training:
\[

\mathbf t_i^{+}
=
\operatorname{softplus}(\widetilde{\mathbf t}_i),
\qquad
\mathbf c_i^{+}
=
\operatorname{softplus}(\widetilde{\mathbf c}_i).

\]
During validation/test scoring:
\[

\mathbf t_i^{+}
=
\operatorname{softplus}(\widetilde{\mathbf t}_i),
\qquad
\mathbf c_i^{+}
=
\operatorname{softplus}(\widetilde{\mathbf c}_i).

\]
The implementation must not train with softplus-transformed embeddings and later score with the raw parameters.
This directly resolves the training/evaluation mismatch present in the original project.
---
## 27.7 Alternative positivity transformations
The exponential transformation is not the frozen primary ng-emb definition.
If an exponential or another positivity transform is studied later, it must be a separately named numerical/parameterization ablation and must use the selected transform consistently during both training and inference.
---
## 27.8 Ranking score
\[

s_{ui}

=

\hat r^{(ng)}_{ui}.

\]
Native diagnostics are RMSE and MAE.
---
# 28. EF-EMB optimization
## 28.1 Optimizer
The primary EF-EMB implementations use PyTorch Adagrad.
This follows the optimization family used in the original EF-EMB work while remaining compatible with the shared experiment pipeline.
The following are explicit configuration values:
- learning rate;
- latent dimension \(K\);
- epochs;
- batch size;
- regularization strength;
- negative/zero sampling counts where applicable;
- context normalization;
- gradient clipping threshold.
---
## 28.2 Minibatching
Training must be vectorized/minibatched.
Per-rating Python loops over complete item histories are prohibited in the publication implementation when a vectorized equivalent is available.
---
## 28.3 Early stopping
Validation NDCG@10 is the primary model-selection metric.
Gaussian native validation diagnostics may also be recorded but do not replace the common primary ranking-selection rule unless a separately documented experiment explicitly changes the selection criterion.
---
## 28.4 Numerical failure
A run fails if any of the following occurs:
- nonfinite loss;
- nonfinite target embeddings;
- nonfinite context embeddings;
- nonfinite bias;
- nonfinite Poisson rate;
- nonfinite prediction;
- nonfinite gradient after stabilization.
Failed runs cannot enter the master comparison.
---
# 29. Gaussian EF-EMB stability protocol
The original project identified heavy-tailed context size as a possible source of Gaussian instability.
The corrected pipeline tests this empirically rather than assuming the conclusion.
---
## 29.1 Primary stable configuration
The primary corrected g-emb and ng-emb configuration uses
```text
context_normalization: mean
```
together with:
- minibatch training;
- finite gradient clipping;
- explicit L2/raw-parameter regularization;
- deterministic initialization;
- target exclusion.
---
## 29.2 Controlled normalization ablation
The stability experiment varies only the context normalization rule:
```text
raw
sqrt
mean
```
while keeping the remaining selected model configuration fixed as much as scientifically practical.
This allows the effect of context magnitude to be studied directly.
---
## 29.3 Quantities recorded
For each Gaussian EF-EMB stability run, record at minimum:
- training loss;
- validation NDCG@10;
- RMSE;
- MAE;
- maximum absolute prediction;
- context-length distribution;
- context-vector norm distribution;
- maximum gradient norm before clipping if available;
- number of nonfinite events;
- run status.
---
## 29.4 Interpretation rule
The paper may claim that heavy-tailed unnormalized context sums cause instability only if the corrected empirical ablation supports that conclusion.
If the corrected experiments do not support the original explanation, the manuscript must report the corrected finding instead.
---
# 30. EF-EMB context diagnostics
Context diagnostics must be generated from the actual frozen training data used by the model.
For each user \(u\), record
\[
|H_u|.
\]
For each positive target \(i\),
\[
|C^+_{ui}|
=
|H_u|-1.
\]
The context-length figures used in the publication must be derived from these corrected contexts rather than from old notebook artifacts.
---
# 31. EF-EMB ranking semantics
The common scoring interface is:
## b-emb
\[
s_{ui}
=
b_i+\mathbf t_i^\top\mathbf h_{ui}.
\]
## p-emb
\[
s_{ui}
=
b_i+\mathbf t_i^\top\mathbf h_{ui}.
\]
This is the Poisson log-rate.
## ap-emb
\[
s_{ui}
=
\log
\left[
\exp(b_i)
+
(\mathbf t_i^{+})^\top\mathbf h_{ui}^{+}
\right].
\]
## g-emb
\[
s_{ui}
=
\mu+b_i+\mathbf t_i^\top\mathbf h_{ui}.
\]
## ng-emb
\[
s_{ui}
=
\mu+b_i+
(\mathbf t_i^{+})^\top\mathbf h_{ui}^{+}.
\]
Every score must be finite.
---
# 32. EF-EMB family-native diagnostics
## b-emb
May report Bernoulli BCE/log-likelihood on a clearly documented diagnostic evaluation set.
## p-emb
May report Poisson negative log-likelihood on the binary occurrence target.
## ap-emb
May report Poisson negative log-likelihood on the same binary occurrence target.
## g-emb
Reports rating RMSE and MAE.
## ng-emb
Reports rating RMSE and MAE.
These diagnostics are supplementary.
Cross-family model ranking is based on the common recommendation metrics.
---
# 33. Mandatory EF-EMB sanity tests
A valid EF-EMB implementation must pass all of the following.
## Shared tests
1. target item never appears in its positive training context;
2. validation/test interactions never appear in train-only contexts;
3. output length equals candidate length;
4. all scores are finite;
5. same seed/config reproduces the intended deterministic result;
6. sorted input and shuffled input produce the same canonical context memberships;
7. empty context behavior is defined;
8. configured normalization is actually applied;
9. target and context embedding tables are separate;
10. no stale pair-generation artifact is used.
## b-emb tests
11. negatives are genuinely unobserved everywhere;
12. logits-based BCE is finite;
13. negative-sampling stream is deterministic.
## p-emb tests
14. Poisson rates are strictly positive and finite;
15. sampled-zero correction is applied when approximating the full zero objective;
16. held-out positives are not treated as zero cells.
## ap-emb tests
17. target embeddings are positive;
18. context embeddings are positive;
19. additive rate is positive;
20. p-emb and ap-emb do not compute identical rate functions;
21. log-space transformation is used at inference.
## g-emb tests
22. explicit training ratings are the prediction target;
23. the target item is excluded from its context;
24. RMSE/MAE predictions are finite.
## ng-emb tests
25. nonnegative transformation is used during training;
26. the same nonnegative transformation is used during inference;
27. raw internal parameters are never used directly for scoring;
28. predictions and RMSE/MAE are finite.
---
# 34. Cross-model fairness
Fair comparison does not mean forcing every model to use an identical likelihood or identical target.
Fairness means that models share:
- the same frozen dataset;
- the same canonical split;
- the same information boundaries;
- the same common ranking candidate protocol;
- the same ranking metrics;
- validation-only hyperparameter selection;
- comparable tuning budgets;
- deterministic experimental controls;
- the same final-evaluation policy.
Differences in model-native objectives are part of the scientific comparison.
---
# 35. Frozen model specification status
The mathematical specification is now frozen for all primary models:
- Random;
- Popularity;
- PCA/SVD;
- BernoulliMF;
- PoissonMF;
- GaussianMF;
- LogisticPCA;
- HPF;
- b-emb;
- p-emb;
- ap-emb;
- g-emb;
- ng-emb.
The following EF-EMB issues from the original project are explicitly resolved:
- target self-context leakage is prohibited;
- training and evaluation contexts are defined separately;
- validation/test context leakage is prohibited;
- context normalization is explicit;
- mean normalization is the corrected primary setting;
- raw/square-root normalization remain controlled ablations;
- old stochastic pair-materialization is removed from the primary implementation;
- any future context cap must be explicit and deterministic;
- p-emb and ap-emb now have genuinely different mathematics;
- p-emb uses a multiplicative Poisson mean;
- ap-emb uses an additive Poisson mean;
- ap-emb uses positive embeddings;
- g-emb uses unconstrained embeddings;
- ng-emb uses softplus-transformed nonnegative embeddings;
- ng-emb uses the same softplus transformation during training and inference;
- Gaussian stability claims require corrected empirical evidence;
- all EF-EMB models expose the same common score interface.
- PoissonMF represents the masked full Poisson likelihood and uses unbiased/rescaled zero subsampling when the zero term is approximated.
No primary model implementation should deviate from this file without creating a new documented specification version.
---
# 36. Corrective issue registry
This section records specification corrections discovered after the original Step 26 freeze but before implementation of the affected models.
## B91 — PoissonMF zero-sampling objective
The earlier draft expressed PoissonMF as a positive term plus an arbitrarily weighted sampled-negative term.
That formulation did not faithfully represent the intended masked full Poisson likelihood.
Revision v1.1 corrects PoissonMF so that:
- the conceptual objective contains every canonical training positive and every genuine unobserved zero in the training catalog;
- validation/test positives are masked, not treated as zeros;
- zero subsampling is only a computational approximation;
- uniform zero sampling uses \(|Z_u|/m_u\) rescaling;
- nonuniform zero sampling requires inverse-probability weighting;
- an intentionally downweighted-zero variant must be separately named.
No already-completed model implementation is affected by B91 because PoissonMF had not yet been implemented when this correction was frozen.
## B92 — ng-emb positivity-transform consistency
The previous model-spec draft froze an exponential positivity transform for ng-emb even though the locked correction bible required the original intended softplus parameterization to be used consistently during training and inference.
Revision v1.1 restores softplus as the primary ng-emb positivity transform and requires the same transformed embeddings in every training and scoring code path.
No already-completed model implementation is affected by B92 because ng-emb had not yet been implemented when this correction was frozen.

## B93 — LogisticPCA literature alignment

The v1.1 specification described LogisticPCA using the correct low-rank
Bernoulli natural-parameter matrix, but additionally imposed explicit
orthogonality and centering constraints on the user-score matrix together
with QR retraction.

Those constraints are not part of the Schein, Saul, and Ungar (2003)
matrix-factorization formulation selected as the Paper 2 LogisticPCA
baseline.

Revision v1.2 therefore freezes LogisticPCA as:

\[
\Theta
=
\mathbf 1\boldsymbol\mu^\top
+
AB^\top,
\]

with Bernoulli likelihood over the masked legal binary matrix and without
an additional orthogonality or centering constraint on \(A\).

The factorization remains non-identifiable up to latent-coordinate
transformations, but this does not alter \(\Theta\) or ranking predictions.

Any numerical reparameterization must preserve \(AB^\top\).

The primary LogisticPCA objective does not add an asymmetric L2 penalty
to only one side of the factorization. The selected primary baseline
optimizes the masked Bernoulli likelihood directly.

The masking of validation/test positives is a Paper 2 leakage-prevention
adaptation to the complete-binary-matrix formulation of Schein, Saul,
and Ungar (2003); it is documented explicitly rather than presented as
part of the original method.

No already-completed model implementation is affected by B93 because
LogisticPCA had not yet been implemented when this correction was frozen.

Future corrections must receive new B-numbers and a new documented model-spec revision before affected implementation work proceeds.
