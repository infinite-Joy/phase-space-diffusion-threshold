# Gap 1 experiment plan: Is the threshold coordinate-invariant?

**Priority:** Critical  
**Status:** Model-specification gate complete; numerical experiment gates pending; no sampling results claimed
**Basis:** The isotropic marginal model and three reverse samplers in `README.md`.

## 1. Objective and interpretation

Determine which parts of the threshold statement survive an invertible linear
change of coordinates, and which change when the noise geometry is changed.
The working hypothesis is that feasibility is intrinsic to the specified
dynamics and marginal covariance schedule, whereas the numerical expression
`lambda_max(Sym(B))` assumes a particular Euclidean metric and isotropic noise.
Changing coordinates alone must not change feasibility.

Separate two interventions throughout:

| Arm | Drift | Conditional marginal noise covariance | Meaning |
|---|---|---|---|
| Baseline | `B` | `sigma(t)^2 I` | Repository model |
| A: transported model | `A B A^{-1}` | `sigma(t)^2 A A^T` | Same model in new coordinates |
| B: re-isotropized model | `A B A^{-1}` | `sigma(t)^2 I` | Different model, with new noise geometry |

Here, conditional covariance means covariance of the additive forward
perturbation given the clean initial state, not the total data covariance.
Transform the initial data in **both** arms. Never use an isotropic threshold
calculated for Arm B to classify Arm A.

### 1.1 Recorded model specification

This subsection records the model-specification gate in Section 9.8.
These are synthetic modeling conventions, not experimentally identified
physical units or new numerical results. Subsequent runners and output
metadata must implement this specification explicitly; recording it here
does not establish that an implementation satisfies it.

#### Units, scales, and coordinates

Use dimensionless variables throughout. Define reference length `L_ref`,
time `t_ref`, and mass `m_ref`; choose their numerical values as one in
arbitrary reference units, without assigning meters, seconds, or kilograms.
The corresponding velocity and momentum scales are
`V_ref = L_ref / t_ref` and `p_ref = m_ref L_ref / t_ref`.
Thus `x = x_physical / L_ref`, `t = t_physical / t_ref`,
`v = v_physical / V_ref`, and `p = p_physical / p_ref`.
Take the dimensionless particle mass to be one, so `p = v`.
The same reference scales remain fixed across all transforms and arms.
The coordinate map `A` acts on these already nondimensional variables;
it does not redefine the reference scales.

| Quantity | Recorded convention |
|---|---|
| Primary state | Column vector `z = (x1, v1, x2, v2)^T`, dimension four |
| Matrix-check pilot | Column vector `z = (x1, v1)^T`, dimension two |
| Canonical interpretation | Unit mass, hence `vi = pi` in dimensionless coordinates |
| Canonical matrix | `J = diag(J2, J2)` in four dimensions, `J = J2` in two, with `J2 = [[0, 1], [-1, 0]]` |
| Baseline drift | `B = diag(B2, B2)`, with `B2 = [[0, 1], [-omega^2, 0]]`; use `B2` alone in two dimensions |
| Primary frequency | `omega = 2`, measured per reference time; the baseline threshold is `1.5` |
| Scalar noise amplitude | `sigma(t)` is dimensionless; isotropy refers to the declared dimensionless coordinates, not equal physical position and momentum units |
| Primary schedule | `sigma(t) = 0.01 exp(0.9 t)`, ending at `sigmaT = 40` |
| Primary horizon | `T = log(40 / 0.01) / 0.9`; use this same horizon in every primary arm and transform |
| Separate rate sweeps | Replace `r = 0.9` by the declared positive rate and set `T = log(40 / 0.01) / r`; retain the same rate and horizon within each coordinate comparison |

For arrays storing samples in rows, the column-vector map `y = A z`
is implemented as `Y = Z @ A.T`. Record the actual matrix `A` in every
transformed run rather than relying only on its family name.

#### Covariance convention and initial smoothing

Let `X` denote the latent clean state, distinct from the noisy process state
`Z_0`. For a fixed symmetric positive-definite shape `Q`, the prescribed
one-time marginal construction is

$$
Z_t=M(t)X+\sigma(t)L_Q\epsilon,\qquad
M(t)=e^{Bt},\quad L_QL_Q^\top=Q,\quad
\epsilon\sim N(0,I),\quad \epsilon\ \text{independent of }X.
$$

This construction specifies individual marginals; independent perturbations
at different times do not themselves define the intended SDE transitions.
The covariance convention and endpoint laws are:

| Object | Definition |
|---|---|
| Conditional perturbation covariance | `C(t) = Cov(Z_t given X) = sigma(t)^2 Q` |
| Total state covariance | `Cov(Z_t) = M(t) Cov(X) M(t)^T + sigma(t)^2 Q` |
| Initial process law / reverse target | `Z_0 = X + 0.01 L_Q epsilon`; do not substitute the clean law |
| Initial conditional covariance | `C(0) = 0.0001 Q` |
| Primary reverse initialization | Exact terminal marginal: draw `X` from the declared clean law and independent `epsilon`, then form `M(T) X + 40 L_Q epsilon` |
| Approximate-prior control | The moment-matched Gaussian terminal prior is a separately labeled optional experiment, not the primary initialization |
| Clean reference laws | A declared nondegenerate Gaussian for the pilot; one fixed fitted 24-component mixture for the main study, with its parameters recorded and reused |

No Gaussian pilot mean/covariance or mixture fit is selected by this gate.
Record those concrete parameters before the corresponding sampling gate.
For either clean law, target samples must include the specified initial
smoothing, independently of sampler noise.

#### Exact definitions of the experimental arms

Write `M(t) = exp(B t)` for the baseline propagator. In the table below,
`epsilon` is standard Gaussian and independent of `X`. Each row defines
both the dynamics and the noise geometry; changing only the drift is not
an implementation of Arm A.

| Arm | Clean law | Drift and propagator | Covariance shape | Time-zero target | Exact terminal draw |
|---|---|---|---|---|---|
| Baseline | `X` | `B`, `M(t)` | `I` | `X + 0.01 epsilon` | `M(T) X + 40 epsilon` |
| A: transported | `A X` | `B_Y = A B A^{-1}`, `M_Y(t) = A M(t) A^{-1}` | `A A^T` | `A X + 0.01 A epsilon` | `A M(T) X + 40 A epsilon` |
| B: re-isotropized | `A X` | Same `B_Y` and `M_Y(t)` as Arm A | `I` | `A X + 0.01 epsilon` | `A M(T) X + 40 epsilon` |

Arm A is precisely the pushforward of the baseline law. Arm B is a
different model: after pullback, its conditional covariance shape is
`A^{-1} A^{-T}` and its time-zero target is
`X + 0.01 A^{-1} epsilon`. Therefore, Arm B quality metrics must use
that pulled-back target, not automatically the baseline smoothed target.
When comparing fully transported sampler paths, also transport the chosen
diffusion factors and reuse the same Brownian increments as specified in
Stage 2; marginal arm definitions alone do not establish pathwise coupling.

#### Feasibility scope and gate outcome

Here "feasible" means that the specified one-time marginals can be realized
by a forward linear SDE

$$
dZ_t=BZ_t\,dt+L(t)\,dW_t,\qquad
L(t)L(t)^\top=G(t)=\dot C(t)-BC(t)-C(t)B^\top,
$$

with real, state-independent diffusion and initial law equal to the smoothed
`Z_0` law above. For the fixed-shape exponential schedule,
`G(t) = 2 sigma(t)^2 (r Q - (B Q + Q B^T)/2)`, so feasibility is
`G(t)` positive semidefinite, equivalently `r >= kappa_Q`.
The equality case permits rank-deficient diffusion and is not required
to be positive definite.

Below threshold, the scheduled one-time marginals still exist, but this
specified forward SDE does not. This is not a claim that all stochastic
dynamics are impossible, that a corrected reverse sampler cannot exist,
or that a preferred physical metric has been identified.

**Gate record: DONE (2026-10-17).** The units, ordering, covariance convention,
smoothing, arms, and feasibility scope are specified above as model choices.
`python verification/model_specification.py` passes all six categories against
`coordinate_invariance_model_spec.json`.
Implementation conformance, deterministic identities, and empirical sampling
accuracy remain unassessed until their respective checks are run. Each future
run must record its dimension, actual matrices, schedule, clean-law parameters,
target and terminal-prior policies, and any explicit departure from this
specification.

#### Executable specification check

The machine-readable primary specification is
`coordinate_invariance_model_spec.json`. Check it with:

```bash
python verification/model_specification.py
```

The checker reports each of the six specification categories and exits
nonzero for missing fields, changed conventions, or inconsistent recorded
matrices. It independently constructs the expected drift, canonical matrix,
transported noise geometry, initial conditional covariance, and pulled-back
noise geometry for a representative four-dimensional squeeze
`A = diag(2, 1/2, 2, 1/2)`. This example checks the arm definitions; it does
not replace the transformation grid or the deterministic-implementation gate.
An alternative file can be checked with
`python verification/model_specification.py path/to/spec.json`; it must
conform to the same primary specification, not an arbitrary rate-sweep model.

Run the checker's regression tests, including deliberately incorrect records,
with `python -m unittest verification.test_model_specification`.
Passing certifies only this structured record against Section 1.1's contract:
the checker does not parse the prose, verify physical units against data,
or inspect a future runner. Concrete clean-law parameters and runner
conformance remain separate prerequisites before sampling.

## 2. Analytical predictions to establish first

For a fixed positive-definite covariance shape $Q$, write

$$
C(t)=\sigma(t)^2Q,\qquad
G(t)=\dot C-BC-CB^\top
     =2\sigma(t)^2(rQ-H_Q),\qquad
H_Q=\tfrac12(BQ+QB^\top).
$$

The appropriate threshold is the largest generalized eigenvalue:

$$
\kappa_Q=\lambda_{\max}(H_Q,Q)
=\lambda_{\max}\!\left(Q^{-1/2}H_QQ^{-1/2}\right),
\qquad G(t)\succeq0\ \Longleftrightarrow\ r\geq\kappa_Q.
$$

Use a symmetric generalized eigensolver or Cholesky whitening, rather than
forming an explicit matrix inverse in the implementation.
For $Q=I$, this recovers the README threshold.

Under $Y=AZ$, transport every relevant object:

$$
B_Y=ABA^{-1},\quad Q_Y=AQA^\top,\quad
G_Y=AGA^\top,\quad M_Y(t)=AM(t)A^{-1}.
$$

Congruence preserves the numbers of positive, zero, and negative eigenvalues
of $G$, not their numerical values. Consequently,
$\kappa_{Q_Y}(B_Y)=\kappa_Q(B)$.
The associated metric is $P=Q^{-1}$, transforming as
$P_Y=A^{-\top}PA^{-1}$. This makes the dynamics--noise--metric
interpretation explicit; it does not identify a unique physically preferred
metric without additional modeling assumptions.

### Decisive Hamiltonian example

For one coordinate pair and the symplectic squeeze
$A_a=\operatorname{diag}(a,a^{-1})$,

$$
B=\begin{pmatrix}0&1\\-\omega^2&0\end{pmatrix},\qquad
B_Y=\begin{pmatrix}0&a^2\\-\omega^2/a^2&0\end{pmatrix},
\quad
\kappa_I(B_Y)=\tfrac12|a^2-\omega^2/a^2|.
$$

At $\omega=2$, the baseline threshold is $1.5$.
At $a=\sqrt{2}$, the transformed drift is skew-symmetric and its
**re-isotropized** threshold is zero. Nevertheless, the transported model
has $Q_Y=\operatorname{diag}(2,1/2)$ and threshold $1.5$.
Thus, at fixed $r=0.9$, Arm A remains infeasible for the specified
state-independent forward diffusion, while Arm B is feasible.
This is an analytical prediction, not yet an empirical finding.

As a separate physical reference, the energy metric
$P_E=\operatorname{diag}(\omega^2,1)$ has
$Q_E=P_E^{-1}$ and $BQ_E+Q_EB^\top=0$.
Its zero threshold concerns another noise geometry; do not present it as a
coordinate-only modification of the baseline isotropic model.
State all matrices in nondimensional coordinates and document the reference
position, momentum/velocity, and time scales. Symplectic tests use canonical
coordinates, with the unit-mass convention relating velocity to momentum.

## 3. Transformation controls

Use the repository ordering `(x1, v1, x2, v2)` and
$J=\operatorname{diag}(J_2,J_2)$,
$J_2=\left(\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\right)$.
Check $A^\top J A=J$ before labeling a transform symplectic.

| Family | Proposed values | Purpose |
|---|---|---|
| Identity | `A = I` | Exact baseline recovery |
| Orthogonal rotations | Pairwise angles `pi/6`, `pi/4`, `pi/2` | Isotropy-preserving control; pairwise rotations are also symplectic |
| Uniform scaling | `A = c I`, `c = 0.5, 2` | Nonorthogonal control with unchanged drift threshold |
| Symplectic squeezes | Repeated `diag(a, 1/a)`, `a = 0.5, 1, sqrt(2), 2` | Main test, including zero re-isotropized threshold |
| Symplectic shears | Repeated `[[1, h], [0, 1]]`, `h = -1, 0.5, 1` | Non-diagonal anisotropic covariance control |
| Noncanonical relative scaling | Repeated `diag(c, 1)`, `c = 0.5, 2` | General invertible-coordinate control |

Start in 2-D for transparent matrix checks, then repeat in the README's
4-D phase space. Record determinant, condition number, symplectic residual,
and orthogonality residual. Keep primary transforms moderately conditioned;
reserve extreme squeezes for explicitly labeled numerical stress tests.
Uniform scaling with unchanged scalar `sigma` still changes Arm B's physical
noise amplitude, even though its threshold agrees with baseline.

## 4. Experimental stages

### Stage 1: Algebraic and Gaussian checks

1. Verify similarity of propagators, congruence of diffusion matrices,
   generalized-threshold invariance, and the squeeze formula above.
2. At rates below, at, and above each relevant threshold, check diffusion
   inertia with scale-aware eigenvalue tolerances. Treat boundary
   semidefiniteness separately from positive definiteness.
3. Use a nondegenerate single Gaussian with exact scores and exact marginal
   means/covariances. Integrate reverse moment equations as a low-cost
   reference before Monte Carlo sampling.
4. Verify density and score transport:
   $p_Y(y,t)=p_Z(A^{-1}y,t)/|\det A|$ and
   $s_Y(y,t)=A^{-\top}s_Z(A^{-1}y,t)$.
5. Confirm that Arm B becomes an anisotropic model after mapping back:
   its baseline-coordinate covariance shape is $A^{-1}A^{-\top}$.
   This gives a second, equivalent implementation of Arm B.

**Gate:** Do not proceed to large sampling runs until these identities hold.
Target relative residuals of `1e-10` in float64 for the primary, moderately
conditioned matrix cases; report absolute and scaled residuals near zero.
Use the assumption audit and independent validation routes in Section 9
to avoid checking identities only against another use of the same code.

### Stage 2: Exact-score sampling

Use the exact Gaussian first, then the repository's fitted 24-component
Gaussian mixture, keeping one mixture fit fixed across all transforms.
Transport mixture means and component covariances by `A` and `A S A^T`.
For Arm A, wrap the baseline exact score using the transport identity.
For Arm B, recompute component marginal covariances and exact scores with
the new isotropic covariance; reusing Arm A's score would be incorrect.

Run the probability-flow ODE, clipped Anderson sampler, and corrected SDE.
Use two complementary comparisons:

- **Coordinate equivalence:** Transport the complete baseline sampler,
  including its chosen diffusion, drift, terminal samples, and Brownian
  factors. Map outputs back and compare against baseline.
- **Local clipping intervention:** Recompute the Euclidean spectral positive
  part of `G_Y` in transformed coordinates, then run locally clipped and
  locally corrected samplers with matched noise.

These comparisons must not be conflated. In general,

$$
(AGA^\top)_+\ne A G_+ A^\top.
$$

Therefore, locally re-clipping an indefinite matrix defines a different
reverse sampler even within Arm A. The corrected sampler remains marginally
exact with any chosen diffusion $D\succeq0$, using forward-time drift
$Bz-\tfrac12(G+D)s$. The clipped sampler uses $Bz-Ds$.
Their marginal discrepancy need not be invariant under a new clipping rule.
Above threshold, local clipping leaves `G` unchanged.

For pathwise transport, if baseline noise has factor `L`, use `A L` with
the same standard Brownian increments, not an independently recomputed
symmetric square root. Equal random seeds alone do not ensure this coupling.
Pair local clipped/corrected runs using the same terminal draws and factors.

Initialize from the **exact terminal marginal** in the primary exact-score
experiment. The existing moment-matched Gaussian prior is an optional
separate control, since prior approximation must not masquerade as a
coordinate effect.
At `t = 0`, compare to the actual smoothed target
`Z_clean + sigma0 * epsilon`, transported appropriately, rather than silently
treating finite `sigma0 = 0.01` as zero.

### Stage 3: Trained-score extension, conditional on Stages 1--2

First wrap a single baseline network with
`s_Y(y, t) = A^{-T} s_Z(A^{-1} y, t)` for Arm A.
This isolates coordinate transport from retraining error.
Only then train separate models for Arm B, with appropriate noise targets
and covariance-aware normalization. Treat separately trained Arm A networks
as an optional optimization-conditioning study, not a test of the theorem.

As an optional likelihood-weighting check, use the quadratic score-error
term `e^T G e`, with `e_Y = A^{-T} e_Z`, to verify invariant sign under
transport. Do not reuse the isotropic epsilon-prediction weight unchanged
for anisotropic perturbations. This extension is not required to resolve
the core feasibility question.

## 5. Rates, replication, and numerical controls

- Primary comparison: `omega = 2`, fixed absolute `r = 0.9`,
  `sigma0 = 0.01`, `sigmaT = 40`, and
  `T = log(sigmaT / sigma0) / r`, identical across transforms and arms.
  Never alter time or rate merely to keep `r/kappa` fixed across coordinates.
- Boundary sweep: fixed absolute rates
  `0.6, 0.9, 1.35, 1.5, 1.65, 1.95`, supplemented by
  `0.9*kappa_I`, `kappa_I`, and `1.1*kappa_I` for nonzero transformed
  thresholds. Label these as separate rate settings, not coordinate pairs.
  For zero thresholds, use positive absolute rates; never divide by zero
  or construct `Process` with `r = 0`.
- Pilot: Gaussian checks plus the identity and decisive squeeze, `N = 2000`.
  Main exact-score runs: `N = 8000`, seeds `0..4`, and `K = 4000`
  uniform steps, matching existing exact-score scale.
- Check representative below/boundary/above cases at `K = 1000, 2000, 4000,
  8000`, aggregating fine Brownian increments for coupled coarse paths.
  Include the most ill-conditioned primary transform.
- Report paired seed-level differences and their spread. Five seeds give
  limited uncertainty resolution; do not use particle count as the number
  of independent experimental replicates.
- Run the full mixture grid only after the Gaussian pilot and timing
  estimate. Keep training optional until the exact-score conclusions settle.

## 6. Measurements and acceptance criteria

Compute primary sample metrics **after mapping samples back to baseline
coordinates**, using fixed projections, kernels, reference samples, and
bandwidths. Euclidean MMD, sliced Wasserstein distance, and covariance trace
in different raw coordinate systems are not directly comparable.

| Measurement | Expected result / acceptance criterion |
|---|---|
| Generalized threshold in Arm A | Equals baseline `1.5` within declared algebraic tolerance |
| Diffusion inertia in Arm A | Same below/at/above classification for every transform |
| Euclidean isotropic threshold in Arm B | Follows transformed-drift calculation; decisive squeeze gives zero |
| Fully transported sampler paths | Agree after pullback up to accumulated floating-point error under identical updates |
| Corrected and ODE Gaussian moments | Agree with exact target within measured integration error |
| Exact-score mixture quality | Corrected/ODE errors decrease toward target with refinement; quantify Monte Carlo uncertainty |
| Local clipped vs corrected | Coincide above threshold under matched updates; quantify below-threshold deviation without assuming a universal nonzero effect |
| Local clipping covariance | Explicitly measure `(A G A^T)_+ - A G_+ A^T`; do not require it to vanish |

Record mean error, full covariance error, pulled-back variance ratios,
MMD squared, and paired Gaussian-moment KL. Label the latter as a
moment-based proxy for non-Gaussian mixtures, not the true distributional KL.
True KL is invariant under an invertible coordinate transformation, but
finite estimators and untransported metric choices can obscure that fact.
Choose sampling tolerances from the Gaussian reference, convergence study,
and independent target-vs-target Monte Carlo controls before interpreting
the main mixture runs. Do not treat small finite-sample negative unbiased
MMD estimates as failures.

## 7. Repository implementation and deliverables

The files below are **proposed implementation work**, not existing commands:

1. Add `verification/coordinate_invariance.py` for matrix identities,
   symplectic checks, exact Gaussian moments, score transport, and clipping
   non-equivariance. Integrate checks with `verification/run_all.py` while
   preserving existing checks.
2. Add `coordinate_invariance.py` as the experiment runner. Reuse dynamics,
   mixture fitting, and metrics from `psdt.py` and `gmm_exact.py`.
   Introduce covariance shape and explicit noise-factor/coupling support
   only where needed, preserving isotropic defaults. Audit the hardcoded
   isotropic covariance in `Process`, `ExactScore.params`, training, and
   `prior_sample`; do not obtain Arm A merely by changing `B`.
3. Save new outputs under `results/coordinate_invariance/`, without
   overwriting baseline results. Record `A`, `B_Y`, `Q`, thresholds,
   diffusion inertia, clipping policy, seeds, time grid, terminal-prior
   policy, score source, numerical tolerances, environment, and code revision.
4. Add `coordinate_invariance_analyze.py` and figures showing thresholds
   against squeeze strength, pulled-back sampling discrepancies, and
   convergence. Separate transported-model and re-isotropized-model panels.
5. Write a results report and add a concise README interpretation section
   only after the experiment has run. Run existing verification plus the
   new focused checks before accepting implementation changes.

## 8. Decision rule and claim boundary

If the predicted distinction is confirmed, conclude:

> The state-independent diffusion feasibility criterion is invariant when
> drift and marginal covariance are transformed consistently. The scalar
> formula based on the Euclidean symmetric drift is specific to isotropic
> marginal noise in the chosen coordinates. Reimposing isotropy after a
> nonorthogonal change of coordinates changes the model, even for
> symplectic transformations.

Distinguish that statement from coordinate dependence of spectral clipping,
score-network conditioning, discretization, or raw Euclidean diagnostics.
If transported feasibility changes, debug covariance transport and numerical
conditioning before claiming a counterexample to the theorem.
The experiment concerns linear transformations and fixed covariance shapes;
it does not establish invariance under nonlinear coordinates or identify a
uniquely correct physical noise model.

## 9. Assumption audit and validation strategy

### 9.1 What "validating assumptions" means

Separate three kinds of evidence in the report:

- **Model choices:** Linear drift, Gaussian perturbations, fixed covariance
  shape, and a selected physical metric define this experiment. Synthetic
  data can verify that the implementation follows these choices, but cannot
  establish that they describe a real physical system.
- **Mathematical consequences:** Threshold invariance, diffusion congruence,
  and score transport admit proofs. Supply short derivations and independent
  numerical checks; simulation is not a substitute for proof.
- **Empirical robustness:** Sampling errors, clipping effects, training
  behavior, and numerical conditioning require measured comparisons with
  uncertainty and convergence controls.

Do not interpret passing numerical checks as proof of physical relevance,
or failing an out-of-scope stress test as a counterexample to the theorem.
The following is a menu of complementary validation routes, not a request
to run every expensive combination. The minimum evidence is in Section 9.8.

### 9.2 Assumption-by-assumption validation matrix

| Assumption or requirement | Independent validation routes | Failure interpretation |
|---|---|---|
| Drift is the intended constant linear field | Check `f(z) = B z` on basis vectors and random states; finite-difference its Jacobian; compare the oscillator's closed-form propagator with `expm` and numerical integration | Wrong drift, ordering, or propagator implementation |
| Coordinate map is constant and invertible | Solve forward/backward round trips; inspect singular values and condition number; check composition and inverse transformations | Singular or ill-conditioned map, transpose error, or mismatched conventions |
| Claimed canonical maps are symplectic | Check `A^T J A = J`, transported Poisson brackets, and Hamiltonian structure of `B_Y`; include cross-pair maps, not only repeated 2-D blocks | Incorrect canonical claim; determinant one alone is insufficient |
| Covariance shape is fixed and positive definite | Check symmetry and Cholesky factorization; estimate conditional residual covariance at multiple times after dividing by `sigma(t)^2`; use correlated, non-diagonal `Q` controls | Wrong covariance, accidental isotropy, or time-varying shape |
| Perturbations are additive, Gaussian, and independent of clean data | Inspect whitened residual means/covariances, projected quantiles and fourth moments; test residual dependence on clean states across held-out groups | Generator violates the chosen marginal law; finite diagnostics cannot prove independence or Gaussianity |
| Schedule is exponential with the declared rate | Compare analytic and finite-difference `dot(C)`; check the slope of `log(sigma)` and independently calculated endpoints and horizon | Rate, time-grid, or scale mismatch |
| Feasibility refers to the specified state-independent forward SDE | Check the covariance evolution, finite-interval transition covariance, and an explicit negative-direction witness below threshold | Invalid forward construction or an inappropriate claim about more general processes |
| All objects are transported consistently in Arm A | Compare direct and transported marginal means, covariances, log densities, scores, drifts, and diffusion factors | Mixed coordinate systems or an unintended change of model |
| Arm B really imposes new isotropic noise | Generate its residuals directly in transformed coordinates and independently via the pulled-back anisotropic model | Arm confusion or accidental reuse of baseline noise/score |
| Score and terminal distribution are exact in reference runs | Check analytic Gaussian scores, finite-difference mixture log densities, independently sampled terminal moments, and mixture component frequencies | Score, mixture, or initialization error rather than a threshold effect |
| Coupling compares the same random experiment | Store terminal draws and Brownian increments; verify transformed increments and summed coarse increments explicitly | Seeds were matched but paths were not |
| Diagnostics compare the same physical quantities | Compare pullback metrics against directly transported metric definitions; calibrate on target-vs-target samples | Coordinate-dependent measurement mistaken for changed sampling behavior |
| Discretization and finite samples do not drive the conclusion | Refine steps, sample counts, precision, and selected transforms independently; compare with exact Gaussian moment evolution | Numerical artifact or insufficient statistical resolution |
| Chosen scales have a physical interpretation | Document units, mass convention, canonical variables, reference scales, and plausible alternative noise covariances | Limits on interpretation, not an algebraic failure |

For Hamiltonian structure, transform the energy matrix as
$K_Y=A^{-\top}KA^{-1}$ and verify $B_Y=J K_Y$ for symplectic $A$,
starting from $B=J K$. An arbitrary invertible map need not preserve the
same numerical canonical matrix $J$.

### 9.3 Independent proofs and numerical oracles

**Threshold by three routes.** Derive the generalized eigenvalue formula
from the PSD condition, compute it using both a symmetric generalized
eigensolver and Cholesky whitening, and independently locate the sign change
of the smallest eigenvalue of `r Q - H_Q` by bracketing.
Do not clip eigenvalues before classifying feasibility.
Report residuals for the computed generalized eigenpairs.

For an eigenvector $v$ attaining the largest generalized eigenvalue,

$$
v^\top G(t)v
=2\sigma(t)^2(r-\kappa_Q)\,v^\top Qv.
$$

This is a direct negative-variance witness when $r<\kappa_Q$.
Transport the witness as $v_Y=A^{-\top}v$ and verify that the quadratic
form is unchanged. Preserve eigenvalue multiplicity at the boundary;
individual eigenvectors need not be unique.

**Feasibility by finite-time transitions.** For $t>s$, independently compute

$$
R_{t,s}=C(t)-e^{B(t-s)}C(s)e^{B^\top(t-s)}
       =\int_s^t e^{B(t-u)}G(u)e^{B^\top(t-u)}\,du.
$$

Compare the closed expression with numerical quadrature.
Above and at threshold, check that `R` is PSD and simulate Gaussian
transitions with mean `exp(B (t-s)) z_s` and covariance `R`.
Below threshold, show a negative quadratic form for sufficiently small
positive `t-s`; a long interval alone need not reveal local infeasibility.
The prescribed Gaussian conditional transition law follows under the linear,
state-independent diffusion model, not from one-time marginals alone.

Because `sigma0` is nonzero, the SDE initial law is the clean data convolved
with `N(0, C(0))`. Start forward simulations from this smoothed law.
Do not confuse it with the latent clean state in the marginal construction.
All scheduled one-time marginals still exist below threshold, even though
these forward transitions cannot realize them with the specified drift.

**Dynamics by a closed-form oracle.** For the oscillator, compare against

$$
M(t)=
\begin{pmatrix}
\cos(\omega t)&\sin(\omega t)/\omega\\
-\omega\sin(\omega t)&\cos(\omega t)
\end{pmatrix}.
$$

Include zero time, small times, several oscillation periods, and the full
diffusion horizon. For Gaussian reverse sampling, use an independently
written mean/covariance ODE, not moments estimated from the same sampler.

**Marginal evolution by a PDE oracle.** At held-out points and times,
check the continuity equation for the probability-flow ODE and the
reverse-time Fokker--Planck equation for the corrected SDE.
Use analytic Gaussian derivatives first and finite-difference mixture
densities second. Check time reversal signs explicitly.
The equation with an indefinite `G` is a formal density identity, not a
valid forward SDE generator.

### 9.4 Property-based and metamorphic tests

Use fixed, recorded random seeds to generate moderate-condition invertible
matrices, correlated SPD covariance shapes, and nonnormal drift matrices.
Include distinct-frequency oscillators and maps mixing the two coordinate
pairs so repeated identical blocks cannot conceal indexing mistakes.
Keep random-matrix checks cheap; sampling need only cover selected cases.

Check the following relations without relying on a fitted network:

- Applying `A`, then `A^{-1}`, recovers every transported object.
- Applying `A1`, then `A2`, agrees with the single map `A2 A1`.
- Whitening the transported covariance recovers the same threshold.
- Replacing `Q` by `c Q` for positive `c` leaves the threshold unchanged,
  though it changes the physical noise amplitude.
- Replacing `B` by `B + b I` shifts the threshold by `b`.
- A `Q`-preserving rotation, `A Q A^T = Q`, requires no new noise shape.
- Metric norms and quadratic score errors agree before and after transport;
  raw Euclidean norms are not required to agree.

Include `B = 0`, skew-symmetric drift with `Q = I`, stable scalar drift,
and a nonnormal shear. A negative threshold is legitimate for contracting
dynamics; do not clamp it to zero. Test zero and negative rates algebraically
without passing them to the existing positive-rate horizon constructor.
At rank-deficient diffusion boundaries use PSD-compatible eigendecomposition,
not Cholesky intended for strictly positive-definite diffusion.

### 9.5 Deliberate failure controls

Demonstrate that the diagnostics detect known mistakes, in isolated test
cases rather than production experiment code:

| Deliberate mistake | Diagnostic that must detect it |
|---|---|
| Transform drift but leave Arm A covariance isotropic | Covariance transport residual and decisive squeeze feasibility mismatch |
| Transform a score as `A s` instead of `A^{-T} s` | Finite-difference log-density gradient and Gaussian score checks |
| Recompute a noise square root but claim identical coupled paths | Increment-level transport and pathwise equivalence checks |
| Use local clipping while claiming a fully transported sampler | Clipping congruence residual for a known indefinite, nonorthogonal case |
| Omit the density Jacobian for a non-unit-determinant map | Absolute log-density transport check; score checks alone will not detect a constant |
| Use the clean law as the finite-`sigma0` target | Exact endpoint covariance check |
| Compare raw Euclidean covariance traces across squeezes | Disagreement with pulled-back variance diagnostics |
| Omit the covariance derivative for a changing schedule | Finite-difference covariance balance check |

Require a known counterexample, not just a random draw, for each
non-invariance control. Orthogonal maps, positive diffusion, or special
commuting matrices can accidentally make an incorrect implementation pass.

### 9.6 Sensitivity studies and scope boundaries

After the core tests pass, vary one assumption at a time:

- **Noise geometry:** Compare isotropic, energy-aligned, and correlated SPD
  shapes at matched declared amplitudes. Separately match determinant or
  trace in documented reference coordinates to distinguish total noise
  amount from orientation and anisotropy; neither matching convention is
  universally physically preferred.
- **Data and scores:** Compare a single Gaussian, separated mixtures,
  overlapping mixtures, and the fitted two-moons mixture. Add controlled
  score errors after the exact-score study; distinguish transported errors
  from independently generated errors and independently trained networks.
- **Initialization:** Compare exact terminal samples with the repository's
  Gaussian moment prior, recording each prior's own target discrepancy.
- **Schedule:** Test a smooth positive non-exponential `sigma(t)` using the
  instantaneous rate `dot(sigma)/sigma`. For time-varying SPD `Q(t)`, retain
  `sigma(t)^2 dot(Q)` in `G`; the fixed-shape threshold is no longer the
  complete criterion.
- **Coordinate class:** A time-dependent linear map requires the additional
  drift term `dot(A) A^{-1}`. A nonlinear map generally requires Itô
  corrections and state-dependent diffusion. These are separate follow-on
  studies, not cases governed by the current constant-`A` formula.
- **Physical specification:** If laboratory or mechanistic evidence is
  available, compare predicted residual covariance, local drift, and scale
  choices with it. Without such evidence, state explicitly that the study
  establishes mathematical consistency, not physical identification.

Gaussianity supports the exact density and score references, but the
covariance congruence and generalized-eigenvalue arguments do not depend
on the clean data being Gaussian. Do not overstate Gaussianity as necessary
for those algebraic conclusions.

### 9.7 Statistical and numerical evidence standards

Keep deterministic identity checks separate from stochastic quality metrics.
Use independent target samples for calibration, shared reference samples for
paired method comparisons, and separate RNG streams for data, initialization,
Brownian noise, and metric projections.
Report seed-level paired effects with intervals and raw values; any bootstrap
must resample independent experimental replicates, not treat coupled
particles as independent runs.

For equivalence claims, set practical margins from the Gaussian oracle and
pilot before the main runs, then show uncertainty lies within those margins.
Failure to reject a difference is not evidence of equivalence.
For exact pathwise identities, use scaled floating-point residuals rather
than a statistical significance test.
Predeclare primary comparisons; label the large transform/rate exploration
as descriptive unless multiplicity is addressed.

Check both time refinement and particle-count refinement; one does not
replace the other. Repeat selected algebraic checks at higher precision
only if conditioning or near-zero eigenvalues make float64 classification
ambiguous. Record such cases as numerically unresolved until resolved;
never hide them with larger tolerances or automatic PSD projection.
Use scale-aware boundary bands with a reported residual-based uncertainty
instead of pretending floating point determines an exact zero.

### 9.8 Minimum evidence and execution gates

| Gate | Status | Required evidence before proceeding |
|---|---|---|
| Model specification | [x] Done (2026-10-17); specification checker passes | Recorded units, coordinate order, covariance convention, initial smoothing, arm definitions, and feasibility scope |
| Deterministic implementation | [ ] Pending | Closed-form squeeze and propagator checks; generalized threshold via independent routes; score/density transport; inverse/composition tests; deliberate failures detected |
| Forward feasibility | [ ] Pending | PSD transition covariances above/at threshold and a local negative-direction witness below it |
| Gaussian sampling | [ ] Pending | Independent moment reference, exact terminal initialization, transported-path equivalence, and step refinement |
| Mixture sampling | [ ] Pending | Independently checked exact scores, paired seeds, target-vs-target calibration, and pulled-back diagnostics |
| Optional generalization | [ ] Not assessed; optional | Sensitivity to other noise shapes, priors, data, and trained scores reported separately from core conclusions |

The final report should include an assumption ledger with one row per
requirement in Section 9.2: classification as model choice, identity, or
empirical requirement; validation performed; evidence artifact; tolerance
or uncertainty; outcome; and remaining limitation.
Use outcomes `supported`, `violated`, `numerically unresolved`, or
`not assessed`. A synthetic model choice being correctly implemented is
not evidence that it is physically true.

## 10. Experiment inventory and execution order

The table below consolidates the experiments and validation suites into an
execution sequence. The prerequisite model-specification gate in Section 9.8
is complete; none of the 27 suites below is marked complete by that check.
"Core" identifies the main evidence sequence; "supporting"
identifies complementary validation routes, not additional mandatory large
sampling grids. Optional extensions are not prerequisites for resolving the
core coordinate-invariance question.

Order numbers are execution positions, not experiment identifiers from other
lists. Checks within a phase may run together once their inputs are available.
Run cheap deterministic checks across the primary transform families, starting
in 2-D and repeating in 4-D; reserve expensive sampling for the gated cases
and grids specified in Sections 4--5.

| Order | Phase | Experiment or check suite | Work and required comparison | Scope / gate | Plan sections |
|---|---|---|---|---|---|
| 1 | Deterministic foundations | Transformation and dynamics validation | Check invertibility, round trips, conditioning, orthogonality, symplectic claims, Hamiltonian structure, and the closed-form oscillator propagator. | Core | 3, 9.2--9.4 |
| 2 | Deterministic foundations | Generalized-threshold invariance | Compare generalized eigensolution, Cholesky whitening, and independently bracketed sign changes; verify Arm A preserves the baseline threshold. | Core; after 1 | 2, 4 Stage 1, 9.3 |
| 3 | Deterministic foundations | Decisive squeeze comparison | At `omega = 2`, `a = sqrt(2)`, `r = 0.9`, verify Arm A has threshold `1.5` and is infeasible, while Arm B has threshold zero and is feasible; check the squeeze formula across strengths. | Core; after 2 | 2, 5 |
| 4 | Deterministic foundations | Diffusion feasibility and boundary sweep | Check below/at/above-threshold inertia, boundary multiplicities, and transported negative-direction witnesses using scale-aware tolerances. | Core; after 2 | 4 Stage 1, 5, 9.3 |
| 5 | Deterministic foundations | Marginal-law and transport validation | Verify means, covariances, density Jacobians, scores, schedule, and conditional perturbation law; independently validate Gaussian and mixture scores. | Core transport/score checks; supporting residual diagnostics | 4 Stage 1, 9.2--9.3 |
| 6 | Deterministic foundations | Independent Arm B implementation | Compare direct isotropic noise in transformed coordinates with the pulled-back anisotropic model of shape `A^{-1} A^{-T}`. | Core; after 5 | 4 Stage 1, 9.2 |
| 7 | Deterministic foundations | Property-based and metamorphic checks | Test inverse/composition, whitening, covariance scaling, drift shifts, metric identities, correlated shapes, nonnormal drifts, distinct frequencies, and cross-pair maps. | Core inverse/composition checks; supporting broader cases | 9.4, 9.8 |
| 8 | Deterministic foundations | Deliberate-failure controls | Detect wrong covariance/score transport, false path coupling, local clipping mislabeled as transport, omitted Jacobians, wrong endpoint targets, raw-coordinate metrics, and missing covariance derivatives. | Core; exercise each diagnostic when available and finish before accepting its results | 9.5, 9.8 |
| 9 | Forward feasibility | Finite-time transition validation | Compare direct transition covariance with quadrature; check and simulate PSD transitions above/at threshold, and exhibit a local negative direction below it. | Core; after deterministic feasibility checks | 9.3, 9.8 |
| 10 | Gaussian pilot | Gaussian reverse-moment reference | Use exact Gaussian scores and an independently written mean/covariance ODE; compare all three samplers to exact marginal moments. | Core; deterministic gate must pass | 4 Stages 1--2, 9.3 |
| 11 | Gaussian pilot | Fully transported sampler equivalence | Transport terminal draws, drift, diffusion factors, and Brownian increments for each sampler; verify pulled-back paths agree using `A L` and identical updates. | Core; Gaussian first, selected mixture confirmation later | 4 Stage 2, 6 |
| 12 | Gaussian pilot | Local spectral-clipping intervention | Measure `(A G A^T)_+ - A G_+ A^T`; compare locally clipped/corrected samplers separately from fully transported samplers, including boundary and above-threshold agreement. | Core; after transport and score checks | 4 Stage 2, 6 |
| 13 | Gaussian pilot | Marginal-evolution PDE checks | Check probability-flow continuity and corrected reverse-SDE Fokker--Planck identities at held-out points/times, using Gaussian derivatives first and mixture finite differences later. | Supporting independent oracle | 9.3 |
| 14 | Calibration and refinement | Metric and Monte Carlo calibration | Use target-vs-target controls, shared references, and pulled-back metrics; set practical equivalence margins before the main mixture comparisons. | Core; after Gaussian reference; calibrate mixture metrics before main grid | 6, 9.2, 9.7 |
| 15 | Calibration and refinement | Time-step and particle-count convergence | Refine representative below/boundary/above cases at `K = 1000, 2000, 4000, 8000` with coupled paths; independently refine particle count and include the worst-conditioned primary transform. | Core Gaussian gate; repeat selected mixture cases before final interpretation | 5, 9.7--9.8 |
| 16 | Main exact-score study | Fitted-GMM sampling grid | Keep one 24-component fit fixed; run ODE, clipped, and corrected samplers for baseline and both arms with exact terminal samples, correct arm-specific scores, and smoothed targets. | Core; Gaussian gate, metric calibration, and timing estimate must pass | 4 Stage 2, 5--6 |
| 17 | Trained-score extension | Transported baseline network | Wrap one baseline network for Arm A and compare pulled-back results, isolating transport from retraining error. | Conditional on settled exact-score conclusions | 4 Stage 3 |
| 18 | Trained-score extension | Separate Arm B training | Train for the new isotropic geometry with appropriate targets and covariance-aware normalization. | Conditional; after 17 | 4 Stage 3 |
| 19 | Trained-score extension | Arm A retraining and conditioning | Retrain in transformed coordinates to measure optimization/conditioning effects, not theorem validity. | Optional; after transported-network baseline | 4 Stage 3 |
| 20 | Likelihood extension | Quadratic score-error invariance | Verify transport preserves `e^T G e` and its sign, using anisotropic-covariance-aware score errors. | Optional; after deterministic transport checks; no retraining required | 4 Stage 3, 9.4 |
| 21 | Sensitivity | Alternative noise geometries | Compare isotropic, energy-aligned, and correlated SPD shapes, with separately documented trace- and determinant-matching controls. | Optional; after core exact-score study | 2, 9.6 |
| 22 | Sensitivity | Alternative data and controlled score errors | Compare single Gaussian, separated/overlapping mixtures, and fitted two-moons data; then distinguish transported from independently introduced score errors. | Optional; controlled errors follow exact-score references | 9.6 |
| 23 | Sensitivity | Terminal-prior approximation | Compare exact terminal initialization with the repository's Gaussian moment prior and measure each prior's own discrepancy. | Optional separate control; after exact-initialization reference | 4 Stage 2, 9.6 |
| 24 | Sensitivity | Alternative schedules | Test positive non-exponential schedules; separately test time-varying covariance shapes with the additional covariance-derivative term. | Optional; separate from fixed-shape exponential claims | 9.6 |
| 25 | Numerical stress | Extreme transforms and precision | Study extreme squeezes separately; use higher precision when conditioning or boundary eigenvalues leave float64 classification unresolved. | Optional stress study; precision escalation is required if a core case is unresolved | 3, 9.7 |
| 26 | Broader coordinates | Time-dependent and nonlinear maps | Treat time-dependent linear maps and nonlinear maps as separate follow-on studies, including additional drift and Ito terms. | Out-of-core follow-on work | 9.6 |
| 27 | Physical interpretation | External physical validation | If evidence is available, compare local drift, residual covariance, units, and scale choices with physical measurements or mechanistic constraints. | Conditional on external evidence; not established by synthetic checks | 9.2, 9.6 |

The core sequence is: model specification, deterministic identities and
failure controls, forward feasibility, Gaussian pilot, calibration and
refinement, then the main mixture study. Mixture-specific score, metric,
transport, and convergence checks accompany the mixture phase rather than
being assumed to follow from the Gaussian pilot.

Use the transforms, absolute rates, seeds, sample counts, and coupled
refinement grids in Sections 3 and 5. Do not interpret this inventory as a
request to run every optional extension or every Cartesian combination.
After the core runs and selected controls are complete, produce the analysis,
figures, results report, README interpretation, and assumption ledger described
in Sections 7 and 9. These are deliverables, not separate experiments.
