# Gap 1 experiment plan: Is the threshold coordinate-invariant?

**Priority:** Critical  
**Status:** Model-specification gate complete; representative forward-feasibility and Gaussian-sampling pilots complete; full numerical experiment gates pending
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

#### Executable Gaussian pilot

```bash
python verification/gaussian_sampling.py --output results/gaussian_sampling.json
python -m unittest verification.test_gaussian_sampling
```

The pilot records its nondegenerate Gaussian clean law and runs the ODE,
naive clipped, and corrected samplers with exact scores and terminal draws.
It uses an independently implemented adaptive reverse-moment ODE as the
continuous reference. Exact moments of the discrete affine updates separate
integration bias from Monte Carlo error; those updates are regression-tested
against `psdt.sample` in the isotropic case.
Defaults cover 2-D and 4-D, absolute rates `0.9, 1.5, 1.95`, step counts
`1000, 2000, 4000, 8000`, particle counts `2000, 4000`, and seeds `0, 1, 2`.
Each rate has its own horizon ending at `sigmaT = 40`, shared across arms.

Baseline paths are compared with fully transported paths for the identity,
the decisive `a = sqrt(2)` squeeze, and the recorded `a = 2` squeeze.
The latter attains the largest condition number in the listed primary
controls. Arm B is sampled with its own isotropic covariance and locally
computed clipping for both nonidentity squeezes; its complete sampler is
also checked against its pullback. Transport uses `A L`, not local
re-clipping. Path comparisons track up to 32 particles at every step.
Fine Brownian increments are summed for coarse paths, and terminal draws
are reused across step counts. Particle refinement uses nested prefixes;
only seeds count as independent experimental replicates.

The JSON records declared acceptance bands, exact target and reference
moments, sampled moments, integration errors, target KL, path residuals,
and seed-level particle-refinement results. Below-threshold naive bias is
measured, not treated as an automatic failure. `--steps`, `--samples`,
`--seeds`, `--dimensions`, and `--rates` can reduce or extend the pilot;
coarse step counts must divide the finest count. Exit status is nonzero
on failed checks. This representative pilot does not automatically clear
the full Gaussian gate or its deterministic/forward-feasibility prerequisites.

#### Executable mixture pilot

Run the exact-score mixture verifier against the fixed fitted mixture:

```bash
OPENBLAS_NUM_THREADS=1 python verification/mixture_sampling.py \
  --output results/mixture_sampling.json \
  --log-file results/mixture_sampling.log
python -m unittest verification.test_mixture_sampling
```

Timestamped logs stream immediately to the console and append to `--log-file`.
They identify the active case/rate, calibration, score checks, seed,
integration progress every 10%, elapsed time, and final outcomes.
Watch the file from another terminal with `tail -f results/mixture_sampling.log`.
The JSON report is written only after the full run finishes.

For a smaller primary-rate pilot (not the default grid):

```bash
OPENBLAS_NUM_THREADS=1 python verification/mixture_sampling.py \
  --rates 0.9 --steps 1000 2000 4000 --samples 128 256 --seeds 0 1 2 \
  --output results/mixture_sampling.json \
  --log-file results/mixture_sampling.log
```

The default cache is `results/gmm_fit.npz`, produced by
`gmm_exact.fit_gmm`; use `--mixture PATH` to select another fixed cache.
Missing or invalid caches fail explicitly rather than triggering a new fit.
The report records all clean mixture parameters and a SHA-256 checksum.

The pilot uses the 4-D baseline and decisive/recorded re-isotropized
squeezes, at rates `0.9, 1.5, 1.95`, with exact terminal-mixture draws.
It checks analytic scores against independent SciPy log-density finite
differences, score transport, and density Jacobians. All three samplers
share terminal draws and Brownian increments; coarse grids sum fine
increments. Full sampler transport uses `A L`, not local re-clipping.
Arm B has its own recomputed exact score and pulled-back smoothed target.

Defaults are `--steps 1000 2000 4000`, `--samples 512 1024`,
`--seeds 0 1 2`, and `--calibration-repeats 12`. The JSON records
pulled-back mean/covariance errors, variance ratios, sliced Wasserstein,
unbiased MMD squared, component-responsibility masses, and a clearly labeled
Gaussian-moment KL proxy. Independent target draws calibrate each particle
count before sampler comparisons, with fixed projections, bandwidths, and
shared references. Seed-level paired naive-minus-corrected effects and
standard errors are reported; nested particle prefixes are not replicates.

Acceptance uses declared score/path tolerances and empirical target-control
bands plus fixed practical margins. ODE/corrected and feasible naive
finest-grid seed-mean metrics must lie within those bands and show no
degradation beyond calibration-scale uncertainty. Below-threshold naive
target bias is measured rather than required to vanish. Stepwise coupled
endpoint differences and particle refinement are reported separately.
These are representative smoke-test bands, not formal equivalence
intervals or proof of asymptotic convergence. Failed checks produce a
nonzero exit status; this pilot does not clear the full mixture gate,
deterministic prerequisites, or full transformation grid automatically.

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

Run the representative forward-feasibility verifier with:

```bash
python verification/forward_feasibility.py --output results/forward_feasibility.json
python -m unittest verification.test_forward_feasibility
```

It validates the recorded model specification, then checks the baseline,
transported, and re-isotropized arms in 2-D and 4-D for the recorded squeeze.
Each arm uses rates `kappa_Q - 0.3`, `kappa_Q`, and `kappa_Q + 0.3` on the
fixed primary horizon; these rate-sweep endpoints need not equal `sigmaT`.
The JSON records matrices, quadrature comparisons, scale-aware roundoff
bands, local witnesses, witness transport, and seeded Gaussian forward-path
moment checks starting from an explicitly recorded smoothed initial law.
Only bounded roundoff negatives may be neutralized when factoring an
already-classified PSD covariance, and their correction is reported.
Exit status is nonzero on failed checks. This representative verifier does
not establish the full transformation grid, clear the deterministic
prerequisite, or automatically mark the Section 9.8 gate complete.

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
| Deterministic implementation | [ ] Partial: inventory row 1 supported (2026-10-21); full gate pending | Transformation/dynamics evidence: [`results/deterministic_foundations.md`](results/deterministic_foundations.md). Remaining gate requires closed-form squeeze threshold checks, generalized threshold via independent routes, score/density transport, and the broader inverse/composition and deliberate-failure controls |
| Forward feasibility | [x] Representative pilot done (2026-10-17); full-grid gate pending | PSD transition covariances above/at threshold and a local negative-direction witness below it; passing pilot evidence: [`results/forward_feasibility.json`](results/forward_feasibility.json) |
| Gaussian sampling | [x] Representative pilot done (2026-10-17); full gate pending | Independent moment reference, exact terminal initialization, transported-path equivalence, and step refinement; passing pilot evidence: [`results/gaussian_sampling.json`](results/gaussian_sampling.json) |
| Mixture sampling | [ ] Pending | Independently checked exact scores, paired seeds, target-vs-target calibration, and pulled-back diagnostics |
| Optional generalization | [ ] Runner implemented; bounded all-axis analysis done (2026-10-21); full study not assessed; optional | Sensitivity to other noise shapes, priors, data, and trained scores reported separately from core conclusions; evidence and implementation details in Section 11 |

Checked pilot entries record completed representative runs, not clearance
of the full gates. Transformation/dynamics checks now cover the primary
transform grid in 2-D and 4-D; full-grid forward/sampling coverage and
certification of the remaining deterministic prerequisites are outstanding.

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
| 5 | Deterministic foundations | Marginal-law and transport validation | Verify means, covariances, density Jaobians, scores, schedule, and conditional perturbation law; independently validate Gaussian and mixture scores. | Core transport/score checks; supporting residual diagnostics | 4 Stage 1, 9.2--9.3 |
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

### 10.1 Executed transformation and dynamics validation (inventory row 1)

**Status:** Implemented and supported (2026-10-21). This completes row 1,
not the entire deterministic-foundations phase (rows 1--8).

```bash
OPENBLAS_NUM_THREADS=1 python verification/deterministic_foundations.py \
  --output results/deterministic_foundations.json
python -m unittest verification.test_deterministic_foundations
```

The runner reuses the production `hamiltonian_B` and `Process.M` and checks
them against independently written oscillator equations and the closed-form
propagator. It validates the recorded primary specification before execution.
All six Section 3 families and their listed strengths run in 2-D and 4-D;
the 4-D grid additionally includes a cross-pair canonical map. A second 4-D
grid uses distinct frequencies `(1.3, 2.7)` to expose ordering/mixing errors.
States include basis vectors and fixed-seed random probes. Tests cover
singular values, conditioning, determinant, orthogonality and symplectic
claims, Poisson brackets, drift Jacobians, Hamiltonian/energy transport,
state/drift/energy round trips, state/drift composition, propagator similarity,
negative-time inversion, and independent numerical evolution.
Noncanonical maps use the transported Poisson matrix, not an incorrectly
fixed canonical `J`.

The [analysis](results/deterministic_foundations.md) records **47 supported
cases, 1,774 passing identity/classification checks, and six detected
deliberate errors**. The maximum primary condition number is 4.
Maximum normalized residuals are `3.26e-14` for propagator similarity,
`9.09e-12` for the independently integrated baseline oscillator, and
`1.75e-11` for transformed numerical evolution. The integration acceptance
band is `2e-10`; matrix and finite-difference bands are recorded separately.
Controls reject singular maps, false orthogonality, false symplectic claims,
wrong transformed drift, wrong oscillator signs, and a determinant-one
noncanonical 4-D map.

The [machine-readable evidence](results/deterministic_foundations.json)
includes every matrix, singular value, residual, tolerance, control,
environment version, and source/specification hash. The runner writes the
JSON and adjacent Markdown analysis and returns nonzero on failure.
Regression tests are included in CI. These results establish numerical
consistency of the implemented coordinate maps and dynamics only; threshold,
diffusion, score/density, sampler, and physical claims remain separate checks.

## 11. Implementation plan: optional generalization

**Status:** Runner, regression coverage, and bounded all-axis analysis
implemented/executed (2026-10-21); full empirical matrix not assessed.
This section specifies work for inventory
items 17--23; it does not mark any main-study experiment as passed.
Implement cheap infrastructure and unit tests first, but accept empirical
generalization results only after the relevant core gates in Section 9.8 pass.
The current representative pilots do not clear those full gates.

Here "priors" means **reverse-time terminal initialization**, distinct from
the clean data law. Alternative clean distributions are covered under data.
The finite matrix below covers each requested axis, not every possible
distribution or every Cartesian combination. Non-exponential schedules,
time-dependent shapes, nonlinear coordinates, and physical identification
remain separate follow-on work under inventory items 24--27.

#### Executable runner

`verification/generalization.py` consolidates the case specification, geometry,
data, priors, training, and orchestration responsibilities proposed below.
It reuses the existing samplers and metrics rather than introducing separate
implementations for each axis. `generalization_analyze.py` produces coverage,
paired comparisons, nested training/sampler uncertainty summaries, and figures.
Neither runner certifies the outstanding core prerequisite gates.

Inspect the generated full matrix without executing it:

```bash
python verification/generalization.py --profile full \
  --write-spec /tmp/generalization-full.json
python verification/generalization.py --spec /tmp/generalization-full.json \
  --dry-run --axis noise
```

Start with a small smoke subset in its own output directory:

```bash
OPENBLAS_NUM_THREADS=1 python verification/generalization.py \
  --profile smoke --axis noise --max-cases 3 \
  --output results/coordinate_invariance/generalization-smoke
python generalization_analyze.py \
  --input results/coordinate_invariance/generalization-smoke
```

Smoke defaults use two sampler seeds, 64/128 particles, 128/256 steps,
and only 20 training updates; they exercise the pipeline, not main-study
accuracy. A coarse smoke case can legitimately fail quality checks, retain
its evidence, and exit nonzero. Do not widen acceptance bands to make it pass.
Use a custom manifest to refine a problematic case in a **new** output
directory. To make a bounded runtime estimate, select one case with
`--case CASE_ID` from `--dry-run` output.

After checking prerequisites, budgets, and pilot calibration, execute selected
full-profile axes or cases:

```bash
OPENBLAS_NUM_THREADS=1 python verification/generalization.py \
  --spec /tmp/generalization-full.json --axis noise data prior \
  --output results/coordinate_invariance/generalization-full --resume
OPENBLAS_NUM_THREADS=1 python verification/generalization.py \
  --spec /tmp/generalization-full.json --axis score trained interaction \
  --output results/coordinate_invariance/generalization-full --resume
python generalization_analyze.py \
  --input results/coordinate_invariance/generalization-full
python -m unittest verification.test_generalization
```

The full manifest is deliberately expensive: it includes 8000 particles,
8000-step refinement, five sampler seeds, and three training seeds with
12000 updates each. It is never executed by CI. Review the manifest rather
than treating these commands as a request to launch the whole grid immediately.
Use `--mixture PATH` when generating a manifest to select the frozen fitted
mixture; missing caches fail explicitly without fitting new data.

Outputs include `manifest.json`, `progress.log`, immutable `cases/*.json`,
`seeds/*-SEED.json`, `failures/*.json`, `checkpoints/*-{best,last}.npz`, and incremental
`summary.json`. Completed cases are reused only with `--resume` and matching
manifest, source/environment, input, and checkpoint fingerprints.
Completed sampler seeds are also persisted before proceeding to the next seed,
so interrupted cases can resume without repeating their completed integrations;
seed-level reuse requires the same manifest, data, code, and selected checkpoint.
Training resumes from the last optimizer state but samples with the checkpoint
selected by held-out validation loss. Each case records target controls,
terminal discrepancy, pulled-back endpoint metrics, coupled refinement,
analytic score checks, Gaussian reverse-moment references where applicable,
time-binned score error, timing, and process peak RSS.
The analyzer writes `analysis/analysis.json`, `analysis/report.md`, separate
`analysis/{noise,data,prior,score,trained,interaction}-summary.json` files,
`analysis/assumption-ledger.json`, and plots
for quality, positions, refinement, thresholds, prior discrepancy, score error,
particle-count refinement, and training variability. It also exposes
held-out raw-data metrics, velocity moments, component responsibility masses,
paired baseline-arm effects, missing cases, measured compute cost, and
full-profile operation-count ratios. Analysis rejects changed checkpoints.
Raw-data held-out metrics and signed `e^T G e` remain explicitly
labeled; descriptive seed intervals are not formal equivalence certificates.
The analyzer's `study_acceptance` remains `not assessed`: completing a
smoke/custom manifest, even with all implementation checks passing, does not
clear the prerequisite gates or the full-budget study.

The smoke generator covers both Gaussian and frozen fitted-mixture noise
controls and all six trained-data families. Its default single training seed
is only an interface smoke check; nested training-variability evidence
requires a separately frozen manifest with at least three training seeds.
The full generator currently contains 4224 cases. Review its compute budget
before execution; its Cartesian training coverage is much larger than the
bounded smoke matrix.

#### Bounded execution evidence (2026-10-21)

The frozen [all-axis manifest](results/coordinate_invariance/generalization-all-axes/manifest.json)
completed 213 cases at 64/128 particles, 128/256 steps, two sampler seeds,
and three training seeds with 20 updates each. This is pipeline and numerical
pilot evidence, not the full empirical study. See the
[analysis report](results/coordinate_invariance/generalization-all-axes/analysis/report.md),
[machine-readable analysis](results/coordinate_invariance/generalization-all-axes/analysis/analysis.json),
and [assumption ledger](results/coordinate_invariance/generalization-all-axes/analysis/assumption-ledger.json).

| Axis | Supported implementation/declared controls | Violated quality controls | Not assessed |
|---|---:|---:|---:|
| Noise shape | 0 | 42 | 0 |
| Data family | 2 | 16 | 0 |
| Terminal prior | 36 | 6 | 0 |
| Controlled score errors | 12 | 3 | 0 |
| Trained scores | 72 | 0 | 0 |
| Selected interactions | 22 | 2 | 0 |

All 213 cases passed their algebra, independent score/density, quadratic
score-error, and coupled-path implementation checks. The 69 `violated`
outcomes are retained coarse-grid quality failures, not harness exceptions;
the runner correctly returned nonzero. Acceptance bands were not widened.
`supported` learned-score or approximate-prior cases mean their applicable
implementation/control requirements passed, not that their samples match
the exact-score target or that 20 updates establish learned-score accuracy.
There are 150 within-axis paired comparisons, 148 baseline-arm comparisons,
30 nested training groups, and 29 figures. Across the 128 paired sampler
runs eligible for the full-transport identity (excluding independent errors),
the maximum pulled-back endpoint preview residual was `1.25e-13`.
Case execution totaled 416.5 seconds with 133.9 MiB process peak RSS;
the latter is a process-wide high-water mark, not per-case memory.

The [refinement pilot](results/coordinate_invariance/generalization-refinement/analysis/report.md)
uses 256/512 particles and 1000/2000 steps for Gaussian and frozen fitted-mixture
controls at rates `0.9` and `1.95`, with all three arms and two sampler seeds.
Its original process was interrupted after 11 of 12 completed cases;
those 11 immutable records remain valid under their original source/revision
fingerprint and are not silently merged with a different execution identity.
The missing fitted-mixture Arm B case was subsequently completed in its own
[one-case execution](results/coordinate_invariance/generalization-refinement-completion/analysis/report.md)
and was supported. Thus every case in the 12-case pilot has evidence, but the
original interrupted manifest correctly retains one `not assessed` entry.
One completed Gaussian Arm B case at `r = 0.9` still violated the covariance
band at 2000 steps, despite passing its independent moment and transport checks.

That case was separately refined at 1000/2000/4000/8000 steps and
512/2000 particles, with unchanged acceptance policy. The
[fine Gaussian report](results/coordinate_invariance/generalization-gaussian-fine/analysis/report.md)
records a supported outcome. Within its coupled 2000-particle runs, the
naive/corrected covariance error decreased from `0.3680` at 1000 steps to
`0.1160` at 8000 steps, below the independently calibrated band `0.1937`.
The ODE covariance error at 8000 steps was `0.1057`; all three samplers
passed their independent Gaussian-reference controls. This is evidence of
numerical refinement in this case, not clearance of the full Gaussian gate.

After adding immutable per-seed integration records and explicit checkpoint
data/specification hashes and normalization matrices, a separate
[six-case trained-score run](results/coordinate_invariance/generalization-checkpoint-resume/analysis/report.md)
verified the final code path for Gaussian baseline/transported networks at
three training seeds and two sampler seeds. All six applicable control
outcomes were supported, all 12 sampler-seed records were saved, and a CLI
`--resume` reused the completed cases without repeating sampling.
These 20-update networks are still training-interface evidence, not accepted
learned-score accuracy.

The [full-profile manifest](results/coordinate_invariance/generalization-full-spec.json)
is frozen for inspection but was not executed. Relative to this bounded
profile, one full-profile case has about 6104 times the sampler
particle-step exposure and 11719 times the calibration particle-pair
exposure; empirical oracles also have 32 times as many components. These
are operation-count ratios, not wall-clock estimates. The outstanding
Section 9.8 core gates, full-budget runs, and accepted empirical conclusions
remain separate, uncompleted work.

### 11.1 Shared infrastructure and implementation sequence

Keep the existing primary specification and baseline outputs unchanged.
Store the extension specification in a proposed
`coordinate_invariance_generalization_spec.json`, with explicit matrices,
distribution parameters, case IDs, seeds, and acceptance margins. Validate
it separately rather than weakening `model_specification.validate` to permit
unrecorded departures from the primary model.

| Step | Proposed implementation | Depends on | Completion evidence |
|---|---|---|---|
| G1: case specification | Add `verification/generalization_specification.py` and focused tests for the extension manifest, dimensions, finite values, SPD shapes, normalization, prior policies, and score sources. | Recorded primary specification | Valid manifest accepted; malformed and incompatible cases rejected with case-specific errors |
| G2: reusable exact-score execution | Extract only needed reusable model, marginal, metric, and coupled-integration helpers from `gaussian_sampling.py` and `mixture_sampling.py`; add explicit clean-law, terminal-initializer, and callable score inputs. | G1 | Existing pilot results preserved; Gaussian and mixture regressions pass; arbitrary-time scores evaluated correctly at both Heun stages |
| G3: noise geometry | Add `verification/noise_geometry.py` and tests using the generalized threshold and independent Gaussian moment reference. | G2 and deterministic/forward gates | Shape-specific algebra, forward covariance, and reverse sampling reports |
| G4: data families | Add `verification/generalization_data.py` and tests for exact synthetic laws and a frozen empirical-data oracle. | G2; G3 before non-isotropic data runs | Independently checked scores, density transport, target draws, and per-family calibration |
| G5: terminal priors | Add `verification/terminal_priors.py` and tests for exact, moment-Gaussian, noise-only, and controlled-mismatch initialization. | G3--G4 exact-initialization references | Initial and final discrepancy reports with unchanged score and model |
| G6: score errors and training | Add `verification/trained_scores.py` and tests; adapt the existing NumPy `MLP`/`ScoreModel` only where required for covariance-aware training and checkpoints. | G3--G4; exact-prior reference from G5 | Analytic score-error controls, transported-network checks, held-out errors, and seeded training comparisons |
| G7: orchestration and reporting | Add `verification/generalization.py` and `generalization_analyze.py`; integrate cheap tests into CI and independent deterministic checks into `verification/run_all.py`. | Relevant G1--G6 tests | Validated run manifest, resumable case artifacts, separate axis summaries, and an assumption ledger |

The per-responsibility module names in the table are architectural proposals;
the executable implementation currently consolidates them in
`verification/generalization.py` with `verification/test_generalization.py`.
Do not create duplicate implementations solely to match the proposed filenames.
Reuse `GaussianModel`, `Mixture`, `marginal`, `draw_marginal`, `threshold`,
the coupled Brownian construction, and calibrated metrics where applicable.
Do not duplicate the samplers for each axis or require a new training framework.
Preserve existing command-line defaults, reports, and isotropic behavior.

### 11.2 Noise-shape cases and checks

Start with fixed SPD shapes in the recorded four-dimensional coordinates:

| Case | Shape before amplitude matching | Purpose |
|---|---|---|
| Isotropic control | `Q = I` | Reproduce the baseline |
| Energy-aligned | `Q = H^{-1}`, `H = diag(4, 1, 4, 1)` | Isotropy in oscillator energy coordinates; `B Q + Q B^T = 0`, hence threshold zero |
| Coordinate anisotropy | `Q = diag(4, 1, 0.25, 1)` | Unequal position/momentum and cross-pair scales |
| Correlated, cross-pair | `Q = L L^T`, with unit diagonal lower-triangular `L` and `L[1,0]=0.5`, `L[2,0]=0.3`, `L[3,1]=-0.4` (zero-based indices) | Off-diagonal covariance and indexing coverage |

For each nonbaseline shape, run two separately labeled normalization policies:
`Q_trace = d Q / tr(Q)` and
`Q_det = Q / det(Q)^(1/d)`. Compute determinant normalization through
`slogdet`; record the raw shape, normalized shape, eigenvalues, condition
number, trace, determinant, and policy. Normalization is defined in the
baseline coordinates and then transported, never reapplied in Arm A.
Include a cheap `Q -> c Q` threshold identity test.

Generalize the baseline to the chosen `Q`. Arm A must use
`Q_Y = A Q A^T`; Arm B continues to mean reimposing `Q_Y = I`, not silently
reusing the alternative baseline shape. Use each arm's own smoothed target
and exact terminal law. Recompute the generalized threshold for every model;
do not reuse the isotropic value `1.5` for classification.

Before sampling, compare generalized eigensolution and whitening, covariance
balance, finite-time transitions, and transported negative-direction witnesses.
Use PSD-compatible factors at threshold. Check exact Gaussian reverse moments
and fully transported paths, keeping local clipping distinct from transporting
the baseline clipped factor. Energy alignment supplies a known zero-threshold
control; correlated shapes supply noncommuting transport controls.

### 11.3 Data families and independent references

Record complete generator parameters and freeze them before collecting main
results. Use the existing Gaussian fixture and fitted 24-component cache as
controls; never silently refit a cache during a sampling run.

| Family | Initial concrete cases | Score/reference strategy |
|---|---|---|
| Single Gaussian | Existing pilot; add a fixed correlated SPD covariance | Closed-form marginal score and independent moment ODE |
| Separated mixture | Two equal-weight components with means `+/- (2,0,0,0)` and covariance `0.1 I` | Exact propagated mixture; finite-difference log-density validation |
| Overlapping mixture | Same weights/covariance, means `+/- (0.3,0,0,0)` | Exact mixture; overlap-sensitive responsibility diagnostics |
| Imbalanced mixture | Separated case with weights `0.9, 0.1` | Exact mixture; minority-component mass and uncertainty |
| Fitted two-moons mixture | Existing fixed fit and its checksum | Existing exact-score mixture oracle; conclusions concern the fitted law |
| Raw two-moons data | Freeze 2048 training and 2048 held-out points from independent `two_moons` streams | Smoothed finite empirical-mixture oracle for training points; held-out generalization reported separately |

For the empirical law, each clean training point is a point mass; at time
`t` the marginal has component mean `M(t) x_i` and shared SPD covariance
`sigma(t)^2 Q`. Implement its score with stable log-sum-exp and chunked
component accumulation, not by supplying singular clean covariances to
`Mixture`, which currently requires SPD components. Exact terminal and target
draws select a point uniformly and add the declared Gaussian perturbation.
Check this score against an independently evaluated density on a small fixture.
Label its exactness as relative to the frozen empirical law, not an unknown
population density. Measure held-out smoothed-data quality separately; it is
not an exact-target equivalence check.

Validate marginal moments, finite-difference scores, score/density transport,
and initial smoothing for every oracle. Calibrate metrics separately by data
law and shape. Add component-mass diagnostics for mixtures and position-space
plots plus velocity moments for moons; do not call moment KL the true KL.
Use the same declared data scaling across arms and record any preprocessing.

### 11.4 Terminal-prior comparison

Hold the data law, score, schedule, and sampler fixed while changing only the
terminal initializer:

| Initializer | Definition | Interpretation |
|---|---|---|
| Exact terminal control | `M(T) X + sigmaT L_Q epsilon` | Reference for every model/arm |
| Oracle moment Gaussian | Mean `M(T) m_X`, covariance `M(T) S_X M(T)^T + sigmaT^2 Q` | Isolates loss of non-Gaussian terminal structure |
| Repository-style estimated moment Gaussian | Same formula using one frozen clean-data sample's empirical moments | Separates moment-estimation error from Gaussian approximation |
| Noise-only Gaussian | Zero mean, covariance `sigmaT^2 Q` | Measures omission of propagated clean signal |
| Controlled mismatch | Oracle moment prior with mean shifted by `0.5` along a declared unit baseline direction, or covariance multiplied by `0.8` or `1.2` | Positive controls for prior-sensitivity diagnostics |

Use analytic moments when available; keep estimated moments labeled as
estimated rather than "exact." Arm A priors, including deliberate shifts,
must be pushforwards of baseline priors, with transported sampling factors.
Recompute Arm B moments for its own model. Share base standard normals where
possible, but do not claim non-Gaussian and Gaussian terminal draws are equal.

Measure discrepancy from the exact terminal law **before integration** and
from the correct smoothed target **after integration**. Report approximate-
minus-exact prior effects paired by run seed, separately for every sampler.
A Gaussian clean law makes the oracle moment prior exactly correct and
provides a mandatory null control. Do not require prior mismatch to vanish
under step refinement or attribute its residual to threshold failure.

### 11.5 Controlled errors and trained-score experiments

First inject score errors into exact Gaussian/mixture scores: zero error,
a fixed vector bias, multiplicative scale error, and a smooth bounded
time-dependent error, at amplitudes `0.05, 0.1, 0.2` relative to a frozen
reference score RMS. Record the normalization metric and calibration sample.
Transport the complete erroneous score in Arm A; contrast with separately
generated transformed-coordinate errors as a different intervention.
Use exact initialization first so prior and score errors are not confounded.

Then implement the following training sequence:

1. Verify the existing isotropic training/inference interface, including
   batched-time broadcasting in `ScoreModel.score`, checkpoint round trips,
   finite gradients, and deterministic seeded smoke runs. For general `Q`,
   train on `z_t = M(t) x + sigma(t) L_Q epsilon`; a whitened epsilon predictor
   returns the score `-L_Q^{-T} epsilon_theta / sigma(t)`. The row-array form
   is `-epsilon_theta @ L_Q^{-1} / sigma(t)[:, None]`. Test both forms against
   the analytic conditional score, and use covariance-aware input scaling.
   Record whether loss is epsilon MSE or a specified score-space quadratic;
   never carry over an isotropic score weighting without deriving it.
2. Train one baseline network per selected data/noise case. Start with a
   Gaussian, separated mixture, and fitted moons mixture, then run the
   overlapping, imbalanced, and raw-data cases after timing. Use the existing
   width 128, depth 3, batch 512, and 12000-update setup as the initial budget.
   Freeze configuration after pilot tuning; choose checkpoints by validation
   loss, not final sampler quality. Save model parameters, normalization,
   optimizer state, step, seed, configuration, and data/specification hashes.
3. Wrap that same network for Arm A:
   `s_Y(y,t) = A^{-T} s_Z(A^{-1} y,t)`. Compare scores and coupled paths after
   pullback before any retraining. This identity must hold even for an
   inaccurate network; sample quality need not match the exact-score oracle.
4. Train separate Arm B models with their own perturbation geometry and
   smoothed targets. Separately retrain Arm A models to measure conditioning
   and optimization sensitivity, not coordinate-invariance validity.
   Match architecture, data splits, update budget, and checkpoint policy;
   record elapsed time and any tuning departures.
5. Evaluate held-out denoising loss and exact-score error where an oracle
   exists, stratified by time/noise level, plus endpoint sampling quality.
   Report `e^T Q e` as a nonnegative transported score-error norm.
   If reporting `e^T G e`, preserve its sign and label it as a signed
   quadratic when `G` is indefinite, not a nonnegative loss.

Use disjoint training, validation, metric-calibration, and final evaluation
streams. For empirical moons, report error relative to the training empirical
oracle separately from quality on unseen data. No failed training run may be
silently replaced by an exact score or a different seed.

### 11.6 Run matrix, acceptance, and resource controls

Use identity, the decisive symplectic squeeze, and one recorded moderate-
condition cross-pair map initially; include the recorded non-unit-determinant
map for density-Jacobian checks. Keep `sigma0 = 0.01`, `sigmaT = 40`, and the
same absolute rate/horizon within every coordinate comparison.
Start with rates `0.9` and `1.95`. Add `0.9*kappa_Q`, `kappa_Q`, and
`1.1*kappa_Q` only for positive thresholds, as separate rate settings; use
positive absolute rates for zero/negative thresholds.

Run one-axis-at-a-time comparisons before interactions:
noise shapes with the Gaussian and fixed fitted mixture; data families with
`Q = I` and exact priors; priors with exact scores; then trained scores with
exact priors. Finally cross the correlated trace-matched shape, imbalanced
mixture, moment-Gaussian prior, and learned score in selected cases to expose
interactions. Label interaction results separately, not as isolated effects.
Ensure every family/policy above has a completed case or an explicit blocked/
not-assessed record; a small pilot is not completion of the planned matrix.

Use `N = 256, 512`, seeds `0,1`, and `K = 1000,2000` for timing/smoke pilots.
For accepted exact-score comparisons use `N = 8000`, seeds `0..4`, and
`K = 4000`, with selected `1000,2000,4000,8000` coupled refinement and a
separate particle refinement. Use at least three independent training seeds,
each evaluated with the same five sampler seeds; report between-training
variation and within-checkpoint sampler variation separately. Do not count
the fifteen nested runs as fifteen independent training replicates.

Before the main runs, freeze primary metrics, practical margins, and
uncertainty procedures using independent pilot/target controls for each
law/shape. Five sampler seeds and three training seeds are limited evidence;
report raw effects and intervals, and leave equivalence unresolved if
intervals do not fit inside declared margins.

| Evidence class | Acceptance / interpretation |
|---|---|
| Algebra and transport | Scaled residuals satisfy declared condition-aware tolerances; injected wrong covariance/score transforms, factors, and density Jacobians are detected |
| Exact Gaussian reference | Corrected/ODE moments match their exact-initialization target with refinement; prior-mismatched cases match an independently initialized reverse-moment ODE instead |
| Exact-score non-Gaussian quality | Corrected/ODE exact-prior endpoints meet frozen target-control bands with refinement; feasible naive cases use the same standard, below-threshold naive bias is measured rather than required |
| Prior controls | Gaussian exact/moment null agrees; deliberately mismatched priors are distinguished at initialization; final impact is measured, not presumed universally nonzero |
| Learned-score controls | Transported-network scores and paths agree; held-out error and quality shifts have seeded uncertainty; retraining differences are not theorem counterexamples |
| Generalization conclusions | State which tested cases meet criteria, fail, remain numerically unresolved, or were not assessed; never infer universal robustness from this finite grid |

Estimate time and peak memory before expanding the grid, especially for the
empirical oracle and quadratic-cost MMD. Batch score evaluation and metric
work without changing the declared estimator. Persist immutable per-case/
seed artifacts and explicit failure records under
`results/coordinate_invariance/generalization/`; resume only when code,
specification, checkpoint, and input hashes match. Invalid inputs or failed
required checks exit nonzero, while expected quality degradation in a
labeled mismatch experiment is a measured outcome, not a harness error.

Every artifact records matrices, normalization, generalized threshold,
feasibility classification, target/prior definition, score source, data and
checkpoint hashes, all RNG seeds, time grid, tolerances, environment, code
revision and dirty status, elapsed time, and outcome. Keep expensive training
and sampling outside routine CI; CI covers tiny deterministic fixtures,
deliberate failures, checkpoint reloads, and CLI success/failure behavior.

### 11.7 Completion and reporting

Produce separate noise-shape, data-family, prior, and trained-score summaries,
plus a clearly labeled interaction appendix. Plot pulled-back sample quality,
threshold/shape comparisons, terminal-versus-endpoint prior discrepancy,
time-binned score error, training-seed variability, and numerical refinement.
Record coverage against the manifest, missing cases, compute cost, and the
assumption ledger from Section 9.8. Only then update the optional-generalization
gate with links to actual evidence; keep core gate statuses independent.
