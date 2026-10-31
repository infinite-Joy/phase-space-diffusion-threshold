"""Verification suite: every analytical statement and every number in the paper is checked numerically.
Run:  python verification/run_all.py      (exit code 0 iff all checks pass)"""
import sys, os, json
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from psdt import hamiltonian_B, cld_B, kappa_of, Process, batch_M
from verification.deterministic_foundations import run as run_deterministic_foundations
from verification.model_specification import DEFAULT_SPEC

results = []
def check(name, ok, info=''):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")

I4 = np.eye(4)
Bh, Bc = hamiltonian_B(2.0), cld_B(beta=1.0)

# 1-2. kappa values
check('kappa(Hamiltonian, omega=2) = |1-omega^2|/2 = 1.5', abs(kappa_of(Bh) - 1.5) < 1e-12)
for beta in [0.5, 1.0, 4.0]:
    check(f'kappa(CLD, beta={beta}) = beta/2', abs(kappa_of(cld_B(beta=beta)) - beta / 2) < 1e-12)
check('tr B_CLD = -4 beta per position dimension (d=2: -8 beta)', abs(np.trace(cld_B(beta=1.0, d=1)) + 4) < 1e-12 and abs(np.trace(Bc) + 8) < 1e-12)

# 3. G = dC/dt - BC - CB^T equals 2 sigma^2 (rI - SymB), by finite differences
for B in [Bh, Bc]:
    p = Process(B, 0.7 * kappa_of(B)); t, h = 1.3, 1e-6
    C = lambda s: p.sigma(s)**2 * I4
    G_fd = (C(t + h) - C(t - h)) / (2 * h) - B @ C(t) - C(t) @ B.T
    check('G formula matches definition', np.abs(G_fd - p.G(t)).max() / np.abs(G_fd).max() < 1e-6)

# 4. sign of lambda_min(G) flips exactly at r = kappa
for B, nm in [(Bh, 'ham'), (Bc, 'cld')]:
    k = kappa_of(B)
    ok = True
    for rr in [0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0]:
        lm = np.linalg.eigvalsh(Process(B, rr * k).G(0.5)).min()
        ok &= (lm >= -1e-12) == (rr >= 1.0)
    check(f'G >= 0  <=>  r >= kappa ({nm})', ok)

# 5. G_+ and sqrt(G_+)
p = Process(Bh, 0.6 * 1.5)
check('sqrt(G+)^2 = G+', np.allclose(p.sqrtGplus(0.7) @ p.sqrtGplus(0.7), p.Gplus(0.7)))
check('G+ >= 0 and G+ - G >= 0', np.linalg.eigvalsh(p.Gplus(0.7)).min() > -1e-12
      and np.linalg.eigvalsh(p.Gplus(0.7) - p.G(0.7)).min() > -1e-12)

# 6. interpolated expm(Bt) accuracy
for B in [Bh, Bc]:
    p = Process(B, kappa_of(B))
    ts = np.random.default_rng(0).uniform(0, p.T, 50)
    err = max(np.abs(batch_M(p, ts)[i] - expm(B * ts[i])).max() / np.abs(expm(B * ts[i])).max() for i in range(50))
    check('batch_M interpolation rel. error < 1e-5', err < 1e-5, f'({err:.1e})')

# 7. forward marginal: Monte Carlo moments of M z0 + sigma eps vs theory
rng = np.random.default_rng(1)
m0 = np.array([0.5, 0.1, -0.3, 0.0]); S0 = np.diag([1.0, 0.05, 0.5, 0.05])
p = Process(Bh, 1.5); t = 0.8
z0 = rng.multivariate_normal(m0, S0, 400000)
zt = z0 @ expm(Bh * t).T + p.sigma(t) * rng.standard_normal(z0.shape)
Ct = expm(Bh * t) @ S0 @ expm(Bh * t).T + p.sigma(t)**2 * I4
check('forward marginal covariance (MC, 4e5 samples)', np.abs(np.cov(zt.T) - Ct).max() < 0.03)

# 8. Fokker-Planck consistency: for Gaussian p_t, dC/dt = BC + CB^T + G holds along the forward law
for B in [Bh, Bc]:
    p = Process(B, 0.6 * kappa_of(B)); t, h = 0.9, 1e-5
    Cf = lambda s: expm(B * s) @ S0 @ expm(B * s).T + p.sigma(s)**2 * I4
    lhs = (Cf(t + h) - Cf(t - h)) / (2 * h)
    rhs = B @ Cf(t) + Cf(t) @ B.T + p.G(t)
    check('dC/dt = BC + CB^T + G (also for indefinite G)', np.abs(lhs - rhs).max() / np.abs(lhs).max() < 1e-6)

# 9. Exact Gaussian reverse dynamics: KL(generated || true) via moment ODEs
def kl(m1, C1, m2, C2):
    iC2 = np.linalg.inv(C2); d = m2 - m1
    return 0.5 * (np.trace(iC2 @ C1) + d @ iC2 @ d - 4 + np.log(np.linalg.det(C2) / np.linalg.det(C1)))
def reverse_kl(B, r, mode):
    p = Process(B, r)
    def mom(t):
        M = expm(B * t); return M @ m0, M @ S0 @ M.T + p.sigma(t)**2 * I4
    def rhs(tau, y):
        t = p.T - tau; mu = y[:4]; Sg = y[4:].reshape(4, 4); m, C = mom(t); P = np.linalg.inv(C)
        G, Gp = p.G(t), p.Gplus(t)
        K = {'ode': 0.5 * G, 'naive': Gp, 'fixed': 0.5 * (G + Gp)}[mode]
        D = np.zeros((4, 4)) if mode == 'ode' else Gp
        A = -B - K @ P
        return np.concatenate([A @ mu + K @ P @ m, (A @ Sg + Sg @ A.T + D).ravel()])
    m, C = mom(p.T)
    y = solve_ivp(rhs, [0, p.T], np.concatenate([m, C.ravel()]), rtol=1e-9, atol=1e-12, method='DOP853').y[:, -1]
    mt, Ct = mom(0); Sg = y[4:].reshape(4, 4)
    return kl(y[:4], (Sg + Sg.T) / 2, mt, Ct)
for B, nm in [(Bh, 'ham'), (Bc, 'cld')]:
    k = kappa_of(B)
    for rr in [0.6, 0.9, 1.0, 1.3]:
        kls = {md: reverse_kl(B, rr * k, md) for md in ['ode', 'naive', 'fixed']}
        check(f'Theorem 4 (D=0, ODE) exact, {nm} r/k={rr}', kls['ode'] < 1e-8, f"({kls['ode']:.1e})")
        check(f'Theorem 4 (D=G+, corrected) exact, {nm} r/k={rr}', kls['fixed'] < 1e-8, f"({kls['fixed']:.1e})")
        if rr >= 1.0:
            check(f'Proposition 5/6: clipped sampler exact for r >= kappa, {nm} r/k={rr}', kls['naive'] < 1e-8, f"({kls['naive']:.1e})")
        else:
            check(f'Proposition 6: clipped sampler output wrong for r < kappa (Gaussian), {nm} r/k={rr}', kls['naive'] > 1e-6, f"({kls['naive']:.1e})")

# 10. control design: r = 0.9 fixed, omega = sqrt(1 + 2 kappa) puts kappa = 0.9 / ratio, T identical
Ts = []
for rr in [0.6, 0.8, 0.9, 1.0, 1.1, 1.3]:
    k = 0.9 / rr; B = hamiltonian_B(np.sqrt(1 + 2 * k))
    check(f'control ratio {rr}: kappa(omega) = 0.9/ratio', abs(kappa_of(B) - k) < 1e-12)
    Ts.append(Process(B, 0.9).T)
    kn = reverse_kl(B, 0.9, 'naive'); kf = reverse_kl(B, 0.9, 'fixed')
    check(f'control ratio {rr}: corrected exact, clipped exact iff ratio >= 1',
          kf < 1e-8 and ((kn < 1e-8) == (rr >= 1.0)), f'(clipped {kn:.1e})')
check('control: diffusion time T identical for all ratios', np.ptp(Ts) < 1e-12, f'(T={Ts[0]:.4f})')
check('control ratio 0.6 coincides with main sweep ratio 0.6 (omega=2, r=0.9)',
      abs(np.sqrt(1 + 2 * 1.5) - 2.0) < 1e-12)

# 11. (a) likelihood weighting: weight matrix G, reduction to VDM, indefiniteness below kappa
from elbo_gauss import Setup, J_value, KL_value, random_pert
b_s, r_s = -0.7, 0.4                                    # scalar drift B = b I: weight G/sigma^2 = -(log SNR)'
p = Process(b_s * I4, r_s); t, h = 1.1, 1e-6
logsnr = lambda s: np.log(np.exp(2 * b_s * s) / p.sigma(s)**2)
check('scalar drift: G/sigma^2 = -(d/dt) log SNR (VDM weight)',
      np.allclose(p.G(t) / p.sigma(t)**2, -(logsnr(t + h) - logsnr(t - h)) / (2 * h) * I4, rtol=1e-6))
for rr in [0.8, 1.0, 1.3]:
    pp = Process(Bh, rr * 1.5); W = pp.G(0.3) / pp.sigma(0.3)**2
    check(f'W = G/sigma^2 time-independent with eigenvalues 2(r -/+ kappa) (r/k={rr})',
          np.allclose(W, pp.G(2.0) / pp.sigma(2.0)**2) and
          np.allclose(np.sort(np.linalg.eigvalsh(W)), np.sort([2 * (pp.r - 1.5)] * 2 + [2 * (pp.r + 1.5)] * 2)))
rng_a = np.random.default_rng(3)
for rr in [1.0, 1.3]:
    S = Setup(rr); ok = True
    for i in range(3):
        A_, b_ = random_pert(rng_a); ok &= KL_value(S, A_, b_, 0.5) <= J_value(S, A_, b_, 0.5, n=1001)
    check(f'Proposition 7(i): KL(data || Anderson model) <= J with weight G (r/k={rr}, 3 random score errors)', ok)
for rr in [0.6, 0.9]:
    S = Setup(rr); u = np.linalg.eigh(S.p.SymB)[1][:, -1]; A_, b_ = np.outer(u, u), u
    J1, J3 = J_value(S, A_, b_, 1.0, n=1001), J_value(S, A_, b_, 3.0, n=1001)
    check(f'Proposition 7(iii): J < 0 along u and J(3e)=9 J(e), unbounded below (r/k={rr})',
          J1 < 0 and abs(J3 / J1 - 9) < 1e-9 and KL_value(S, A_, b_, 1.0, 'ode') > 0, f'(J={J1:.3g})')

# 12. (b) exact mixture score
from gmm_exact import ExactScore
from gmm_mechanism import mixture
from scipy.special import logsumexp
wq, mq, Sq = mixture(0.8); pq = Process(Bh, 0.9); scq = ExactScore(pq, wq, mq, Sq); tq = 0.7
def logp_mix(z, t):
    M = expm(Bh * t); out = []
    for k in range(len(wq)):
        C = M @ Sq[k] @ M.T + pq.sigma(t)**2 * I4; d = z - M @ mq[k]
        out.append(np.log(wq[k]) - 0.5 * d @ np.linalg.solve(C, d) - 0.5 * np.linalg.slogdet(C)[1])
    return logsumexp(out)
zq = np.random.default_rng(4).standard_normal((5, 4)); hq = 1e-5
fd = np.array([[(logp_mix(z + hq * e, tq) - logp_mix(z - hq * e, tq)) / (2 * hq) for e in I4] for z in zq])
check('mixture score = gradient of mixture log-density (finite differences)', np.abs(fd - scq(zq, tq)).max() < 1e-6)
w1, m1, S1 = mixture(0.0); sc1 = ExactScore(pq, w1, m1, S1)
C1 = expm(Bh * tq) @ S1[0] @ expm(Bh * tq).T + pq.sigma(tq)**2 * I4
check('mixture score reduces to -C^{-1}(z-m) for a single Gaussian', np.allclose(sc1(zq, tq), -(zq - expm(Bh * tq) @ m1[0]) @ np.linalg.inv(C1)))

# 13. Paper, Section 2-3: theory statements (general data via a two-component mixture where possible)
from gmm_exact import ExactScore as _ES, gmm_sample
from gmm_mechanism import mixture as _mix
import json as _json, json
def _mix_logp(w, mu, S, B, p, z, t):
    M = expm(B * t); out = []
    for k in range(len(w)):
        C = M @ S[k] @ M.T + p.sigma(t)**2 * I4; d = z - M @ mu[k]
        out.append(np.log(w[k]) - 0.5 * np.einsum('ni,ij,nj->n', d, np.linalg.inv(C), d)
                   - 0.5 * np.linalg.slogdet(C)[1] - 2 * np.log(2 * np.pi))
    return logsumexp(np.array(out), axis=0)
wq, mq, Sq = _mix(0.8)
# 13a. Lemma 1 (forward equation) for non-Gaussian data, with definite and indefinite G
for rr in [0.6, 1.3]:
    p = Process(Bh, rr * 1.5); t = 0.4; h = 1e-4
    P = lambda z, s: np.exp(_mix_logp(wq, mq, Sq, Bh, p, np.atleast_2d(z), s))[0]
    zs = np.random.default_rng(11).standard_normal((4, 4)) * 0.7; G = p.G(t); err = 0
    for z in zs:
        dpt = (P(z, t + h) - P(z, t - h)) / (2 * h)
        grad = np.array([(P(z + h * e, t) - P(z - h * e, t)) / (2 * h) for e in I4])
        H = np.array([[(P(z + h * a + h * b, t) - P(z + h * a - h * b, t) - P(z - h * a + h * b, t)
                        + P(z - h * a - h * b, t)) / (4 * h * h) for b in I4] for a in I4])
        rhs = -(np.trace(Bh) * P(z, t) + (Bh @ z) @ grad) + 0.5 * np.trace(G @ H)
        err = max(err, abs(dpt - rhs) / (abs(dpt) + abs(rhs) + 1e-12))
    check(f'Lemma 1: forward equation holds for mixture data (r/k={rr}, G {"indefinite" if rr < 1 else "PSD"})',
          err < 1e-4, f'(rel err {err:.1e})')
# 13b. Lemma 2 (entropy balance): Gaussian data exactly, mixture data by Monte Carlo with common random numbers
for B, nm in [(Bh, 'ham'), (Bc, 'cld')]:
    for rr in [0.6, 1.3]:
        p = Process(B, rr * kappa_of(B)); t, h = 0.7, 1e-5
        Ct = lambda s: expm(B * s) @ S0 @ expm(B * s).T + p.sigma(s)**2 * I4
        Sg = lambda s: 0.5 * np.linalg.slogdet(2 * np.pi * np.e * Ct(s))[1]
        lhs = (Sg(t + h) - Sg(t - h)) / (2 * h)
        rhs = np.trace(B) + 0.5 * np.trace(p.G(t) @ np.linalg.inv(Ct(t)))
        check(f'Lemma 2: dS/dt = tr B + tr(GF)/2, Gaussian data ({nm}, r/k={rr})', abs(lhs - rhs) < 1e-6 * max(1, abs(rhs)))
rng_m = np.random.default_rng(12)
x0m = gmm_sample(wq, mq, Sq, 200000, rng_m); epm = rng_m.standard_normal(x0m.shape)
for rr in [0.6, 1.3]:
    p = Process(Bh, rr * 1.5); t, h = 0.3, 1e-3
    Smc = lambda s: -_mix_logp(wq, mq, Sq, Bh, p, x0m @ expm(Bh * s).T + p.sigma(s) * epm, s).mean()
    lhs = (Smc(t + h) - Smc(t - h)) / (2 * h)
    yt = x0m @ expm(Bh * t).T + p.sigma(t) * epm
    sc = _ES(p, wq, mq, Sq)(yt, t); Fm = sc.T @ sc / len(sc)
    rhs = np.trace(Bh) + 0.5 * np.trace(p.G(t) @ Fm)
    check(f'Lemma 2: entropy balance for mixture data (Monte Carlo, r/k={rr})', abs(lhs - rhs) < 0.03 * abs(rhs) + 2e-3,
          f'({lhs:.4f} vs {rhs:.4f})')
# 13c. Lemma 2, "only if": data with Fisher matrix concentrating on u give dS/dt < tr B when r < kappa
p = Process(Bh, 0.6 * 1.5); t = 0.5; u = np.linalg.eigh(p.SymB)[1][:, -1]
F = np.linalg.inv(1e6 * (I4 - np.outer(u, u)) + p.sigma(t)**2 * I4)
lim = u @ p.G(t) @ u / p.sigma(t)**2
check('Lemma 2 (converse): tr(GF) -> u^T G u / sigma^2 < 0 for the constructed data',
      abs(np.trace(p.G(t) @ F) - lim) < 1e-5 * abs(lim) and lim < 0, f'({lim:.3f})')
# 13d. rewinding: dC~/dt = M^-1 G M^-T and S(t) = h(X0 + N~) + t tr B
for B, nm in [(Bh, 'ham'), (Bc, 'cld')]:
    p = Process(B, 0.6 * kappa_of(B)); t, h = 0.8, 1e-6
    Ctl = lambda s: np.linalg.inv(expm(B * s)) @ (p.sigma(s)**2 * I4) @ np.linalg.inv(expm(B * s)).T
    Mi = np.linalg.inv(expm(B * t))
    ok1 = np.allclose((Ctl(t + h) - Ctl(t - h)) / (2 * h), Mi @ p.G(t) @ Mi.T, rtol=1e-5, atol=1e-8)
    Sa = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * (expm(B * t) @ S0 @ expm(B * t).T + p.sigma(t)**2 * I4))[1]
    Sb = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * (S0 + Ctl(t)))[1] + t * np.trace(B)
    check(f'rewinding identities ({nm})', ok1 and abs(Sa - Sb) < 1e-9)
# 13e. Theorem 1 (a)=>(b): the SDE with L = G^{1/2} reproduces C(t)
for B, nm in [(Bh, 'ham'), (Bc, 'cld')]:
    p = Process(B, 1.3 * kappa_of(B)); C0 = p.sigma(0)**2 * I4
    sol = solve_ivp(lambda t, q: (B @ q.reshape(4, 4) + q.reshape(4, 4) @ B.T + p.G(t)).ravel(), [0, 2.0],
                    np.zeros(16), rtol=1e-10, atol=1e-14)
    Q = sol.y[:, -1].reshape(4, 4); M2 = expm(B * 2.0)
    check(f'Theorem 3 (a=>b): SDE noise covariance + M C(0) M^T = C(t) ({nm})',
          np.allclose(M2 @ C0 @ M2.T + Q, p.sigma(2.0)**2 * I4, rtol=1e-6))
# 13f. Proposition 3: tr(G_- F) > 0 for mixture data iff r < kappa
for rr in [0.6, 0.9, 1.0, 1.3]:
    p = Process(Bh, rr * 1.5); t = 0.3
    yt = x0m[:50000] @ expm(Bh * t).T + p.sigma(t) * epm[:50000]
    sc = _ES(p, wq, mq, Sq)(yt, t); Fm = sc.T @ sc / len(sc)
    v = np.trace((p.Gplus(t) - p.G(t)) @ Fm)
    check(f'Remark 8: entropy deficit tr(G_- F) {">" if rr < 1 else "="} 0 (r/k={rr})', (v > 1e-6) if rr < 1 else abs(v) < 1e-12, f'({v:.2e})')
# 13g. Proposition 4(ii): DSM identity with indefinite W (closed form, Gaussian data, linear eps-models)
p = Process(Bh, 0.6 * 1.5); t = 0.9; Mt = expm(Bh * t); sg = p.sigma(t); W = p.G(t) / sg**2
m0v = np.array([0.4, 0.1, -0.2, 0.0]); C = Mt @ S0 @ Mt.T + sg**2 * I4
Sw = np.block([[S0, np.zeros((4, 4))], [np.zeros((4, 4)), I4]]); Kz = np.hstack([Mt, sg * I4])
def quad(a, K): return a @ W @ a + np.trace(K.T @ W @ K @ Sw)
diffs = []
for sd in range(3):
    r_ = np.random.default_rng(sd); A = r_.standard_normal((4, 4)); c = r_.standard_normal(4)
    a = A @ Mt @ m0v + c
    dsm = quad(a, A @ Kz - np.hstack([np.zeros((4, 4)), I4]))
    true = quad(a, A @ Kz - sg * np.linalg.inv(C) @ Kz)
    diffs.append(dsm - true)
check('Proposition 7(ii): J_DSM - J is independent of the model, also for indefinite W', np.ptp(diffs) < 1e-9 * max(1, abs(diffs[0])),
      f'(W eigs {np.round(np.linalg.eigvalsh(W), 2)})')

# 14. Every number quoted in the paper, read back from the result files
R = os.path.join(os.path.dirname(__file__), '..', 'results')
D2 = _json.load(open(os.path.join(R, 'summary_day2.json'))); EL = _json.load(open(os.path.join(R, 'summary_elbo.json')))
GM = _json.load(open(os.path.join(R, 'summary_gmm.json'))); EG = _json.load(open(os.path.join(R, 'elbo_gauss.json')))
def sub(d, rr): return [r for r in D2[d]['rows'] if r['ratio'] == rr]
def ms(vals): return np.mean(vals), np.std(vals)
rd = lambda x, n: round(float(x), n)
ok = True
for rr, m_, s_ in [(0.6, 0.0273, 0.0011), (0.8, 0.0064, 0.0004), (0.9, 0.0014, 0.0001)]:
    m, s = ms([r['paired_KL'] for r in sub('ham', rr)]); ok &= rd(m, 4) == m_ and rd(s, 4) == s_
for rr, m_, s_ in [(0.8, 0.0074, 0.0003), (0.9, 0.0019, 0.0001)]:
    m, s = ms([r['paired_KL'] for r in sub('ctrl', rr)]); ok &= rd(m, 4) == m_ and rd(s, 4) == s_
check('paper: paired KL mean +/- s.d. (main and control)', ok)
th = D2['ham']['theory']
check('paper: Gaussian-moment predictions 0.040, 0.009, 0.0023 and ratio about 70%',
      rd(th['0.6']['KL'], 3) == 0.040 and rd(th['0.8']['KL'], 3) == 0.009 and rd(th['0.9']['KL'], 4) == 0.0023
      and 0.65 < np.mean([r['paired_KL'] for r in sub('ham', 0.6)]) / th['0.6']['KL'] < 0.75)
check('paper: variance shrinks in 5/5 seeds and bootstrap CI excludes 0 at every r < kappa (both sweeps)',
      all(r['log_vr'] < 0 and r['log_vr_ci'][1] < 0 for d in ['ham', 'ctrl'] for rr in [0.6, 0.8, 0.9] for r in sub(d, rr)))
check('paper: paired quantities vanish for r >= kappa',
      all(abs(r['paired_KL']) < 1e-12 for d in ['ham', 'ctrl'] for rr in [1.0, 1.1, 1.3] for r in sub(d, rr)))
wins = {(d, rr): sum(r['mmd4']['naive'] > r['mmd4']['fixed'] for r in sub(d, rr)) for d in ['ham', 'ctrl'] for rr in [0.6, 0.8, 0.9]}
check('paper: corrected beats clipped in MMD 5/5 (0.6, 0.8) and 4/5 (0.9) in both sweeps',
      all(wins[(d, 0.6)] == 5 and wins[(d, 0.8)] == 5 and wins[(d, 0.9)] == 4 for d in ['ham', 'ctrl']))
ode_c = [np.mean([r['mmd4']['ode'] for r in sub('ctrl', rr)]) for rr in [0.8, 0.9, 1.0, 1.1, 1.3]]
check('paper: control ODE MMD^2 between 0.0007 and 0.0011 for r/k >= 0.8', min(ode_c) >= 0.0007 and max(ode_c) <= 0.0011,
      f'({min(ode_c):.5f}-{max(ode_c):.5f})')
check('paper: ELBO exact Gaussian, 240 cases KL <= J, max ratio 0.47', len(EG['above']) == 240
      and all(x['KL'] <= x['J'] for x in EG['above']) and rd(max(x['KL'] / x['J'] for x in EG['above']), 2) == 0.47)
neg = [x for x in EG['below'] if x['ratio'] == 0.6 and x['eps'] == 3.0 and x['kind'] == 'negative direction'][0]
check('paper: J = -47.1 and KL = 6.66 at r/k = 0.6, eps = 3; ODE control 58/80',
      rd(neg['J'], 1) == -47.1 and rd(neg['KL'], 2) == 6.66
      and sum(x['KL'] > x['J'] for x in EG['ode_control_above']) == 58 and len(EG['ode_control_above']) == 80)
tr_ = EL['train']
check('paper: likelihood-weighted training diverges iff r < kappa (loss < 0 by step 100, final -1e20..-1e22, error > 1e20)',
      all(all(s is not None and s <= 100 for s in tr_[k]['first_step_loss_negative'])
          and all(-1e22 < v < -1e20 for v in tr_[k]['final_weighted_loss']) and min(tr_[k]['final_mse']) > 1e20 for k in ['0.8', '0.9'])
      and all(all(s is None for s in tr_[k]['first_step_loss_negative']) and max(tr_[k]['final_mse']) < 5 for k in ['1.0', '1.1', '1.3']))
check('paper: at r = kappa the u-direction error stays near 1.0 (vs about 0.5 above)',
      0.9 < np.median(tr_['1.0']['final_mse_u']) < 1.2 and all(0.4 < np.median(tr_[k]['final_mse_u']) < 0.6 for k in ['1.1', '1.3']))
tab = {('both', 0.6): ('ham', 0.6), ('main', 0.8): ('ham', 0.8), ('main', 0.9): ('ham', 0.9),
       ('control', 0.8): ('ctrl', 0.8), ('control', 0.9): ('ctrl', 0.9)}
ST = _json.load(open(os.path.join(R, 'summary_steps.json')))
quoted = {('both', 0.6): ('0.0397', '0.0231', '0.0005', '0.0273', '0.0011', '-0.1947', '-0.0625', '0.0035', '-0.0553', '0.0067'),
          ('main', 0.8): ('0.0093', '0.0050', '0.0003', '0.0064', '0.0004', '-0.0913', '-0.0239', '0.0019', '-0.0242', '0.0015'),
          ('main', 0.9): ('0.0023', '0.0011', '0.0001', '0.0014', '0.0001', '-0.0511', '-0.0163', '0.0009', '-0.0151', '0.0007'),
          ('control', 0.8): ('0.0104', '0.0062', '0.0001', '0.0074', '0.0003', '-0.1027', '-0.0366', '0.0015', '-0.0333', '0.0015'),
          ('control', 0.9): ('0.0026', '0.0016', '0.00003', '0.0019', '0.0001', '-0.0512', '-0.0187', '0.0017', '-0.0161', '0.0013')}
ok = True; rel = []; klr = []
f4 = lambda x: '%.4f' % x
def sdfmt(x):
    s4 = f4(x)
    return ('%.5f' % x).rstrip('0') if s4 == '0.0000' else s4
for key, (d, rr) in tab.items():
    row = [r for r in GM['rows'] if r['dyn'] == d and r['ratio'] == rr][0]
    got = (f4(row['theory_KL']), f4(np.mean(row['exact_KL'])), sdfmt(np.std(row['exact_KL'])),
           f4(np.mean(row['net_KL'])), f4(np.std(row['net_KL'])), f4(row['theory_log_vr']),
           f4(np.mean(row['exact_log_vr'])), f4(np.std(row['exact_log_vr'])), f4(np.mean(row['net_log_vr'])), f4(np.std(row['net_log_vr'])))
    if got != quoted[key]: print('   table 2 mismatch', key, got)
    ok &= got == quoted[key]
    rel.append(abs(np.mean(row['net_log_vr']) / np.mean(row['exact_log_vr']) - 1))
    klr.append(np.mean(row['net_KL']) / np.mean(row['exact_KL']) - 1)
check('paper: Table 2 entries', ok)
check('paper: variance shrinkage network vs exact differs by between 1% and 14%; network KL 18%-27% above exact',
      round(100 * min(rel)) == 1 and round(100 * max(rel)) == 14 and round(100 * min(klr)) == 18 and round(100 * max(klr)) == 27,
      f'({np.round(rel, 3)}, {np.round(klr, 3)})')
# Table 3 (three noise seeds, coupled paths)
q3 = {('both', 0.6): ('0.0229', '0.0001', '0.0224', '0.0001', '-1.9', '0.1', '-0.0642', '0.0042', '-0.0640', '0.0054', '-0.4', '2.0'),
      ('main', 0.8): ('0.0055', '0.0005', '0.0054', '0.0004', '-1.2', '0.8', '-0.0258', '0.0064', '-0.0255', '0.0044', '+1.2', '9.4'),
      ('main', 0.9): ('0.0011', '0.00004', '0.0012', '0.00005', '+2.0', '2.6', '-0.0166', '0.0002', '-0.0164', '0.0009', '-1.1', '4.3'),
      ('control', 0.8): ('0.0060', '0.0003', '0.0061', '0.0001', '+1.2', '3.5', '-0.0338', '0.0018', '-0.0345', '0.0016', '+2.2', '3.4'),
      ('control', 0.9): ('0.0015', '0.0001', '0.0015', '0.00004', '-0.7', '2.0', '-0.0163', '0.0025', '-0.0176', '0.0016', '+9.1', '9.8')}
ok = True
for key, (d, rr) in tab.items():
    x = [s for s in ST if s['dyn'] == d and s['ratio'] == rr][0]
    assert len(json.load(open(os.path.join(R, f'gmm_steps_{d}_r{rr}_n2.json')))['K16000']) > 0
    k, l = x['paired_KL'], x['log_vr']
    got = (f4(k['K4000']['mean']), sdfmt(k['K4000']['sd']), f4(k['inf']['mean']), sdfmt(k['inf']['sd']),
           '%+.1f' % (100 * k['rel']['mean']), '%.1f' % (100 * k['rel']['sd']),
           f4(l['K4000']['mean']), sdfmt(l['K4000']['sd']), f4(l['inf']['mean']), sdfmt(l['inf']['sd']),
           '%+.1f' % (100 * l['rel']['mean']), '%.1f' % (100 * l['rel']['sd']))
    if got != q3[key]: print('   table 3 mismatch', key, got)
    ok &= got == q3[key]
check('paper: Table 3 entries (three seeds, per-seed extrapolation)', ok)
klm = [x['paired_KL']['rel']['mean'] for x in ST]; kls = [x['paired_KL']['rel']['sd'] for x in ST]
lvm = [x['log_vr']['rel']['mean'] for x in ST]; lvs = [x['log_vr']['rel']['sd'] for x in ST]
c06 = [x for x in ST if x['dyn'] == 'ham' and x['ratio'] == 0.6][0]['log_vr_to_truth_fixed']
check('paper: step text (KL -1.9%..+2.0%, spread <= 3.5%; log-var -1.1%..+9.1%, spread 9.8%; monotone 3/15 and 7/15; 0.032, 0.018, 0.010)',
      ('%+.1f' % (100 * min(klm)), '%+.1f' % (100 * max(klm)), '%.1f' % (100 * max(kls)),
       '%+.1f' % (100 * min(lvm)), '%+.1f' % (100 * max(lvm)), '%.1f' % (100 * max(lvs))) == ('-1.9', '+2.0', '3.5', '-1.1', '+9.1', '9.8')
      and sum(x['paired_KL']['n_monotone'] for x in ST) == 3 and sum(x['log_vr']['n_monotone'] for x in ST) == 7
      and tuple('%.3f' % c06[K]['mean'] for K in ['K4000', 'K8000', 'K16000']) == ('0.032', '0.018', '0.010'))
check('paper: log-variance step changes not distinguishable from zero (|mean| <= spread for every configuration)',
      all(abs(m) <= s for m, s in zip(lvm, lvs)))
mech = GM['mechanism']; mg = GM['mechanism_gauss_theory']
x1 = {a: np.mean([m['log_var_ratio_x1'] for m in mech if m['a'] == a]) for a in sorted(set(m['a'] for m in mech))}
x2 = [m['log_var_ratio_x2'] for m in mech]
check('paper: mechanism x2 at -0.20 (prediction -0.20); x1 from -0.197 to -0.016',
      all(rd(v, 2) == -0.20 for v in x2) and rd(mg['x2'], 2) == -0.20 and rd(x1[0.0], 3) == -0.197 and rd(x1[0.98], 3) == -0.016)
from gmm_exact import fit_gmm
from psdt import two_moons, mmd2
wg, mg_, Sg_ = fit_gmm()
mmd_g = mmd2(gmm_sample(wg, mg_, Sg_, 4000, np.random.default_rng(5)), two_moons(4000, np.random.default_rng(99)))
check('paper: GMM fit MMD^2 to two moons = 2.5e-4', rd(mmd_g * 1e4, 1) == 2.5, f'({mmd_g:.2e})')
# CLD: Gaussian-moment prediction of the paired KL at r/kappa = 0.6 is about 1e-3 (vs 4e-2 for the Hamiltonian flow)
from scipy.integrate import solve_ivp as _ivp
def _theory_kl(B, r):
    pp = Process(B, r); ref = two_moons(400000, np.random.default_rng(1)); m0_, S0_ = ref.mean(0), np.cov(ref.T)
    def mom(t):
        M = expm(B * t); return M @ m0_, M @ S0_ @ M.T + pp.sigma(t)**2 * I4
    def rhs(tau, y):
        t = pp.T - tau; mu = y[:4]; Sg = y[4:].reshape(4, 4); m, C = mom(t); P = np.linalg.inv(C)
        Gp = pp.Gplus(t); A = -B - Gp @ P
        return np.concatenate([A @ mu + Gp @ P @ m, (A @ Sg + Sg @ A.T + Gp).ravel()])
    m, C = mom(pp.T); y = _ivp(rhs, [0, pp.T], np.concatenate([m, C.ravel()]), rtol=1e-9, atol=1e-12, method='DOP853').y[:, -1]
    Sg = y[4:].reshape(4, 4); mt, Ct = mom(0); iC = np.linalg.inv(Ct); dv = mt - y[:4]; Sg = (Sg + Sg.T) / 2
    return 0.5 * (np.trace(iC @ Sg) + dv @ iC @ dv - 4 + np.log(np.linalg.det(Ct) / np.linalg.det(Sg)))
kc = _theory_kl(Bc, 0.6 * 0.5)
check('paper: CLD Gaussian-moment paired KL at r/k = 0.6 is about 1e-3', 0.8e-3 < kc < 1.4e-3, f'({kc:.2e})')

# 15. Revision: Proposition 5 (intermediate marginals, non-Gaussian data), Proposition 6 (Gaussian output),
#     forward score-corrected process (remark after Theorem 3), Lemma 2 split, Proposition 7(iii) sign switch
# 15a. Prop 5: for mixture data, div(G_- grad p_t) is not identically zero when r < kappa (p_t cannot solve the clipped FP)
for rr in [0.6, 0.9]:
    p = Process(Bh, rr * 1.5); t = 0.4; h = 1e-4
    P = lambda z, s: np.exp(_mix_logp(wq, mq, Sq, Bh, p, np.atleast_2d(z), s))[0]
    Gm_ = p.Gplus(t) - p.G(t); vals = []
    for z in np.random.default_rng(21).standard_normal((6, 4)) * 0.7:
        H = np.array([[(P(z + h * a + h * b, t) - P(z + h * a - h * b, t) - P(z - h * a + h * b, t)
                        + P(z - h * a - h * b, t)) / (4 * h * h) for b in I4] for a in I4])
        vals.append(abs(np.trace(Gm_ @ H)) / P(z, t))
    check(f'Proposition 5: div(G_- grad p_t) != 0 for mixture data (r/k={rr})', max(vals) > 1e-3, f'(max rel {max(vals):.2e})')
# 15b. Prop 6: Gaussian data, clipped output: mean exact, Delta <= 0, tr Delta < 0 iff r < kappa
def clipped_gauss_output(B, r, m0_, S0_):
    pp = Process(B, r)
    def mom(t):
        M = expm(B * t); return M @ m0_, M @ S0_ @ M.T + pp.sigma(t)**2 * I4
    def rhs(tau, y):
        t = pp.T - tau; mu = y[:4]; Sg = y[4:].reshape(4, 4); m, C = mom(t); P_ = np.linalg.inv(C)
        Gp = pp.Gplus(t); A = -B - Gp @ P_
        return np.concatenate([A @ mu + Gp @ P_ @ m, (A @ Sg + Sg @ A.T + Gp).ravel()])
    m, C = mom(pp.T)
    y = solve_ivp(rhs, [0, pp.T], np.concatenate([m, C.ravel()]), rtol=1e-10, atol=1e-12, method='DOP853').y[:, -1]
    mt, Ct = mom(0); Sg = y[4:].reshape(4, 4); return y[:4] - mt, (Sg + Sg.T) / 2 - Ct
for B, nm in [(Bh, 'ham'), (Bc, 'cld')]:
    for rr in [0.6, 0.9, 1.0, 1.3]:
        dm, D_ = clipped_gauss_output(B, rr * kappa_of(B), m0, S0); ev = np.linalg.eigvalsh(D_)
        ok = np.abs(dm).max() < 1e-7 and ev.max() < 1e-8 and ((np.trace(D_) < -1e-6) if rr < 1 else abs(np.trace(D_)) < 1e-8)
        check(f'Proposition 6: clipped output mean exact, Delta <= 0, tr Delta {"< 0" if rr < 1 else "= 0"} ({nm}, r/k={rr})', ok,
              f'(tr Delta {np.trace(D_):.2e})')
# 15c. remark after Theorem 3: forward score-corrected process dZ = [BZ - (G-D)s/2]dt + D^(1/2) dW has marginals p_t for any r
for B, nm in [(Bh, 'ham'), (Bc, 'cld')]:
    pp = Process(B, 0.6 * kappa_of(B))
    def rhs(t, y):
        Sg = y.reshape(4, 4); C = expm(B * t) @ S0 @ expm(B * t).T + pp.sigma(t)**2 * I4; P_ = np.linalg.inv(C)
        D = pp.Gplus(t); A = B + 0.5 * (pp.G(t) - D) @ P_
        return (A @ Sg + Sg @ A.T + D).ravel()
    C0 = S0 + pp.sigma(0)**2 * I4
    Sg = solve_ivp(rhs, [0, 2.0], C0.ravel(), rtol=1e-10, atol=1e-12, method='DOP853').y[:, -1].reshape(4, 4)
    check(f'Remark (Section 2.4): forward score-corrected process reproduces p_t below kappa ({nm})',
          np.allclose(Sg, expm(B * 2.0) @ S0 @ expm(B * 2.0).T + pp.sigma(2.0)**2 * I4, rtol=1e-6))
# 15d. Fig. 2 left: sign of J along u switches exactly at kappa
ud = _json.load(open(os.path.join(R, 'elbo_gauss.json')))['u_direction']
check('Proposition 7 / Fig. 2: J along u is > 0 for r > kappa, = 0 at r = kappa, < 0 below',
      all((x['J'] > 0) if x['ratio'] > 1 else (abs(x['J']) < 1e-12 if x['ratio'] == 1 else x['J'] < 0) for x in ud))

try:
    deterministic_report = run_deterministic_foundations(
        json.loads(DEFAULT_SPEC.read_text()))
except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
    check('deterministic foundations: specification and transformation checks', False,
          f'({error})')
else:
    for case in deterministic_report['cases'] + deterministic_report['negative_controls']:
        check(f"deterministic foundations: {case['name']}",
              case['outcome'] == 'supported')

n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
