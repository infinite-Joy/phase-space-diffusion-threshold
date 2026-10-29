"""Phase-space diffusion with an isotropic marginal schedule C(t) = sigma(t)^2 I.

Forward marginal:  Z_t = M(t) Z_0 + sigma(t) * eps,   M(t) = expm(B t),   sigma(t) = s0 * exp(r t).
Required diffusion matrix:  G(t) = dC/dt - B C - C B^T = 2 sigma^2 (r I - Sym B).
G >= 0  <=>  r >= kappa = lambda_max(Sym B).

Reverse samplers (all written in forward time t, integrated from T down to 0):
  ode   : dz/dt = Bz - 1/2 G s                                    (probability flow)
  naive : Anderson reverse SDE with G replaced by G_+ (negative eigenvalues clipped)
  fixed : drift Bz - 1/2 (G + G_+) s, noise sqrt(G_+)              (exact for any r)
where s = grad log p_t.  Pure numpy (no torch).
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import expit

# ---------------------------------------------------------------- dynamics
def hamiltonian_B(omega, d=2):
    b = np.array([[0.0, 1.0], [-omega**2, 0.0]])
    return np.kron(np.eye(d), b)          # z = (x1, v1, x2, v2)

def cld_B(beta=1.0, M=0.25, Gamma=1.0, d=2):
    b = beta * np.array([[0.0, 1.0 / M], [-1.0, -Gamma / M]])
    return np.kron(np.eye(d), b)

def kappa_of(B):
    return np.linalg.eigvalsh((B + B.T) / 2).max()

class Process:
    def __init__(self, B, r, s0=0.01, sT=40.0):
        self.B, self.r, self.s0, self.sT = B, r, s0, sT
        self.n = B.shape[0]
        self.T = np.log(sT / s0) / r
        self.SymB = (B + B.T) / 2
        w, V = np.linalg.eigh(self.SymB)
        self.w, self.V = w, V                   # G = 2 sigma^2 V diag(r - w) V^T
    def sigma(self, t):
        return self.s0 * np.exp(self.r * t)
    def M(self, t):
        return expm(self.B * t)
    def G(self, t):
        return 2 * self.sigma(t)**2 * (self.V @ np.diag(self.r - self.w) @ self.V.T)
    def Gplus(self, t):
        return 2 * self.sigma(t)**2 * (self.V @ np.diag(np.clip(self.r - self.w, 0, None)) @ self.V.T)
    def sqrtGplus(self, t):
        return np.sqrt(2) * self.sigma(t) * (self.V @ np.diag(np.sqrt(np.clip(self.r - self.w, 0, None))) @ self.V.T)

# ---------------------------------------------------------------- data
def two_moons(n, rng, noise=0.05, vstd=0.25):
    t = rng.uniform(0, np.pi, n)
    lab = rng.integers(0, 2, n)
    x = np.where(lab[:, None] == 0,
                 np.stack([np.cos(t), np.sin(t)], 1),
                 np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1))
    x = x + noise * rng.standard_normal((n, 2))
    x = (x - np.array([0.5, 0.25])) / 0.7          # roughly unit scale
    v = vstd * rng.standard_normal((n, 2))
    z = np.empty((n, 4)); z[:, 0::2] = x; z[:, 1::2] = v
    return z

# ---------------------------------------------------------------- MLP (manual backprop, Adam)
def silu(a):  return a * expit(a)
def dsilu(a):
    sg = expit(a); return sg * (1 + a * (1 - sg))

class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= 0.1
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.k = 0
    def forward(self, h):
        self.cache = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = h @ W + b
            self.cache.append((h, a))
            h = silu(a) if i < len(self.W) - 1 else a
        return h
    def backward(self, g):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            h, a = self.cache[i]
            if i < len(self.W) - 1: g = g * dsilu(a)
            gW[i] = h.T @ g; gb[i] = g.sum(0)
            g = g @ self.W[i].T
        return gW + gb
    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.k += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1**self.k)) / (np.sqrt(v / (1 - b2**self.k)) + eps)

# ---------------------------------------------------------------- score model: eps-prediction
class ScoreModel:
    """s(z,t) = -eps_theta(z,t)/sigma(t).  Inputs normalised by sqrt(sigma^2 + a^2)."""
    def __init__(self, proc, rng, width=128, depth=3, a=2.0):
        self.p, self.a = proc, a
        sizes = [proc.n + 5] + [width] * depth + [proc.n]
        self.net = MLP(sizes, rng)
        self.Mcache = {}
    def features(self, z, t):
        p = self.p
        sig = p.sigma(t)
        u = (np.log(sig) - np.log(p.s0)) / np.log(p.sT / p.s0)       # in [0,1]
        th = t / p.T * 2 * np.pi
        tf = np.stack([u, np.sin(th), np.cos(th), np.sin(2 * th), np.cos(2 * th)], -1)
        tf = np.broadcast_to(tf, (z.shape[0], 5)) if tf.ndim == 1 else tf
        return np.concatenate([z / np.sqrt(sig**2 + self.a**2)[..., None], tf], 1)
    def eps(self, z, t):
        t = np.broadcast_to(np.asarray(t, float), (z.shape[0],))
        return self.net.forward(self.features(z, t))
    def score(self, z, t):
        return -self.eps(z, t) / self.p.sigma(t)

def batch_M(proc, t):
    """expm(B t) for a vector of times: exact on a fine grid (works for non-diagonalisable B,
    e.g. critically damped CLD), linear interpolation in between (relative error ~1e-7)."""
    if not hasattr(proc, "_Mgrid"):
        proc._tg = np.linspace(0, proc.T, 20001)
        proc._Mgrid = expm(proc._tg[:, None, None] * proc.B[None])
    x = t / proc.T * (len(proc._tg) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(proc._tg) - 2)
    w = (x - i)[:, None, None]
    return (1 - w) * proc._Mgrid[i] + w * proc._Mgrid[i + 1]

def train(model, data, rng, steps=12000, bs=512, lr=2e-3, log=None):
    p = model.p
    for k in range(steps):
        lr_k = lr * 0.5 * (1 + np.cos(np.pi * k / steps))           # cosine decay
        z0 = data[rng.integers(0, len(data), bs)]
        t = rng.uniform(0, p.T, bs)
        Mt = batch_M(p, t)
        eps = rng.standard_normal((bs, p.n))
        zt = np.einsum('nij,nj->ni', Mt, z0) + p.sigma(t)[:, None] * eps
        out = model.net.forward(model.features(zt, t))
        diff = out - eps
        loss = (diff**2).sum(1).mean()
        grads = model.net.backward(2 * diff / bs)
        model.net.adam(grads, lr_k)
        if log is not None and (k % 2000 == 0 or k == steps - 1):
            log(k, loss)
    return model

# ---------------------------------------------------------------- prior and samplers
def prior_sample(proc, data, n, rng):
    """Gaussian prior with the exact first two moments of p_T (data moments propagated)."""
    MT = proc.M(proc.T)
    m = MT @ data.mean(0)
    C = MT @ np.cov(data.T) @ MT.T + proc.sigma(proc.T)**2 * np.eye(proc.n)
    return rng.multivariate_normal(m, C, n)

def sample(score_fn, proc, z, mode, rng, K=1000):
    """Integrate from t=T to t=0 with K uniform steps.  score_fn(z, t) -> grad log p_t."""
    ts = np.linspace(proc.T, 0, K + 1)
    B = proc.B
    def drift(z, t):
        s = score_fn(z, t)
        G = proc.G(t)
        if mode == 'ode':   return z @ B.T - 0.5 * s @ G.T
        Gp = proc.Gplus(t)
        if mode == 'naive': return z @ B.T - s @ Gp.T
        if mode == 'fixed': return z @ B.T - 0.5 * s @ (G + Gp).T
        raise ValueError(mode)
    for k in range(K):
        t, t2 = ts[k], ts[k + 1]; h = t - t2
        if mode == 'ode':                                   # Heun
            f1 = drift(z, t); zp = z - h * f1
            f2 = drift(zp, t2); z = z - 0.5 * h * (f1 + f2)
        else:                                               # Euler-Maruyama
            z = z - h * drift(z, t) + np.sqrt(h) * rng.standard_normal(z.shape) @ proc.sqrtGplus(t).T
    return z

# ---------------------------------------------------------------- metrics
def sliced_w2(a, b, rng, nproj=500):
    d = a.shape[1]
    P = rng.standard_normal((d, nproj)); P /= np.linalg.norm(P, axis=0)
    n = min(len(a), len(b))
    pa = np.sort(a[:n] @ P, 0); pb = np.sort(b[:n] @ P, 0)
    return np.sqrt(((pa - pb)**2).mean())

def mmd2(a, b, scales=(0.1, 0.2, 0.5, 1.0)):
    def k(x, y):
        d2 = (x**2).sum(1)[:, None] + (y**2).sum(1)[None] - 2 * x @ y.T
        return sum(np.exp(-d2 / (2 * s * s)) for s in scales)
    n, m = len(a), len(b)
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    return ((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
            - 2 * kab.mean())
