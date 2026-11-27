"""Probability of detection (MIL-HDBK-1823A): â-versus-a and hit/miss analyses, confidence
bounds, a₉₀ and a₉₀/₉₅, noise / false-call analysis, ROC curves and model-assisted POD.

â vs a: ln â = β₀ + β₁ ln a + ε, ε ~ N(0, τ²), with left censoring below the recording
threshold and right censoring at saturation (Tobit likelihood). With decision threshold â_th:
    POD(a) = Φ((ln a − μ)/σ),  μ = (ln â_th − β₀)/β₁,  σ = τ/β₁ ,
covariance of (μ, σ) by the delta method from the observed information of (β₀, β₁, τ), and the
lower one-sided 95 % bound a_p,95 = exp(μ + z_p σ + 1.645 √(V_μμ + z_p² V_σσ + 2 z_p V_μσ)).

Hit/miss: POD(a) = F(β₀ + β₁ ln a) with logit / probit / cloglog / loglog links, fitted by
maximum likelihood; the lower bound is the likelihood-ratio bound — the minimum of POD(a) over
the region 2(ℓ̂ − ℓ(β)) ≤ χ²₁(0.90) = 2.706 (one-sided 95 %).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

Z90 = norm.ppf(0.90)
Z95 = norm.ppf(0.95)
LR_CRIT = 2.705543454095404           # χ²₁(0.90)


# =========================================================================== â vs a
@dataclass
class AhatResult:
    beta0: float
    beta1: float
    tau: float
    cov: np.ndarray            # (β₀, β₁, τ)
    mu: float
    sigma: float
    cov_ms: np.ndarray         # (μ, σ)
    a50: float
    a90: float
    a90_95: float
    threshold: float
    log_a: bool
    n_censored_low: int
    n_censored_high: int
    loglik: float

    def pod(self, a):
        x = np.log(a) if self.log_a else np.asarray(a, float)
        return norm.cdf((x - self.mu) / self.sigma)

    def pod_lower(self, a):
        """Lower 95 % confidence bound curve: inverse of a_p,95 over p."""
        p = np.linspace(0.001, 0.999, 999)
        z = norm.ppf(p)
        V = self.cov_ms
        x95 = self.mu + z * self.sigma + Z95 * np.sqrt(np.maximum(
            V[0, 0] + z ** 2 * V[1, 1] + 2 * z * V[0, 1], 0))
        x = np.log(a) if self.log_a else np.asarray(a, float)
        order = np.argsort(x95)
        return np.interp(x, x95[order], p[order], left=0.0, right=1.0)


def _tobit_nll(theta, x, y, lo, hi):
    b0, b1, lt = theta
    tau = np.exp(lt)
    m = b0 + b1 * x
    ll = np.zeros_like(y)
    unc = (y > lo) & (y < hi)
    ll[unc] = norm.logpdf(y[unc], m[unc], tau)
    left = y <= lo
    ll[left] = norm.logcdf((lo - m[left]) / tau)
    right = y >= hi
    ll[right] = norm.logsf((hi - m[right]) / tau)
    return -np.sum(ll)


def _hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n); e_i[i] = h * max(1, abs(x[i]))
            e_j = np.zeros(n); e_j[j] = h * max(1, abs(x[j]))
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) \
                / (4 * e_i[i] * e_j[j])
            H[j, i] = H[i, j]
    return H


def ahat_vs_a(a, ahat, threshold, floor=None, saturation=None, log_a=True, log_ahat=True):
    """Fit the â-vs-a model. `floor`/`saturation` are the censoring limits in â units."""
    a = np.asarray(a, float); ahat = np.asarray(ahat, float)
    x = np.log(a) if log_a else a
    tr = np.log if log_ahat else (lambda v: v)
    y = tr(np.maximum(ahat, 1e-300))
    lo = tr(floor) if floor is not None else -np.inf
    hi = tr(saturation) if saturation is not None else np.inf
    y = np.clip(y, lo, hi)
    unc = (y > lo) & (y < hi)
    A = np.vstack([np.ones(unc.sum()), x[unc]]).T
    b, *_ = np.linalg.lstsq(A, y[unc], rcond=None)
    s0 = np.std(y[unc] - A @ b) + 1e-6
    r = minimize(_tobit_nll, [b[0], b[1], np.log(s0)], args=(x, y, lo, hi), method="Nelder-Mead",
                 options=dict(xatol=1e-9, fatol=1e-10, maxiter=20000))
    r = minimize(_tobit_nll, r.x, args=(x, y, lo, hi), method="BFGS")
    b0, b1, lt = r.x
    tau = np.exp(lt)
    # covariance in (β₀, β₁, τ)
    f = lambda th: _tobit_nll(np.array([th[0], th[1], np.log(max(th[2], 1e-12))]), x, y, lo, hi)
    H = _hessian(f, np.array([b0, b1, tau]))
    cov = np.linalg.inv(H)
    yth = tr(threshold)
    mu = (yth - b0) / b1
    sigma = tau / b1
    J = np.array([[-1 / b1, -(yth - b0) / b1 ** 2, 0.0],
                  [0.0, -tau / b1 ** 2, 1 / b1]])
    cov_ms = J @ cov @ J.T
    back = np.exp if log_a else (lambda v: v)
    x90 = mu + Z90 * sigma
    x9095 = x90 + Z95 * np.sqrt(max(cov_ms[0, 0] + Z90 ** 2 * cov_ms[1, 1]
                                    + 2 * Z90 * cov_ms[0, 1], 0))
    return AhatResult(b0, b1, tau, cov, mu, sigma, cov_ms, float(back(mu)), float(back(x90)),
                      float(back(x9095)), threshold, log_a, int(np.sum(y <= lo)),
                      int(np.sum(y >= hi)), float(-r.fun))


def false_call_probability(threshold, noise_mean, noise_sd, log_ahat=True):
    """P(noise > threshold) for normally distributed (log-)noise."""
    tr = np.log if log_ahat else (lambda v: v)
    return float(norm.sf((tr(threshold) - noise_mean) / noise_sd))


def noise_fit(noise_samples, log_ahat=True):
    v = np.log(np.maximum(noise_samples, 1e-300)) if log_ahat else np.asarray(noise_samples)
    return float(np.mean(v)), float(np.std(v, ddof=1))


def roc_curve(res: AhatResult, a_target, noise_mean, noise_sd, thresholds, log_ahat=True):
    """(P_FA, POD(a_target)) as the decision threshold varies."""
    tr = np.log if log_ahat else (lambda v: v)
    x = np.log(a_target) if res.log_a else a_target
    pfa, pod = [], []
    for th in thresholds:
        pfa.append(norm.sf((tr(th) - noise_mean) / noise_sd))
        mu = (tr(th) - res.beta0) / res.beta1
        pod.append(norm.cdf((x - mu) / res.sigma))
    return np.array(pfa), np.array(pod)


# =========================================================================== hit / miss
LINKS = {
    "logit": (lambda e: 1 / (1 + np.exp(-e)), lambda p: np.log(p / (1 - p))),
    "probit": (norm.cdf, norm.ppf),
    "cloglog": (lambda e: 1 - np.exp(-np.exp(e)), lambda p: np.log(-np.log(1 - p))),
    "loglog": (lambda e: np.exp(-np.exp(-e)), lambda p: -np.log(-np.log(p))),
}


@dataclass
class HitMissResult:
    beta: np.ndarray
    cov: np.ndarray
    link: str
    log_a: bool
    loglik: float
    a50: float
    a90: float
    a90_95: float
    boundary: np.ndarray        # (n, 2) points on the LR confidence-region boundary
    x: np.ndarray
    hits: np.ndarray

    def _eta(self, a, beta=None):
        b = self.beta if beta is None else beta
        x = np.log(a) if self.log_a else np.asarray(a, float)
        return b[0] + b[1] * x

    def pod(self, a):
        return LINKS[self.link][0](self._eta(a))

    def pod_lower(self, a):
        x = np.log(a) if self.log_a else np.asarray(a, float)
        B = self.boundary
        eta = B[:, 0][:, None] + B[:, 1][:, None] * np.atleast_1d(x)[None]
        return LINKS[self.link][0](eta.min(axis=0))

    def pod_upper(self, a):
        x = np.log(a) if self.log_a else np.asarray(a, float)
        B = self.boundary
        eta = B[:, 0][:, None] + B[:, 1][:, None] * np.atleast_1d(x)[None]
        return LINKS[self.link][0](eta.max(axis=0))


def _hm_ll(beta, x, h, link):
    F = LINKS[link][0]
    p = np.clip(F(beta[0] + beta[1] * x), 1e-12, 1 - 1e-12)
    return np.sum(h * np.log(p) + (1 - h) * np.log(1 - p))


def hit_miss(a, hits, link="logit", log_a=True, n_boundary=360):
    a = np.asarray(a, float); h = np.asarray(hits, float)
    x = np.log(a) if log_a else a
    ginv = LINKS[link][1]
    pm = np.clip(np.mean(h), 0.05, 0.95)
    b0 = [ginv(pm) - 1.0 * np.mean(x), 1.0]
    nll = lambda b: -_hm_ll(b, x, h, link)
    r = minimize(nll, b0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12,
                                                            maxiter=20000))
    r = minimize(nll, r.x, method="BFGS")
    beta = r.x
    ll = -r.fun
    H = _hessian(nll, beta)
    cov = np.linalg.inv(H)
    # LR confidence-region boundary: radial bisection in whitened coordinates
    Lc = np.linalg.cholesky(cov)
    pts = []
    for ang in np.linspace(0, 2 * np.pi, n_boundary, endpoint=False):
        d = Lc @ np.array([np.cos(ang), np.sin(ang)])
        lo, hi = 0.0, 1.0
        while 2 * (ll - _hm_ll(beta + hi * d, x, h, link)) < LR_CRIT and hi < 1e3:
            hi *= 2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if 2 * (ll - _hm_ll(beta + mid * d, x, h, link)) < LR_CRIT:
                lo = mid
            else:
                hi = mid
        pts.append(beta + lo * d)
    B = np.array(pts)
    F, Finv = LINKS[link]
    back = np.exp if log_a else (lambda v: v)
    a50 = float(back((Finv(0.5) - beta[0]) / beta[1]))
    a90 = float(back((Finv(0.9) - beta[0]) / beta[1]))
    # a90/95: smallest a with lower bound ≥ 0.9
    xs = np.linspace(x.min() - 2, x.max() + 3, 4000)
    lower = F((B[:, 0][:, None] + B[:, 1][:, None] * xs[None]).min(axis=0))
    ok = np.nonzero(lower >= 0.9)[0]
    a9095 = float(back(xs[ok[0]])) if len(ok) else float("nan")
    return HitMissResult(beta, cov, link, log_a, ll, a50, a90, a9095, B, x, h)


# =========================================================================== synthetic data
def synthetic_ahat(n=60, a_range=(0.3e-3, 6e-3), b0=4.0, b1=1.1, tau=0.35, floor=None,
                   saturation=None, seed=0, a_unit=1e-3):
    """â = exp(β₀ + β₁ ln(a/unit) + ε) for log-uniform sizes; returns (a, â)."""
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(a_range[0]), np.log(a_range[1]), n))
    ahat = np.exp(b0 + b1 * np.log(a / a_unit) + tau * rng.standard_normal(n))
    if floor is not None:
        ahat = np.maximum(ahat, floor)
    if saturation is not None:
        ahat = np.minimum(ahat, saturation)
    return a, ahat


def synthetic_hitmiss(n=120, a_range=(0.2e-3, 5e-3), a50=1.2e-3, slope=4.0, seed=0,
                      link="logit", false_call=0.0, miss_floor=0.0):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(a_range[0]), np.log(a_range[1]), n))
    F = LINKS[link][0]
    p = F(slope * (np.log(a) - np.log(a50)))
    p = false_call + (1 - false_call - miss_floor) * p
    return a, (rng.uniform(size=n) < p).astype(int)


# =========================================================================== model-assisted POD
def mapod_ultrasonic(sizes, setup=None, n_per=4, seed=0, tilt_sd=12.0, depth_range=None):
    """Model-assisted â-vs-a for planar cracks (Kirchhoff discs) under a contact probe:
    random tilt ~ N(0, tilt_sd²) deg, random depth and lateral offset, electronic and grain
    noise; â = peak envelope in the flaw gate (units of p/p₀). Returns (a, â, noise samples)."""
    from dataclasses import replace
    from . import ultrasonics as ut
    from . import signals as sg
    rng = np.random.default_rng(seed)
    s = setup or ut.UTSetup(thickness=40e-3, multiples=1)
    dr = depth_range or (0.3 * s.thickness, 0.7 * s.thickness)
    A, Ah = [], []
    for a in sizes:
        for _ in range(n_per):
            z = rng.uniform(*dr)
            fl = ut.Flaw("disc", x=rng.normal(0, 0.5e-3), z=z, size=a,
                         tilt=rng.normal(0, tilt_sd))
            ss = replace(s, seed=int(rng.integers(1 << 30)))
            t, tr = ut.ascan(ss, [fl])
            env = sg.envelope(tr)
            tz = ss.t0() + 2 * z / ss.mat.cL
            gate = np.abs(t - tz) < 3 / ss.probe.f0 + a / ss.mat.cL
            A.append(a); Ah.append(env[gate].max())
    # noise: same gates without a flaw
    noise = []
    for k in range(40):
        ss = replace(s, seed=10_000 + k)
        t, tr = ut.ascan(ss, [])
        env = sg.envelope(tr)
        z = rng.uniform(*dr)
        tz = ss.t0() + 2 * z / ss.mat.cL
        noise.append(env[np.abs(t - tz) < 3 / ss.probe.f0].max())
    return np.array(A), np.array(Ah), np.array(noise)


def mapod_eddy(lengths, coil=None, f=100e3, sigma=1.74e7, n_per=4, seed=0, liftoff_sd=0.05e-3,
               noise_ohm=2e-4):
    """Model-assisted â-vs-a for semi-circular surface cracks (depth = length/2) with random
    lift-off and orientation; â = peak |ΔZ| (after rotating out lift-off) along a line scan."""
    from . import eddy_current as ec
    rng = np.random.default_rng(seed)
    c0 = coil or ec.Coil(r1=0.6e-3, r2=1.2e-3, liftoff=0.2e-3, height=1.0e-3, turns=150)
    spec = ec.Specimen.halfspace(sigma)
    xs = np.linspace(-3e-3, 3e-3, 61)
    A, Ah = [], []
    for L in lengths:
        for _ in range(n_per):
            lo = max(0.02e-3, c0.liftoff + rng.normal(0, liftoff_sd))
            c = ec.Coil(c0.r1, c0.r2, lo, c0.height, c0.turns)
            fl = ec.ECFlaw("crack", x=0.0, y=rng.normal(0, 0.15e-3), depth=L / 2, size=L,
                           angle=rng.normal(0, 15))
            dz = ec.flaw_scan(c, spec, f, [fl], xs)
            dz = dz + noise_ohm * (rng.standard_normal(len(xs)) + 1j * rng.standard_normal(len(xs)))
            A.append(L); Ah.append(np.max(np.abs(dz)))
    noise = np.array([np.max(np.abs(noise_ohm * (rng.standard_normal(61) + 1j *
                                                 rng.standard_normal(61)))) for _ in range(60)])
    return np.array(A), np.array(Ah), noise
