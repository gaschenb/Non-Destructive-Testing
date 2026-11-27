"""Guided waves in isotropic plates: Lamb (symmetric / antisymmetric) and SH dispersion,
group velocity, through-thickness mode shapes, dispersive tone-burst propagation and
pulse-echo inspection of a plate with defects.

Rayleigh–Lamb equations (half thickness h, p² = ω²/c_L² − k², q² = ω²/c_T² − k²), written in
forms that stay real for every (ω, k) (p, q may be imaginary):
    S:  D_S = (k² − q²)² cos(ph) sin(qh)/q + 4k² p sin(ph) cos(qh)
    A:  D_A = (k² − q²)² sin(ph)/p cos(qh) + 4k² q sin(qh) cos(ph)
Both are entire functions of p², q², so every sign change in c_p at fixed f is a root; roots
are bracketed on a phase-velocity grid and polished with Brent's method. SH modes are
analytic: c_p = c_T / √(1 − (n c_T / (2 f h))²) with 2h the thickness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import signals as sg
from .materials import get


def _sinc_like(x, h):
    """sin(xh)/x for x = √(x²) possibly imaginary (x² real)."""
    x = np.asarray(x, complex)
    small = np.abs(x) < 1e-12
    out = np.where(small, h, np.sin(x * h) / np.where(small, 1, x))
    return np.real(out)


def _parts(f, c, cL, cT, h):
    w = 2 * np.pi * f
    k = w / c
    p2 = (w / cL) ** 2 - k ** 2
    q2 = (w / cT) ** 2 - k ** 2
    p = np.sqrt(p2 + 0j)
    q = np.sqrt(q2 + 0j)
    return k, p, q, p2, q2


def dispersion_S(f, c, cL, cT, h):
    k, p, q, p2, q2 = _parts(f, c, cL, cT, h)
    v = (k ** 2 - q2) ** 2 * np.real(np.cos(p * h)) * _sinc_like(q, h) \
        + 4 * k ** 2 * np.real(p * np.sin(p * h)) * np.real(np.cos(q * h))
    return v


def dispersion_A(f, c, cL, cT, h):
    k, p, q, p2, q2 = _parts(f, c, cL, cT, h)
    v = (k ** 2 - q2) ** 2 * _sinc_like(p, h) * np.real(np.cos(q * h)) \
        + 4 * k ** 2 * np.real(q * np.sin(q * h)) * np.real(np.cos(p * h))
    return v


def _scale(f, c, cL, cT, h):
    w = 2 * np.pi * f
    return (w / cT) ** 4 * h + 1e-30


@dataclass
class Dispersion:
    f: np.ndarray                    # Hz
    modes: dict                      # name → phase velocity on f (forward branch; nan where absent)
    group: dict                      # name → group velocity on f
    thickness: float
    material: str
    branches: dict = None            # name → dict(k, f, cp, cg): full curves incl. backward waves

    def fd(self):
        return self.f * self.thickness

    def wavelength(self, name):
        return self.modes[name] / self.f


def lamb_branches(material="aluminium-6061", thickness=1e-3, f_max=5e6, n_k=400, n_scan=3000,
                  max_modes=6):
    """Lamb modes traced in the (k, ω) plane: at fixed real k the roots ω_n(k) of each family
    are sorted, and modes of one symmetry never cross, so the n-th root is mode n. This
    labels branches unambiguously and captures backward-wave (negative c_g) segments, e.g. S1
    below its zero-group-velocity point."""
    m = get(material)
    cL, cT = m.cL, m.cT
    h = thickness / 2
    k_max = 2 * np.pi * f_max / (0.8 * cT)
    ks = np.linspace(k_max / n_k, k_max, n_k)
    # log-spaced start so that the A0 root (f ∝ k² at small k) is never missed: a missing
    # lowest root would shift every mode index at that k
    fs = np.concatenate([np.geomspace(f_max * 1e-7, f_max / n_scan, 200, endpoint=False),
                         np.linspace(f_max / n_scan, f_max * 1.02, n_scan)])
    out = {}
    for fam, D in (("S", dispersion_S), ("A", dispersion_A)):
        roots = np.full((n_k, max_modes), np.nan)
        F, K = np.meshgrid(fs, ks)
        C = 2 * np.pi * F / K
        with np.errstate(all="ignore"):
            V = D(F, C, cL, cT, h) / _scale(F, C, cL, cT, h)
        for i, k in enumerate(ks):
            v = V[i]
            idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
            rs = []
            for j in idx:
                a, b = fs[j], fs[j + 1]
                g = lambda ff: D(ff, 2 * np.pi * ff / k, cL, cT, h) / _scale(ff, 2 * np.pi * ff / k, cL, cT, h)
                try:
                    r = brentq(g, a, b, xtol=1e-12 * b, maxiter=100)
                except ValueError:
                    continue
                c = 2 * np.pi * r / k
                if abs(c - cT) / cT < 2e-4 or abs(c - cL) / cL < 2e-4:
                    continue
                rs.append(r)
            for jj, r in enumerate(sorted(rs)[:max_modes]):
                roots[i, jj] = r
        for jj in range(max_modes):
            fr = roots[:, jj]
            good = np.isfinite(fr) & (fr <= f_max)
            if good.sum() < 4:
                continue
            kk, ff = ks[good], fr[good]
            w = 2 * np.pi * ff
            out[f"{fam}{jj}"] = dict(k=kk, f=ff, cp=w / kk, cg=np.gradient(w, kk))
    return out


def lamb_dispersion(material="aluminium-6061", thickness=1e-3, f_max=5e6, n_f=200,
                    max_modes=6, n_k=400, **_):
    """Phase and group velocity of the S and A Lamb modes on a frequency grid (forward
    branches), plus the full (k, ω) branches."""
    m = get(material)
    f = np.linspace(f_max / n_f, f_max, n_f)
    br = lamb_branches(material, thickness, f_max, n_k=n_k, max_modes=max_modes)
    modes, group = {}, {}
    for name, b in br.items():
        i0 = int(np.argmin(b["f"]))               # forward branch starts at the minimum of ω(k)
        ff, cp, cg = b["f"][i0:], b["cp"][i0:], b["cg"][i0:]
        keep = np.concatenate([[True], np.diff(ff) > 0])
        ff, cp, cg = ff[keep], cp[keep], cg[keep]
        if len(ff) < 2:
            continue
        modes[name] = np.interp(f, ff, cp, left=np.nan, right=np.nan)
        group[name] = np.interp(f, ff, cg, left=np.nan, right=np.nan)
    return Dispersion(f, modes, group, thickness, m.name, br)


def zgv_frequency(disp: Dispersion, name="S1"):
    """Frequency of the zero-group-velocity point of a branch (nan if ω(k) has no minimum)."""
    b = (disp.branches or {}).get(name)
    if b is None:
        return float("nan")
    i = int(np.argmin(b["f"]))
    if i == 0 or i >= len(b["f"]) - 1:
        return float("nan")
    return float(b["f"][i])


def group_velocity(f, cp):
    """c_g = dω/dk from phase velocity samples along a branch (finite differences in k)."""
    k = 2 * np.pi * f / cp
    cg = np.full_like(cp, np.nan)
    good = np.isfinite(k)
    idx = np.nonzero(good)[0]
    if len(idx) < 3:
        return cg
    w = 2 * np.pi * f[idx]
    kk = k[idx]
    cg[idx] = np.gradient(w, kk)
    return cg


def sh_dispersion(material, thickness, f, n_modes=4):
    m = get(material)
    cT = m.cT
    out = {}
    for n in range(n_modes):
        with np.errstate(invalid="ignore", divide="ignore"):
            arg = 1 - (n * cT / (2 * f * thickness)) ** 2
            cp = np.where(arg > 0, cT / np.sqrt(np.clip(arg, 1e-30, None)), np.nan)
            cg = np.where(arg > 0, cT * np.sqrt(np.clip(arg, 0, None)), np.nan)
        out[f"SH{n}"] = (cp, cg)
    return out


def cutoff_frequencies(material, thickness, n=4):
    """Cut-off frequency-thicknesses of higher Lamb modes (Hz·m):
    S: f d = n c_T (n odd/2…) / L-type; A: likewise — returns both thickness-resonance sets."""
    m = get(material)
    d = thickness
    shear = [(j * m.cT / (2 * d)) for j in range(1, n + 1)]
    longi = [(j * m.cL / (2 * d)) for j in range(1, n + 1)]
    return dict(shear=shear, longitudinal=longi)


def mode_shape(material, thickness, f, cp, family="S", n=81):
    """Through-thickness displacement (u_x, u_z) of a Lamb mode at (f, c_p), normalised."""
    m = get(material)
    cL, cT = m.cL, m.cT
    h = thickness / 2
    w = 2 * np.pi * f
    k = w / cp
    p = np.sqrt((w / cL) ** 2 - k ** 2 + 0j)
    q = np.sqrt((w / cT) ** 2 - k ** 2 + 0j)
    z = np.linspace(-h, h, n)
    if family == "S":
        # potentials φ = A cos(pz), ψ = B sin(qz); traction-free at z = ±h
        A = 1.0
        B = -2j * k * p * np.sin(p * h) / ((k ** 2 - q ** 2) * np.sin(q * h) + 1e-300)
        ux = 1j * k * A * np.cos(p * z) + q * B * np.cos(q * z)
        uz = -p * A * np.sin(p * z) - 1j * k * B * np.sin(q * z)
    else:
        A = 1.0
        B = 2j * k * p * np.cos(p * h) / ((k ** 2 - q ** 2) * np.cos(q * h) + 1e-300)
        ux = 1j * k * A * np.sin(p * z) - q * B * np.sin(q * z)
        uz = p * A * np.cos(p * z) - 1j * k * B * np.cos(q * z)
    # choose the global phase making ux real where it is largest
    ph = np.angle(ux[np.argmax(np.abs(ux))]) if np.max(np.abs(ux)) > 0 else 0
    ux, uz = np.real(ux * np.exp(-1j * ph)), np.real(uz * np.exp(-1j * ph))
    s = max(np.max(np.abs(ux)), np.max(np.abs(uz)), 1e-300)
    return z, ux / s, uz / s


def phase_velocity_at(disp: Dispersion, name, f):
    v = disp.modes[name]
    good = np.isfinite(v)
    if good.sum() < 2:
        return np.full(np.shape(f), np.nan) if np.ndim(f) else np.nan
    return np.interp(f, disp.f[good], v[good], left=np.nan, right=np.nan)


def propagate(disp: Dispersion, name, f0, cycles, distances, fs=None, t_max=None,
              attenuation_db_m=0.0):
    """Tone-burst of `cycles` at f0 propagated over each distance with mode `name`:
    u(x, t) = IFFT[U(ω) e^{−i k(ω) x}]. Returns (t, signals (n_x, n))."""
    distances = np.atleast_1d(distances)
    fs = fs or 20 * f0
    v = disp.modes[name]
    good = np.isfinite(v)
    cmin = np.nanmin(disp.group[name][good]) if np.any(good) else 1000
    t_max = t_max or (distances.max() / max(cmin, 200) * 1.3 + 3 * cycles / f0)
    n = int(2 ** np.ceil(np.log2(t_max * fs)))
    t = np.arange(n) / fs
    u0 = sg.tone_burst(t, f0, cycles)
    U = np.fft.rfft(u0)
    f = np.fft.rfftfreq(n, 1 / fs)
    cp = np.interp(f, disp.f[good], v[good], left=np.nan, right=np.nan)
    k = np.where(np.isfinite(cp), 2 * np.pi * f / np.where(np.isfinite(cp), cp, 1), 0)
    mask = np.isfinite(cp) & (np.abs(U) > 1e-4 * np.abs(U).max())
    att = attenuation_db_m / 8.686
    out = np.empty((len(distances), n))
    for i, x in enumerate(distances):
        H = np.where(mask, np.exp(-1j * k * x - att * x), 0)
        out[i] = np.fft.irfft(U * H, n) / np.sqrt(max(x, 1e-3) / 0.1)      # cylindrical spreading
    return t, out


def excitability_wavelength(disp: Dispersion, name, wavelength):
    """Frequency at which mode `name` has the given wavelength (comb / interdigital
    transducer tuning): solve c_p(f)/f = λ."""
    v = disp.modes[name]
    good = np.isfinite(v)
    if good.sum() < 2:
        return np.nan
    lam = v[good] / disp.f[good]
    ff = disp.f[good]
    order = np.argsort(lam)
    return float(np.interp(wavelength, lam[order], ff[order], left=np.nan, right=np.nan))


@dataclass
class PlateDefect:
    """A reflector in a plate at distance `x` from the transducer with reflection
    coefficient `R` per mode (simple scattering model: R ≈ wall-loss depth ratio for A0/S0)."""
    x: float = 0.3
    depth_ratio: float = 0.3
    width: float = 5e-3


def pulse_echo_plate(disp: Dispersion, name, f0, cycles, defects, length=1.0, fs=None,
                     attenuation_db_m=2.0, noise=0.01, seed=0, mode_conversion=0.15):
    """Long-range pulse-echo screening along a plate strip: echoes from defects and the far
    end. Notch reflection |R| ≈ (depth ratio)·|sin(π w/λ)| + 0.05·depth ratio (thin-notch
    interference model); part of each defect echo returns mode-converted to the other
    fundamental mode (path x with `name`, x with the other mode). Spreading 1/√(path)."""
    rng = np.random.default_rng(seed)
    fs = fs or 20 * f0
    other = "S0" if name.startswith("A") else "A0"
    v = disp.modes[name]
    good = np.isfinite(v)
    cg = disp.group[name][np.isfinite(disp.group[name])]
    tmax = 2 * length / max(np.nanmin(cg[cg > 0]) if np.any(cg > 0) else 500, 200) * 1.25 \
        + 3 * cycles / f0
    n = int(2 ** np.ceil(np.log2(tmax * fs)))
    t = np.arange(n) / fs
    U = np.fft.rfft(sg.tone_burst(t, f0, cycles))
    f = np.fft.rfftfreq(n, 1 / fs)
    band = np.abs(U) > 1e-4 * np.abs(U).max()

    def kof(mode):
        vv = disp.modes.get(mode)
        if vv is None:
            return None
        g = np.isfinite(vv)
        cp = np.interp(f, disp.f[g], vv[g], left=np.nan, right=np.nan)
        ok = np.isfinite(cp) & band & (f > 0)
        return np.where(ok, 2 * np.pi * f / np.where(ok, cp, 1), 0), ok

    k1, ok1 = kof(name)
    k2 = kof(other)
    att = attenuation_db_m / 8.686
    lam = phase_velocity_at(disp, name, f0) / f0
    spec = np.zeros_like(U)
    for d in defects:
        R = d.depth_ratio * abs(np.sin(np.pi * d.width / max(lam, 1e-9))) + 0.05 * d.depth_ratio
        spec += R * np.where(ok1, np.exp(-1j * k1 * 2 * d.x - att * 2 * d.x), 0) / np.sqrt(2 * d.x)
        if k2 is not None and mode_conversion > 0:
            kk2, ok2 = k2
            spec += mode_conversion * R * np.where(ok1 & ok2, np.exp(
                -1j * (k1 + kk2) * d.x - att * 2 * d.x), 0) / np.sqrt(2 * d.x)
    spec += np.where(ok1, np.exp(-1j * k1 * 2 * length - att * 2 * length), 0) / np.sqrt(2 * length)
    sig = np.fft.irfft(U * spec, n)
    sig = sig + noise * np.max(np.abs(sig)) * rng.standard_normal(n)
    return t, sig, float(phase_velocity_at(disp, name, f0))
