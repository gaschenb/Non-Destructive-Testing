"""Eddy-current testing with the Dodd–Deeds analytic model.

Coil: rectangular cross-section r₁ ≤ r ≤ r₂, lift-off l₁ ≤ z ≤ l₂ = l₁ + h, N turns.
Specimen: stack of conducting layers (σᵢ, μᵢ, tᵢ) over a half-space.

    ΔZ(ω) = jωπμ₀N² / ((r₂−r₁)²(l₂−l₁)²) ∫₀^∞ P(α)²/α⁶ (e^{−αl₁} − e^{−αl₂})² φ(α) dα
    P(α)  = ∫_{αr₁}^{αr₂} x J₁(x) dx = I(αr₂) − I(αr₁),  I(x) = (πx/2)[J₁(x)H₀(x) − J₀(x)H₁(x)]
    αᵢ    = √(α² + jωμ₀μᵢσᵢ)

φ(α) is the reflection coefficient of the stack, built bottom-up with the transfer
relation Y_top = (αᵢ/μᵢ)(Y μᵢ/αᵢ + tanh αᵢtᵢ)/(1 + (Y μᵢ/αᵢ) tanh αᵢtᵢ) and φ = (α − Y)/(α + Y)
(half-space: φ = (μα − α₁)/(μα + α₁)). The isolated-coil reactance X₀ uses the kernel
2(αh + e^{−αh} − 1).

Small flaws use reciprocity (Auld): ΔZ = (1/I²) ∫ (σ − σ_flaw) E₀·E dV with E₀ = −jωA_φ φ̂
from the same spectral solution (sign and normalisation verified against the exact layered
solution for a thin perturbed layer). Voids (spheres) use the quasi-static internal field E = (3/2)E₀; thin cracks
use the dipole-layer polarisability of a penny-shaped crack, (8/3)a³ for E₀ normal to it,
distributed with the crack-opening shape √(1 − ρ²). Pulsed eddy currents invert ΔL(s) with the
Gaver–Stehfest algorithm (α₁ is real for real s, so no FFT is needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import j0, j1, struve, factorial

MU0 = 4e-7 * np.pi


@dataclass
class Coil:
    r1: float = 2.0e-3
    r2: float = 4.0e-3
    liftoff: float = 0.5e-3
    height: float = 2.0e-3
    turns: int = 200

    @property
    def l1(self):
        return self.liftoff

    @property
    def l2(self):
        return self.liftoff + self.height


@dataclass
class Layer:
    thickness: float
    sigma: float
    mur: float = 1.0


@dataclass
class Specimen:
    """Layers from the top; `base` is the half-space underneath (default: air)."""
    layers: list = field(default_factory=list)
    base: Layer = field(default_factory=lambda: Layer(np.inf, 0.0, 1.0))

    @staticmethod
    def halfspace(sigma, mur=1.0):
        return Specimen([], Layer(np.inf, sigma, mur))

    @staticmethod
    def plate(thickness, sigma, mur=1.0):
        return Specimen([Layer(thickness, sigma, mur)])

    @staticmethod
    def coated(coat_t, coat_sigma, sub_sigma, sub_mur=1.0, coat_mur=1.0):
        return Specimen([Layer(coat_t, coat_sigma, coat_mur)], Layer(np.inf, sub_sigma, sub_mur))


def _I(x):
    return 0.5 * np.pi * x * (j1(x) * struve(0, x) - j0(x) * struve(1, x))


def P_alpha(alpha, r1, r2):
    return _I(alpha * r2) - _I(alpha * r1)


def alpha_grid(coil: Coil, n=1600):
    """Log-spaced quadrature nodes and weights (trapezoid in ln α)."""
    amax = min(40.0 / max(2 * coil.l1, 1e-5), 3000.0 / coil.r2)
    amin = 1e-3 / coil.r2
    u = np.linspace(np.log(amin), np.log(amax), n)
    a = np.exp(u)
    w = np.full(n, u[1] - u[0]) * a
    w[0] *= 0.5; w[-1] *= 0.5
    return a, w


def _sqrt_pos(z):
    r = np.sqrt(z + 0j)
    return np.where(np.real(r) < 0, -r, r)


def _tanh(z):
    e = np.exp(-2 * z)
    return (1 - e) / (1 + e)


def admittance(alpha, spec: Specimen, s):
    """Y at the top surface for Laplace variable s (= jω) and spectral α (broadcast)."""
    b = spec.base
    a_b = _sqrt_pos(alpha ** 2 + s * MU0 * b.mur * b.sigma)
    Y = a_b / b.mur
    for L in reversed(spec.layers):
        ai = _sqrt_pos(alpha ** 2 + s * MU0 * L.mur * L.sigma)
        u = Y * L.mur / ai
        th = _tanh(ai * L.thickness)
        Y = (ai / L.mur) * (u + th) / (1 + u * th)
    return Y


def reflection(alpha, spec, s):
    Y = admittance(alpha, spec, s)
    return (alpha - Y) / (alpha + Y)


def _kernel(coil, a):
    P = P_alpha(a, coil.r1, coil.r2)
    A = (np.exp(-a * coil.l1) - np.exp(-a * coil.l2)) ** 2
    return P ** 2 / a ** 6, A


def coil_constant(coil):
    return np.pi * MU0 * coil.turns ** 2 / ((coil.r2 - coil.r1) ** 2 * coil.height ** 2)


def isolated_reactance(coil: Coil, f):
    """X₀ = ωL₀ of the coil in air."""
    a, w = alpha_grid(coil)
    P = P_alpha(a, coil.r1, coil.r2)
    h = coil.height
    ker = P ** 2 / a ** 6 * 2 * (a * h + np.expm1(-a * h))
    L0 = coil_constant(coil) * np.sum(w * ker)
    return 2 * np.pi * np.asarray(f, float) * L0


def inductance(coil: Coil):
    return float(isolated_reactance(coil, 1.0) / (2 * np.pi))


def delta_z(coil: Coil, spec: Specimen, f):
    """Impedance change ΔZ = ΔR + jΔX [Ω] at frequencies f (array)."""
    f = np.atleast_1d(np.asarray(f, float))
    a, w = alpha_grid(coil)
    K, A = _kernel(coil, a)
    s = 2j * np.pi * f[:, None]
    phi = reflection(a[None], spec, s)
    integ = np.sum(w * K * A * phi, axis=-1)
    return s[:, 0] * coil_constant(coil) * integ


def normalized_impedance(coil, spec, f):
    """(R/X₀, X/X₀) with the coil's own resistance excluded."""
    dz = delta_z(coil, spec, f)
    X0 = isolated_reactance(coil, f)
    return np.real(dz) / X0, 1 + np.imag(dz) / X0


def characteristic_frequency(coil, sigma, mur=1.0):
    """f₉₀-like scaling frequency f_c = 1/(πμσ r̄²) (Förster), r̄ = mean coil radius."""
    rbar = 0.5 * (coil.r1 + coil.r2)
    return 1 / (np.pi * MU0 * mur * sigma * rbar ** 2)


def skin_depth(f, sigma, mur=1.0):
    return 1 / np.sqrt(np.pi * np.asarray(f, float) * MU0 * mur * sigma)


# =========================================================================== impedance plane
def liftoff_curves(coil, spec, f, liftoffs):
    out = []
    for lo in liftoffs:
        c = Coil(coil.r1, coil.r2, lo, coil.height, coil.turns)
        out.append(normalized_impedance(c, spec, f))
    return np.array(out)                               # (n_lo, 2, n_f)


def conductivity_locus(coil, f, sigmas, mur=1.0, thickness=None):
    """Normalised impedance at fixed f as conductivity varies (thick or plate specimen)."""
    pts = []
    for s in sigmas:
        sp = Specimen.halfspace(s, mur) if thickness is None else Specimen.plate(thickness, s, mur)
        r, x = normalized_impedance(coil, sp, [f])
        pts.append((r[0], x[0]))
    return np.array(pts)


def thickness_locus(coil, f, sigma, thicknesses, mur=1.0):
    pts = []
    for t in thicknesses:
        r, x = normalized_impedance(coil, Specimen.plate(t, sigma, mur), [f])
        pts.append((r[0], x[0]))
    return np.array(pts)


def invert_liftoff_conductivity(coil, f, z_meas, sigma0=1e7, lo0=0.5e-3, mur=1.0):
    """Estimate (σ, lift-off) of a half-space from one measured complex ΔZ at frequency f
    by Gauss–Newton on (log σ, lift-off)."""
    from scipy.optimize import least_squares

    def res(p):
        c = Coil(coil.r1, coil.r2, p[1], coil.height, coil.turns)
        dz = delta_z(c, Specimen.halfspace(np.exp(p[0]), mur), [f])[0]
        return np.array([dz.real - z_meas.real, dz.imag - z_meas.imag]) / abs(z_meas)
    r = least_squares(res, [np.log(sigma0), lo0], bounds=([np.log(1e4), 0], [np.log(1e8), 0.02]))
    return float(np.exp(r.x[0])), float(r.x[1])


# =========================================================================== fields & flaws
def field_in_specimen(coil, spec, f, r, depth):
    """E_φ (V/m per ampere) at radial distances r (array) and depth (scalar, > 0)."""
    a, w = alpha_grid(coil, n=1400)
    s = 2j * np.pi * f
    P = P_alpha(a, coil.r1, coil.r2)
    n = coil.turns / ((coil.r2 - coil.r1) * coil.height)
    C = 0.5 * MU0 * n * P / a ** 3 * (np.exp(-a * coil.l1) - np.exp(-a * coil.l2))
    phi = reflection(a, spec, s)
    amp = C * (1 + phi)                                # a(z = 0⁻)
    # walk down through the layers. Inside a layer of thickness t (depth s from its top)
    # a(s) = P[e^{−α s} + ρ e^{−α(2t − s)}], ρ = (1 − u_b)/(1 + u_b), u_b = μ Y_bottom / α,
    # which stays bounded for every α (no growing exponentials).
    z_top = 0.0
    layers = list(spec.layers) + [spec.base]
    Ys = _layer_tops(a, spec, s)
    av = None
    for i, L in enumerate(layers):
        ai = _sqrt_pos(a ** 2 + s * MU0 * L.mur * L.sigma)
        if not np.isfinite(L.thickness):
            av = amp * np.exp(-ai * (depth - z_top))
            break
        t = L.thickness
        ub = L.mur * Ys[i + 1] / ai
        rho = (1 - ub) / (1 + ub)
        et = np.exp(-ai * t)
        P = amp / (1 + rho * et * et)
        if depth <= z_top + t:
            sd = depth - z_top
            av = P * (np.exp(-ai * sd) + rho * np.exp(-ai * (2 * t - sd)))
            break
        amp = P * et * (1 + rho)
        z_top += t
    J = j1(np.outer(np.atleast_1d(r), a))
    A_phi = J @ (w * av)
    return -s * A_phi, L.sigma


def _layer_tops(a, spec, s):
    """Admittance Y at the top of every layer (and the base)."""
    b = spec.base
    Y = _sqrt_pos(a ** 2 + s * MU0 * b.mur * b.sigma) / b.mur
    tops = [Y]
    for L in reversed(spec.layers):
        ai = _sqrt_pos(a ** 2 + s * MU0 * L.mur * L.sigma)
        u = Y * L.mur / ai
        th = _tanh(ai * L.thickness)
        Y = (ai / L.mur) * (u + th) / (1 + u * th)
        tops.append(Y)
    return tops[::-1]


@dataclass
class ECFlaw:
    """kind 'void' (sphere of diameter `size` centred at `depth`), 'crack' (thin
    semi-elliptical surface crack of length `size`, depth `depth`, normal at `angle` deg from
    x), 'subcrack' (embedded penny crack radius size/2 at `depth`, vertical plane)."""
    kind: str = "void"
    x: float = 0.0
    y: float = 0.0
    depth: float = 1e-3
    size: float = 1e-3
    angle: float = 0.0


def flaw_points(fl: ECFlaw, n=9):
    """Quadrature points (x, y, depth), weights (m³ polarisability) and normals for a flaw."""
    if fl.kind == "void":
        V = np.pi / 6 * fl.size ** 3
        return (np.array([[fl.x, fl.y, fl.depth]]), np.array([1.5 * V]), None)
    if fl.kind in ("crack", "subcrack"):
        c = fl.size / 2                        # half length
        a = fl.depth if fl.kind == "crack" else fl.size / 2
        b = min(a, c)
        u = (np.arange(n) + 0.5) / n * 2 - 1          # along length (−1..1)
        if fl.kind == "crack":
            v = (np.arange(n) + 0.5) / n               # depth fraction 0..1
            z0 = 0.0
        else:
            v = (np.arange(n) + 0.5) / n * 2 - 1
            z0 = fl.depth
        U, V = np.meshgrid(u, v, indexing="ij")
        rho2 = U ** 2 + V ** 2
        inside = rho2 < 1
        dA = (2 * c / n) * ((a if fl.kind == "crack" else 2 * a) / n)
        wgt = (4 * b / np.pi) * np.sqrt(np.clip(1 - rho2, 0, None)) * dA * inside
        th = np.radians(fl.angle)
        t_dir = np.array([-np.sin(th), np.cos(th)])    # crack runs ⟂ to its normal
        nrm = np.array([np.cos(th), np.sin(th)])
        xs = fl.x + U.ravel() * c * t_dir[0]
        ys = fl.y + U.ravel() * c * t_dir[1]
        zs = (V.ravel() * a + z0) if fl.kind == "crack" else (z0 + V.ravel() * a)
        keep = inside.ravel()
        pts = np.stack([xs, ys, np.maximum(zs, 1e-6)], -1)[keep]
        return pts, wgt.ravel()[keep], nrm
    raise ValueError(fl.kind)


def flaw_scan(coil, spec, f, flaws, xs, ys=None):
    """ΔZ_flaw(x, y) for the coil scanned over the surface (line if ys is None)."""
    xs = np.atleast_1d(xs)
    ys = np.zeros(1) if ys is None else np.atleast_1d(ys)
    X, Y = np.meshgrid(xs, ys)
    out = np.zeros(X.shape, complex)
    for fl in flaws:
        pts, wgt, nrm = flaw_points(fl)
        for d in np.unique(pts[:, 2]):
            sel = pts[:, 2] == d
            px, py, ww = pts[sel, 0], pts[sel, 1], wgt[sel]
            dx = X[..., None] - px
            dy = Y[..., None] - py
            r = np.hypot(dx, dy)
            rr = np.unique(np.round(r.ravel(), 7))
            E, sig = field_in_specimen(coil, spec, f, rr, d)
            Ei = np.interp(r.ravel(), rr, E.real) + 1j * np.interp(r.ravel(), rr, E.imag)
            Ei = Ei.reshape(r.shape)
            if nrm is not None:
                # E₀ is azimuthal about the coil axis: φ̂ = (−sin φ, cos φ), φ = atan2(p − coil)
                ph = np.arctan2(-dy, -dx)
                proj = -np.sin(ph) * nrm[0] + np.cos(ph) * nrm[1]
                Ei = Ei * proj
            out += sig * np.sum(ww * Ei ** 2, axis=-1)
    return out if ys is not None and len(ys) > 1 else out[0]


def differential(dz_scan, xs, separation):
    """Differential probe signal ΔZ(x + s/2) − ΔZ(x − s/2) from an absolute scan."""
    xs = np.asarray(xs)
    re = np.interp(xs + separation / 2, xs, dz_scan.real) - np.interp(xs - separation / 2, xs, dz_scan.real)
    im = np.interp(xs + separation / 2, xs, dz_scan.imag) - np.interp(xs - separation / 2, xs, dz_scan.imag)
    return re + 1j * im


def rotate_phase(z, deg):
    return z * np.exp(1j * np.radians(deg))


def liftoff_phase(coil, spec, f, d_lo=0.05e-3):
    """Phase (deg) of the lift-off direction in the impedance plane (instruments rotate it to
    horizontal)."""
    c2 = Coil(coil.r1, coil.r2, coil.liftoff + d_lo, coil.height, coil.turns)
    d = delta_z(c2, spec, [f])[0] - delta_z(coil, spec, [f])[0]
    return float(np.degrees(np.angle(d)))


# =========================================================================== pulsed EC
def stehfest_weights(N=14):
    V = np.zeros(N)
    M = N // 2
    for k in range(1, N + 1):
        s = 0.0
        for j in range((k + 1) // 2, min(k, M) + 1):
            s += j ** M * factorial(2 * j) / (factorial(M - j) * factorial(j) * factorial(j - 1)
                                               * factorial(k - j) * factorial(2 * j - k))
        V[k - 1] = (-1) ** (k + M) * s
    return V


def pulsed_response(coil, spec, t, N=14):
    """Transient voltage after switching off a 1 A coil current (V), from
    v(t) = −L⁻¹{ΔL(s) − ΔL(∞)} with ΔL(s) = ΔZ(s)/s, via Gaver–Stehfest."""
    t = np.atleast_1d(np.asarray(t, float))
    a, w = alpha_grid(coil)
    K, A = _kernel(coil, a)
    V = stehfest_weights(N)
    ln2 = np.log(2)
    cc = coil_constant(coil)
    phi_inf = np.real(reflection(a[None], spec, np.array([[1e12]])))
    out = np.zeros(len(t))
    for c0 in range(0, len(t), 16):
        tt = t[c0:c0 + 16]
        s = (np.arange(1, N + 1)[None, :] * ln2 / tt[:, None]).reshape(-1, 1)
        phi = np.real(reflection(a[None], spec, s))
        dL = (cc * np.sum(w * K * A * (phi - phi_inf), axis=-1)).reshape(len(tt), N)
        out[c0:c0 + 16] = -ln2 / tt * (dL @ V)
    return out


def pec_decay_constant(t, v, frac=(0.5, 0.9)):
    """Late-time exponential decay constant τ from a log-linear fit of |v(t)|."""
    t, v = np.asarray(t), np.abs(np.asarray(v))
    i0, i1 = int(frac[0] * len(t)), int(frac[1] * len(t))
    p = np.polyfit(t[i0:i1], np.log(np.maximum(v[i0:i1], 1e-300)), 1)
    return -1 / p[0]
