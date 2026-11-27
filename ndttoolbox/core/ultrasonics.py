"""Bulk-wave ultrasonics: interfaces, transducer beams, pulse-echo A/B/C-scans, DGS, TOFD, SAFT.

Pulse-echo model
----------------
Every flaw is a set of Kirchhoff surface patches (position, outward normal, area dA, reflection
coefficient 𝓡). With the transducer (radius a) on the surface, the received spectrum of one
patch at distance R and angle θ from the beam axis is, by reciprocity,

    S(ω) = 𝓡 · |n̂·r̂| · dA/(πa²) · p̂(R, θ, ω)² · e^{−2α(ω)R − iω 2R/c(ω)} · P(ω)

with  p̂ = (1 − e^{−ik(√(R²+a²) − R)}) · 2J₁(ka sinθ')/(ka sinθ')  (exact on axis; off axis the
far-field directivity at an effective range that keeps the near-field beam crystal-wide and
narrowest at N) and the causal power-law attenuation of :mod:`.signals`. In the far
field this reduces to the classical DGS relations p_FBH/p₀ = π²a²b²/(λ²z²). The back wall
uses the exact Seki–Granato–Truell diffraction correction
D(s) = 1 − e^{−ika²/s}[J₀(ka²/s) + iJ₁(ka²/s)] and is shadowed by the flaws' projected area
weighted by the beam profile. Grain noise is coloured Rayleigh backscatter (∝ f²) whose rms
scales as 0.1 (grain size / λ)^{3/2} relative to p₀, falls off as 1/√(1 + (z/N)²) beyond the
near field (incoherent sum over the spreading beam) and carries the interface transmission.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import gaussian_filter
from scipy.special import j0, j1

from . import signals as sg
from .materials import Material, get

# =========================================================================== interfaces


def normal_incidence(Z1, Z2):
    """Pressure reflection and transmission coefficients for normal incidence 1 → 2 and the
    energy coefficients. Returns dict(R, T, ER, ET)."""
    R = (Z2 - Z1) / (Z2 + Z1)
    T = 2 * Z2 / (Z2 + Z1)
    return dict(R=R, T=T, ER=R ** 2, ET=4 * Z1 * Z2 / (Z1 + Z2) ** 2)


def snell(theta1, c1, c2):
    """Refraction angle (rad) from Snell's law; nan beyond the critical angle."""
    s = np.sin(theta1) * c2 / c1
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(s) <= 1, np.arcsin(np.clip(s, -1, 1)), np.nan)


def critical_angles(m1: Material, m2: Material):
    """First (L) and second (S) critical angles in degrees for incidence from m1 (L wave)."""
    m1, m2 = get(m1), get(m2)
    out = {}
    for key, c in (("first", m2.cL), ("second", m2.cT)):
        out[key] = float(np.degrees(np.arcsin(m1.cL / c))) if c > m1.cL else float("nan")
    return out


def _cos_from_sin(s):
    s = np.asarray(s, complex)
    c = np.sqrt(1 - s ** 2)
    return np.where(np.imag(c) > 0, -c, c)           # evanescent branch, e^{iωt} convention


def fluid_solid(theta, fluid, solid):
    """Plane L wave from a fluid onto a solid at incidence angles `theta` (rad).

    Brekhovskikh impedance form: Z = ρ₁c₁/cosθ, Z_L = ρ₂c_L/cosθ_L, Z_S = ρ₂c_S/cosθ_S,
    R = (Z_L cos²2θ_S + Z_S sin²2θ_S − Z)/(Z_L cos²2θ_S + Z_S sin²2θ_S + Z).
    Returns the complex pressure reflection coefficient and the energy-flux coefficients
    ER, EL, ES (ER + EL + ES = 1) and the refraction angles."""
    f, s = get(fluid), get(solid)
    th = np.asarray(theta, float)
    s1 = np.sin(th)
    sL, sS = s1 * s.cL / f.cL, s1 * s.cT / f.cL
    cL, cS = _cos_from_sin(sL), _cos_from_sin(sS)
    c1 = np.cos(th) + 0j
    Z1 = f.rho * f.cL / c1
    ZL = s.rho * s.cL / cL
    ZS = s.rho * s.cT / cS
    cos2 = 1 - 2 * sS ** 2
    sin2 = 2 * sS * cS
    Zt = ZL * cos2 ** 2 + ZS * sin2 ** 2
    den = Zt + Z1
    R = (Zt - Z1) / den
    EL = 4 * np.real(Z1) * np.real(ZL) * np.abs(cos2) ** 2 / np.abs(den) ** 2
    ES = 4 * np.real(Z1) * np.real(ZS) * np.abs(sin2) ** 2 / np.abs(den) ** 2
    with np.errstate(invalid="ignore"):
        thL = np.where(np.abs(sL) <= 1, np.arcsin(np.clip(np.real(sL), -1, 1)), np.nan)
        thS = np.where(np.abs(sS) <= 1, np.arcsin(np.clip(np.real(sS), -1, 1)), np.nan)
    return dict(R=R, ER=np.abs(R) ** 2, EL=EL, ES=ES, theta_L=thL, theta_S=thS)


def solid_solid(theta, m1, m2):
    """P wave in solid m1 incident on a welded interface with solid m2 (Zoeppritz equations,
    displacement amplitudes). Returns reflected/transmitted P and S amplitudes and energy
    coefficients (ERp + ERs + ETp + ETs = 1 below all critical angles; evanescent waves carry
    no energy beyond them)."""
    a, b = get(m1), get(m2)
    th = np.atleast_1d(np.asarray(theta, float))
    p = np.sin(th) / a.cL                         # ray parameter
    sj1, si2, sj2 = p * a.cT, p * b.cL, p * b.cT
    ci1 = np.cos(th) + 0j
    cj1, ci2, cj2 = _cos_from_sin(sj1), _cos_from_sin(si2), _cos_from_sin(sj2)
    si1 = np.sin(th) + 0j
    out = np.zeros((len(th), 4), complex)
    r1, r2, a1, b1, a2, b2 = a.rho, b.rho, a.cL, a.cT, b.cL, b.cT
    for n in range(len(th)):
        M = np.array([
            [-si1[n], -cj1[n], si2[n], cj2[n]],
            [ci1[n], -sj1[n], ci2[n], -sj2[n]],
            [2 * si1[n] * ci1[n], (a1 / b1) * (1 - 2 * sj1[n] ** 2),
             (r2 * b2 ** 2 * a1) / (r1 * b1 ** 2 * a2) * 2 * si2[n] * ci2[n],
             (r2 * b2 * a1) / (r1 * b1 ** 2) * (1 - 2 * sj2[n] ** 2)],
            [-(1 - 2 * sj1[n] ** 2), (b1 / a1) * 2 * sj1[n] * cj1[n],
             (r2 * a2) / (r1 * a1) * (1 - 2 * sj2[n] ** 2),
             -(r2 * b2) / (r1 * a1) * 2 * sj2[n] * cj2[n]]])
        rhs = np.array([si1[n], ci1[n], 2 * si1[n] * ci1[n], 1 - 2 * sj1[n] ** 2])
        out[n] = np.linalg.solve(M, rhs)
    Rp, Rs, Tp, Ts = out.T
    norm = r1 * a1 * np.real(ci1)
    E = lambda rho, c, cos, A: rho * c * np.real(cos) * np.abs(A) ** 2 / norm
    return dict(Rp=Rp, Rs=Rs, Tp=Tp, Ts=Ts, ERp=E(r1, a1, ci1, Rp), ERs=E(r1, b1, cj1, Rs),
                ETp=E(r2, a2, ci2, Tp), ETs=E(r2, b2, cj2, Ts))


# =========================================================================== beams
def near_field_length(diameter, f, c):
    """N = (D² − λ²)/(4λ)."""
    lam = c / f
    return (diameter ** 2 - lam ** 2) / (4 * lam)


def beam_spread(diameter, f, c, drop_db=6.0):
    """Far-field half-angle (rad) of a circular piston at which the pulse-echo amplitude drops
    by `drop_db` (−6 dB pulse-echo = −3 dB one-way; sinθ = 0.51 λ/D)."""
    from scipy.optimize import brentq
    target = 10 ** (-drop_db / 40)                 # one-way amplitude
    u = brentq(lambda u: 2 * j1(u) / u - target, 1e-6, 3.83)
    k = 2 * np.pi * f / c
    s = u / (k * diameter / 2)
    return float(np.arcsin(min(s, 1.0)))


def piston_directivity(u):
    """2 J₁(u)/u with the u → 0 limit."""
    u = np.asarray(u, float)
    out = np.ones_like(u)
    nz = np.abs(u) > 1e-8
    out[nz] = 2 * j1(u[nz]) / u[nz]
    return out


def on_axis_pressure(z, a, k):
    """|p(z)/p₀| = 2 |sin(k(√(z²+a²) − z)/2)| for a baffled circular piston."""
    return 2 * np.abs(np.sin(0.5 * k * (np.sqrt(z ** 2 + a ** 2) - z)))


def seki(s, a, k):
    """Seki–Granato–Truell diffraction correction (receiver-averaged pressure of a piston's
    field after total path s, e.g. a back-wall echo at s = 2T)."""
    x = k * a ** 2 / np.asarray(s, float)
    return 1 - np.exp(-1j * x) * (j0(x) + 1j * j1(x))


def piston_field_cw(x, z, a, f, c, n_r=24, n_phi=48):
    """Continuous-wave pressure |p/p₀| of a baffled circular piston in the x–z plane, by
    direct Rayleigh-integral quadrature over the face (ρcv = 1)."""
    k = 2 * np.pi * f / c
    rr = (np.arange(n_r) + 0.5) / n_r * a
    ph = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi
    R_, P_ = np.meshgrid(rr, ph, indexing="ij")
    xs, ys = (R_ * np.cos(P_)).ravel(), (R_ * np.sin(P_)).ravel()
    dS = (R_ * (a / n_r) * (2 * np.pi / n_phi)).ravel()
    X, Z = np.meshgrid(x, z)
    out = np.zeros(X.shape, complex)
    for i in range(0, len(xs), 128):
        sl = slice(i, i + 128)
        d = np.sqrt((X[..., None] - xs[sl]) ** 2 + ys[sl] ** 2 + Z[..., None] ** 2)
        out += np.sum(np.exp(-1j * k * d) / d * dS[sl], axis=-1)
    return np.abs(1j * k / (2 * np.pi) * out)


# =========================================================================== flaws
@dataclass
class Flaw:
    """A reflector. kind: 'fbh' (flat-bottom hole, face up), 'disc' (planar crack / lack of
    fusion, `tilt` about y), 'sdh' (side-drilled hole along y of length `length`),
    'sphere' (pore or inclusion), 'point'. Coordinates in m, z downward from the scan surface.
    `fill` is the material inside (None: void)."""
    kind: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.02
    size: float = 2e-3                 # diameter
    tilt: float = 0.0                  # degrees
    length: float = 0.03               # sdh length / disc second axis (if aspect given)
    aspect: float = 1.0                # disc: length along y / size
    fill: str | None = None
    label: str = ""

    def refl(self, host: Material):
        if self.fill is None:
            return -1.0
        f = get(self.fill)
        return (f.ZL - host.ZL) / (f.ZL + host.ZL)


@dataclass
class Patches:
    pos: np.ndarray        # (Q,3)
    nrm: np.ndarray        # (Q,3) outward unit normals
    area: np.ndarray       # (Q,)
    refl: np.ndarray       # (Q,)
    two_sided: np.ndarray  # (Q,) bool  (thin planar flaws)

    @staticmethod
    def concat(ps):
        ps = [p for p in ps if p is not None and len(p.area)]
        if not ps:
            z = np.zeros((0, 3))
            return Patches(z, z, np.zeros(0), np.zeros(0), np.zeros(0, bool))
        return Patches(*(np.concatenate([getattr(p, n) for p in ps]) for n in
                         ("pos", "nrm", "area", "refl", "two_sided")))


def flaw_patches(fl: Flaw, host: Material, spacing: float, max_patches=900):
    """Discretise a flaw into Kirchhoff patches with roughly `spacing` resolution."""
    host = get(host)
    R = fl.refl(host)
    c0 = np.array([fl.x, fl.y, fl.z])
    r = fl.size / 2
    if fl.kind in ("fbh", "disc"):
        ry = r * (fl.aspect if fl.kind == "disc" else 1.0)
        h = max(spacing, np.sqrt(np.pi * r * ry / max_patches))
        u = np.arange(-r + h / 2, r, h)
        v = np.arange(-ry + h / 2, ry, h)
        if len(u) == 0:
            u = np.array([0.0])
        if len(v) == 0:
            v = np.array([0.0])
        U, V = np.meshgrid(u, v, indexing="ij")
        inside = (U / r) ** 2 + (V / ry) ** 2 <= 1 + 1e-12
        if not inside.any():
            inside[len(u) // 2, len(v) // 2] = True
        U, V = U[inside], V[inside]
        dA = np.full(U.shape, np.pi * r * ry / U.size)
        b = np.radians(fl.tilt)
        ex = np.array([np.cos(b), 0, np.sin(b)])          # in-plane axis (tilted)
        ey = np.array([0, 1.0, 0])
        n = np.array([np.sin(b), 0, -np.cos(b)])          # faces the scan surface at tilt 0
        pos = c0 + U[:, None] * ex + V[:, None] * ey
        return Patches(pos, np.tile(n, (len(dA), 1)), dA, np.full(len(dA), R),
                       np.full(len(dA), fl.kind == "disc"))
    if fl.kind == "sdh":
        L = fl.length
        nphi = max(9, int(np.ceil(np.pi * r / spacing)))
        ny = max(1, int(np.ceil(L / spacing)))
        while nphi * ny > max_patches and ny > 1:
            ny = max(1, ny // 2)
        phi = (np.arange(nphi) + 0.5) / nphi * np.pi - np.pi / 2
        yy = (np.arange(ny) + 0.5) / ny * L - L / 2
        PH, YY = np.meshgrid(phi, yy, indexing="ij")
        n = np.stack([np.sin(PH), np.zeros_like(PH), -np.cos(PH)], -1).reshape(-1, 3)
        pos = c0 + r * n + np.stack([np.zeros(YY.size), YY.ravel(), np.zeros(YY.size)], -1)
        dA = np.full(len(n), r * np.pi / nphi * L / ny)
        return Patches(pos, n, dA, np.full(len(dA), R), np.zeros(len(dA), bool))
    if fl.kind == "sphere":
        nth = max(6, int(np.ceil(np.pi * r / 2 / spacing)))
        th = (np.arange(nth) + 0.5) / nth * np.pi / 2          # polar angle from −z
        pos, nrm, dA = [], [], []
        for t in th:
            nph = max(4, int(np.ceil(2 * np.pi * r * np.sin(t) / spacing)))
            ph = (np.arange(nph) + 0.5) / nph * 2 * np.pi
            n = np.stack([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph),
                          -np.cos(t) * np.ones_like(ph)], -1)
            nrm.append(n); pos.append(c0 + r * n)
            dA.append(np.full(nph, 2 * np.pi * r ** 2 * np.sin(t) * (np.pi / 2 / nth) / nph))
        nrm, pos, dA = np.concatenate(nrm), np.concatenate(pos), np.concatenate(dA)
        return Patches(pos, nrm, dA, np.full(len(dA), R), np.zeros(len(dA), bool))
    if fl.kind == "point":
        return Patches(c0[None], np.array([[0, 0, -1.0]]), np.array([np.pi * r ** 2]),
                       np.array([R]), np.array([True]))
    raise ValueError(f"unknown flaw kind {fl.kind!r}")


# =========================================================================== pulse echo
@dataclass
class Probe:
    f0: float = 5e6
    bw: float = 0.6
    diameter: float = 10e-3


@dataclass
class UTSetup:
    material: str = "steel"
    thickness: float = 50e-3
    probe: Probe = field(default_factory=Probe)
    mode: str = "contact"              # 'contact' | 'immersion'
    water_path: float = 20e-3
    fs: float = 100e6
    t_max: float | None = None
    noise: float = 3e-5                # electronic noise rms (units of p/p₀)
    grain_noise: bool = True
    grain_scale: float = 1.0
    dispersive: bool = True
    multiples: int = 4
    seed: int = 0

    @property
    def mat(self):
        return get(self.material)

    def lam(self):
        return self.mat.cL / self.probe.f0

    def t0(self):
        """Time of the entry-surface echo / main bang."""
        bw = self.probe.bw * self.probe.f0 / (2 * np.sqrt(2 * np.log(2)))
        delay = 4.0 / (2 * np.pi * bw)
        if self.mode == "immersion":
            return delay + 2 * self.water_path / get("water").cL
        return delay

    def n_samples(self):
        tm = self.t_max or (self.t0() + 2.2 * self.thickness / self.mat.cL * 2)
        return int(np.ceil(tm * self.fs / 2) * 2)

    def time(self):
        return np.arange(self.n_samples()) / self.fs


def _patch_spectra(setup, patches, pos, fb, n_bands=8, chunk=1_500_000):
    """Coherent sum of all patch echoes for transducer positions pos (P,2) → (P,F), plus the
    back-wall shadow factor (P,).

    The smooth part of each patch response A_q(ω) (Kirchhoff amplitude, diffraction,
    attenuation, dispersion) is evaluated at `n_bands` frequencies and interpolated
    piecewise-linearly; the bulk delay e^{−iω τ_q} is applied exactly by splatting each patch
    into a delay-line (linear interpolation, compensated by 1/sinc²) and one FFT per band. Cost
    O(P·Q·B + P·B·n log n) instead of O(P·Q·F)."""
    m = setup.mat
    a = setup.probe.diameter / 2
    P, Q, F = len(pos), len(patches.area), len(fb)
    out = np.zeros((P, F), complex)
    shadow = np.zeros(P)
    if Q == 0:
        return out, np.ones(P)
    imm = setup.mode == "immersion"
    w = get("water")
    fc = setup.probe.f0
    cd = w.cL if imm else m.cL
    kc = 2 * np.pi * fc / cd
    Tt = 1.0
    if imm:
        Tt = normal_incidence(w.ZL, m.ZL)["T"] * normal_incidence(m.ZL, w.ZL)["T"]
    Nf = max(near_field_length(setup.probe.diameter, fc, cd), 1e-6)
    fbands = np.linspace(fb[0], fb[-1], n_bands)
    df = fbands[1] - fbands[0]
    Wb = np.clip(1 - np.abs(fb[None, :] - fbands[:, None]) / df, 0, None)       # (B,F)
    kd = 2 * np.pi * fbands / cd
    # the delay line may be sampled more coarsely than the traces (band-limited content)
    dec = 1
    while fb[-1] / (setup.fs / (dec * 2)) <= 0.22 and setup.n_samples() % (dec * 2) == 0:
        dec *= 2
    fs = setup.fs / dec
    n = setup.n_samples() // dec
    fgrid = np.fft.rfftfreq(n, 1 / fs)
    bins = np.searchsorted(fgrid, fb - 1e-6)
    comp = 1.0 / np.sinc(fb / fs) ** 2
    step = max(1, chunk // max(1, Q * n_bands))
    for i in range(0, P, step):
        sl = slice(i, i + step)
        p = len(pos[sl])
        p3 = np.zeros((p, 3)); p3[:, :2] = pos[sl]
        r = patches.pos[None] - p3[:, None]                     # (p,Q,3)
        R = np.linalg.norm(r, axis=-1)
        rh = r / R[..., None]
        cos_t = np.clip(rh[..., 2], -1, 1)
        sin_t = np.sqrt(1 - cos_t ** 2)
        ndot = np.einsum("pqk,qk->pq", rh, patches.nrm)
        obl = np.where(patches.two_sided[None], np.abs(ndot), np.clip(-ndot, 0, None))
        if imm:
            Reff = setup.water_path + R * m.cL / w.cL
            sin_w = np.clip(sin_t * w.cL / m.cL, 0, 1)
        else:
            Reff, sin_w = R, sin_t
        # near-field beam width: the far-field directivity is evaluated at an effective range
        # R_d = k a w(d)/1.6 with one-way −3 dB half-width w = a(1 − 0.745 (d/N)²) for d < N,
        # so the beam is crystal-wide at the probe, narrowest (−6 dB Ø ≈ 0.26 D) at N and
        # spreads as λ/D beyond (R_d = d)
        rho = R * sin_t                                         # true lateral offset
        d_ax = (setup.water_path + R * cos_t * m.cL / w.cL) if imm else R * cos_t
        Rd = np.where(d_ax < Nf, kc * a * a * (1 - 0.745 * (d_ax / Nf) ** 2) / 1.6, d_ax)
        sin_w = rho / np.sqrt(rho ** 2 + Rd ** 2)
        amp = patches.refl[None] * obl * patches.area[None] / (np.pi * a ** 2) * Tt   # (p,Q)
        Re = Reff[..., None]
        near = 1 - np.exp(-1j * kd * (np.sqrt(Re ** 2 + a ** 2) - Re))           # (p,Q,B)
        D = piston_directivity(kd * a * sin_w[..., None])
        tau = 2 * R / m.cL
        smooth = sg.propagator(m, fbands, 2 * R[..., None], fc, dispersive=setup.dispersive) \
            * np.exp(2j * np.pi * fbands * tau[..., None])                         # remove bulk delay
        A = amp[..., None] * near ** 2 * D ** 2 * smooth                           # (p,Q,B)
        # delay-line splatting
        s = tau * fs
        k0 = np.floor(s).astype(np.int64)
        fr = s - k0
        ok = (k0 >= 0) & (k0 < n - 1)
        pidx = np.broadcast_to(np.arange(p)[:, None] * n, k0.shape)
        i0 = (pidx + k0)[ok]
        w0, w1 = (1 - fr)[ok], fr[ok]
        H = np.empty((p, n_bands, n), complex)
        for b in range(n_bands):
            v = A[..., b][ok]
            re = np.bincount(i0, v.real * w0, minlength=p * n) \
                + np.bincount(i0 + 1, v.real * w1, minlength=p * n)
            im = np.bincount(i0, v.imag * w0, minlength=p * n) \
                + np.bincount(i0 + 1, v.imag * w1, minlength=p * n)
            H[:, b] = (re + 1j * im).reshape(p, n)
        # FFT of a complex delay line: positive-frequency bins carry Σ A e^{−iωτ}
        Hf = sfft.fft(H, axis=2, workers=-1, overwrite_x=True)[..., bins]
        out[sl] = np.einsum("pbf,bf->pf", Hf, Wb) * comp[None]
        # back-wall shadowing by the flaws' projected area, weighted by the beam profile
        Dc = piston_directivity(kc * a * sin_w)
        beam_area = np.pi * a ** 2 + 4 * np.pi * Reff ** 2 / (kc * a) ** 2
        illum = np.where(patches.two_sided[None], np.abs(ndot) * np.abs(patches.nrm[None, :, 2]),
                         np.clip(-ndot, 0, None) * np.clip(-patches.nrm[None, :, 2], 0, None))
        shadow[sl] = np.sum(patches.area[None] * illum * Dc ** 2 / beam_area, axis=1)
    return out, np.clip(1 - shadow, 0, 1)


def pulse_echo(setup: UTSetup, flaws=(), positions=None, max_patches=900):
    """Simulate pulse-echo RF traces for transducer positions (P,2) [x,y] on the surface.
    Returns (t, traces[P, n]) in units of p/p₀ (back wall of an infinite plate at the
    far-field = πa²/(λ·2T))."""
    m = setup.mat
    pr = setup.probe
    pos = np.zeros((1, 2)) if positions is None else np.atleast_2d(np.asarray(positions, float))
    n = setup.n_samples()
    t = np.arange(n) / setup.fs
    f, idx = sg.band(pr.f0, pr.bw, setup.fs, n)
    fb = f[idx]
    Pw = sg.gaussian_spectrum(fb, pr.f0, pr.bw)
    t0 = setup.t0()
    shift = np.exp(-2j * np.pi * fb * t0)
    lam = m.cL / pr.f0
    patches = Patches.concat([flaw_patches(fl, m, lam / 3, max_patches) for fl in flaws])
    spec, shadow = _patch_spectra(setup, patches, pos, fb)
    a = pr.diameter / 2
    imm = setup.mode == "immersion"
    w = get("water")
    k_m = 2 * np.pi * fb / m.cL
    # ---- back wall (and its multiples)
    if imm:
        Rbw = normal_incidence(m.ZL, w.ZL)["R"]
        Tt = normal_incidence(w.ZL, m.ZL)["T"] * normal_incidence(m.ZL, w.ZL)["T"]
        Rin = normal_incidence(m.ZL, w.ZL)["R"]
        kw = 2 * np.pi * fb / w.cL
    else:
        Rbw, Tt, Rin = -1.0, 1.0, -0.95
    bw = np.zeros(len(fb), complex)
    t_last = n / setup.fs - 6.0 / (pr.bw * pr.f0)        # later echoes would wrap around
    for j in range(1, setup.multiples + 1):
        T2 = 2 * j * setup.thickness
        if t0 + T2 / m.cL > t_last:
            break
        if imm:
            dif = seki(2 * setup.water_path + T2 * m.cL / w.cL, a, kw)
        else:
            dif = seki(T2, a, k_m)
        bw += Rbw ** j * Rin ** (j - 1) * Tt * dif * sg.propagator(
            m, fb, T2, pr.f0, dispersive=setup.dispersive)
    spec = spec + shadow[:, None] * bw[None]
    # ---- entry surface
    if imm:
        Rf = normal_incidence(w.ZL, m.ZL)["R"]
        front = Rf * seki(2 * setup.water_path, a, kw)
        surf = front
        if t0 + 2 * setup.water_path / w.cL < t_last:      # second interface echo
            surf = surf + Rf ** 2 * seki(4 * setup.water_path, a, kw) * np.exp(
                -2j * np.pi * fb * 2 * setup.water_path / w.cL)
    else:
        # main bang: transducer ring-down (damped resonance, Q≈2) reaching full screen
        Qf = 2.0
        surf = 1.5 / (1 + 1j * Qf * (fb / pr.f0 - pr.f0 / np.maximum(fb, 1)))
    spec_total = (spec + surf[None]) * Pw[None] * shift[None]
    traces = sg.irfft_band(spec_total, idx, n)
    # normalise: band-limited spectrum → unit-peak pulse
    ref = sg.irfft_band((Pw * shift)[None], idx, n)[0]
    traces /= np.max(np.abs(ref))
    rng = np.random.default_rng(setup.seed)
    if setup.grain_noise and m.grain_um > 0:
        traces += grain_noise(setup, len(pos), t, idx, fb, Pw, rng, positions=pos)
    if setup.noise > 0:
        traces += setup.noise * rng.standard_normal(traces.shape)
    return t, traces


def grain_noise(setup, P, t, idx, fb, Pw, rng, positions=None):
    """Coloured Rayleigh backscatter noise inside the part. rms ∝ (d/λ)^{3/2}·grain_scale,
    spectrum ∝ f² P(f), attenuated two-way, laterally correlated over the beam width."""
    m = setup.mat
    pr = setup.probe
    n = len(t)
    lam = m.cL / pr.f0
    G = 0.1 * setup.grain_scale * (m.grain_um * 1e-6 / lam) ** 1.5
    W = rng.standard_normal((P, len(fb))) + 1j * rng.standard_normal((P, len(fb)))
    if positions is not None and P > 1:
        pos = np.asarray(positions)
        beam = 0.5 * lam / pr.diameter * setup.thickness + pr.diameter / 4
        _smooth_lateral(W, pos, beam)
    spec = W * Pw * (fb / pr.f0) ** 2
    x = sg.irfft_band(spec, idx, n)
    x /= max(np.std(x), 1e-300)
    tf = setup.t0()
    tb = tf + 2 * setup.thickness / m.cL
    inside = (t > tf + 0.5 / pr.f0) & (t < tb)
    z = np.clip(t - tf, 0, None) * m.cL / 2
    env = np.exp(-m.alpha(pr.f0) * m.cL * np.clip(t - tf, 0, None))
    if setup.mode == "immersion":
        w = get("water")
        Tt = normal_incidence(w.ZL, m.ZL)["T"] * normal_incidence(m.ZL, w.ZL)["T"]
        N = near_field_length(pr.diameter, pr.f0, w.cL)
        zeq = setup.water_path + z * m.cL / w.cL
    else:
        Tt, N, zeq = 1.0, near_field_length(pr.diameter, pr.f0, m.cL), z
    env = env * Tt / np.sqrt(1 + (zeq / max(N, 1e-6)) ** 2)
    return G * x * (inside * env)[None]


def _smooth_lateral(W, pos, width):
    """Correlate the noise spectra of neighbouring positions (regular 1-D or 2-D grids)."""
    xs, ys = np.unique(pos[:, 0]), np.unique(pos[:, 1])
    if len(xs) * len(ys) == len(pos) and len(xs) > 1 and len(ys) > 1:
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        shp = (len(ys), len(xs))
        order = np.lexsort((pos[:, 0], pos[:, 1]))
        for part in (W.real, W.imag):
            g = part[order].reshape(shp + (W.shape[1],))
            g = gaussian_filter(g, (width / dy / 2, width / dx / 2, 0), mode="nearest")
            part[order] = g.reshape(len(pos), -1)
    elif len(xs) == len(pos) and len(xs) > 1:
        dx = xs[1] - xs[0]
        order = np.argsort(pos[:, 0])
        for part in (W.real, W.imag):
            part[order] = gaussian_filter(part[order], (width / dx / 2, 0), mode="nearest")
    W /= max(np.std(W) / np.sqrt(2), 1e-300)


def ascan(setup, flaws=(), x=0.0, y=0.0):
    t, tr = pulse_echo(setup, flaws, [[x, y]])
    return t, tr[0]


def bscan(setup, flaws, xs, y=0.0):
    pos = np.stack([xs, np.full(len(xs), y)], -1)
    return pulse_echo(setup, flaws, pos)


def cscan(setup, flaws, xs, ys, gate=None, max_patches=400):
    """Raster C-scan. Returns dict(t, amp[ny,nx], tof[ny,nx], traces) where amp is the peak
    envelope in the gate (default: between entry and back wall) and tof its arrival time."""
    X, Y = np.meshgrid(xs, ys)
    pos = np.stack([X.ravel(), Y.ravel()], -1)
    t, tr = pulse_echo(setup, flaws, pos, max_patches=max_patches)
    env = sg.envelope(tr)
    g0, g1 = gate or default_gate(setup)
    sel = (t >= g0) & (t <= g1)
    e = env[:, sel]
    k = np.argmax(e, axis=1)
    amp = e[np.arange(len(e)), k].reshape(X.shape)
    tof = t[sel][k].reshape(X.shape)
    return dict(t=t, amp=amp, tof=tof, depth=(tof - setup.t0()) * setup.mat.cL / 2,
                traces=tr, env=env, X=X, Y=Y)


def default_gate(setup):
    """Gate from the end of the entry echo (contact: main-bang ring-down, ≈ 40 dB decay of the
    Q = 2 resonance) to just before the back wall."""
    tf = setup.t0()
    m = setup.mat
    pw = 2.5 / (setup.probe.bw * setup.probe.f0)
    start = tf + (pw + 3.0 / setup.probe.f0 if setup.mode == "contact" else pw)
    return start, tf + 2 * setup.thickness / m.cL - pw


def depth_axis(setup, t):
    return (t - setup.t0()) * setup.mat.cL / 2


# =========================================================================== DGS / sizing
def dgs_curves(setup: UTSetup, z, fbh_diams=(1e-3, 2e-3, 4e-3, 8e-3), attenuation=True):
    """Distance–gain–size diagram at the centre frequency: back wall |D(2z)| and flat-bottom
    holes p_axis(z)²(b/a)², optionally with two-way attenuation. Returns dict of dB values
    relative to the back wall at the near-field distance."""
    m, pr = setup.mat, setup.probe
    k = 2 * np.pi * pr.f0 / m.cL
    a = pr.diameter / 2
    z = np.asarray(z, float)
    att = np.exp(-2 * m.alpha(pr.f0) * z) if attenuation else 1.0
    out = {"BW": np.abs(seki(2 * z, a, k)) * att}
    for d in fbh_diams:
        out[f"FBH {d * 1e3:g} mm"] = on_axis_pressure(z, a, k) ** 2 * (d / 2 / a) ** 2 * att
    N = near_field_length(pr.diameter, pr.f0, m.cL)
    ref = np.abs(seki(2 * N, a, k))
    return {key: sg.db(v, ref=ref, floor=-200) for key, v in out.items()} | {"N": N}


def ers_from_echo(setup: UTSetup, z_flaw, ratio_to_bw, thickness=None):
    """Equivalent reflector size (FBH diameter) from the flaw/back-wall echo amplitude ratio,
    inverting p_FBH = p_axis(z)²(b/a)² with attenuation correction."""
    m, pr = setup.mat, setup.probe
    T = thickness or setup.thickness
    k = 2 * np.pi * pr.f0 / m.cL
    a = pr.diameter / 2
    pbw = np.abs(seki(2 * T, a, k)) * np.exp(-2 * m.alpha(pr.f0) * T)
    pax = on_axis_pressure(z_flaw, a, k) ** 2 * np.exp(-2 * m.alpha(pr.f0) * z_flaw)
    b = a * np.sqrt(ratio_to_bw * pbw / pax)
    return 2 * b


def six_db_drop(x, amp, drop_db=6.0):
    """Length over which amp ≥ max·10^(−drop/20), with linear interpolation at both edges."""
    x, amp = np.asarray(x, float), np.asarray(amp, float)
    i = int(np.argmax(amp))
    thr = amp[i] * 10 ** (-drop_db / 20)
    lo = i
    while lo > 0 and amp[lo - 1] >= thr:
        lo -= 1
    hi = i
    while hi < len(amp) - 1 and amp[hi + 1] >= thr:
        hi += 1

    def cross(i0, i1):
        a0, a1 = amp[i0], amp[i1]
        if a0 == a1:
            return x[i0]
        return x[i0] + (thr - a0) / (a1 - a0) * (x[i1] - x[i0])
    xl = cross(lo - 1, lo) if lo > 0 else x[0]
    xr = cross(hi, hi + 1) if hi < len(amp) - 1 else x[-1]
    return float(xr - xl), float(xl), float(xr)


def dac_curve(setup: UTSetup, depths, sdh_diam=3e-3):
    """Distance–amplitude correction from side-drilled holes at the given depths (peak envelope
    amplitude of each SDH echo on axis, noise-free)."""
    s = replace(setup, noise=0.0, grain_noise=False, multiples=1)
    amps = []
    for d in depths:
        t, tr = ascan(s, [Flaw("sdh", z=d, size=sdh_diam, length=0.04)])
        env = sg.envelope(tr)
        tz = s.t0() + 2 * (d - sdh_diam / 2) / s.mat.cL
        win = np.abs(t - tz) < 2.5 / s.probe.f0
        amps.append(env[win].max())
    return np.asarray(depths), np.asarray(amps)


def tcg_gain(setup, t, depths, amps):
    """Time-corrected gain (linear) that flattens the DAC curve to its maximum."""
    z = depth_axis(setup, t)
    a = np.interp(z, depths, amps, left=amps[0], right=amps[-1])
    return np.max(amps) / a


# =========================================================================== SAFT
def saft(t, traces, xs, c, t0, x_img, z_img, aperture_deg=30.0):
    """Synthetic aperture focusing of a monostatic B-scan: delay-and-sum of the analytic
    signals along the hyperbola t = t₀ + 2√((x−xᵢ)² + z²)/c inside the beam aperture."""
    an = sg.analytic(traces)
    X, Z = np.meshgrid(x_img, z_img)
    out = np.zeros(X.shape, complex)
    dt = t[1] - t[0]
    tan_a = np.tan(np.radians(aperture_deg))
    for i, xi in enumerate(xs):
        d = np.sqrt((X - xi) ** 2 + Z ** 2)
        inside = np.abs(X - xi) <= Z * tan_a
        s = (t0 + 2 * d / c - t[0]) / dt
        k = np.clip(s.astype(int), 0, len(t) - 2)
        fr = s - k
        v = an[i, k] * (1 - fr) + an[i, k + 1] * fr
        out += np.where(inside & (s < len(t) - 1), v, 0)
    return np.abs(out)


# =========================================================================== TOFD
@dataclass
class TOFDSetup:
    material: str = "steel"
    thickness: float = 25e-3
    pcs: float = 60e-3               # probe centre separation 2S
    angle: float = 60.0              # nominal refraction angle (deg)
    f0: float = 5e6
    bw: float = 0.7
    crystal: float = 6e-3
    fs: float = 100e6
    noise: float = 3e-3
    seed: int = 0

    @property
    def mat(self):
        return get(self.material)

    def t_lateral(self):
        return self.pcs / self.mat.cL

    def t_backwall(self):
        S = self.pcs / 2
        return 2 * np.sqrt(S ** 2 + self.thickness ** 2) / self.mat.cL


@dataclass
class Crack:
    """Planar crack perpendicular to the scan surface (weld axis along y).
    top/bottom depths (m); top = 0 → surface breaking, bottom = thickness → back-wall breaking."""
    x: float = 0.0
    y0: float = -10e-3
    y1: float = 10e-3
    top: float = 8e-3
    bottom: float = 12e-3


def tofd_depth(t, setup: TOFDSetup):
    """Depth from arrival time (t measured from the lateral-wave origin, i.e. probe delay
    removed): d = √((c t/2)² − S²)."""
    S = setup.pcs / 2
    c = setup.mat.cL
    return np.sqrt(np.clip((c * np.asarray(t) / 2) ** 2 - S ** 2, 0, None))


def tofd_time(d, setup: TOFDSetup, x_off=0.0):
    S = setup.pcs / 2
    c = setup.mat.cL
    return (np.sqrt((S - x_off) ** 2 + d ** 2) + np.sqrt((S + x_off) ** 2 + d ** 2)) / c


def tofd_scan(setup: TOFDSetup, cracks, ys=None, xs=None, n=None):
    """TOFD D-scan (probes straddle the weld at x = 0 and move along y; `ys` given) or B-scan
    (probes move across the weld; `xs` given). Returns dict(t, data[P,n], pos)."""
    m = setup.mat
    c = m.cL
    S = setup.pcs / 2
    T = setup.thickness
    tm = 1.25 * setup.t_backwall() + 1.0e-6
    n = n or int(np.ceil(tm * setup.fs / 2) * 2)
    t = np.arange(n) / setup.fs
    f, idx = sg.band(setup.f0, setup.bw, setup.fs, n)
    fb = f[idx]
    Pw = sg.gaussian_spectrum(fb, setup.f0, setup.bw)
    t_delay = 0.5e-6
    if ys is not None:
        pos = np.stack([np.zeros(len(ys)), ys], -1)
    else:
        pos = np.stack([xs, np.zeros(len(xs))], -1)
    lam = c / setup.f0
    th0 = np.radians(setup.angle)
    spread = max(np.arcsin(min(1.0, 0.51 * lam / setup.crystal * 1.5)), np.radians(12))
    beam = lambda ang: np.exp(-0.5 * ((ang - th0) / spread) ** 2)
    yw = setup.crystal * 1.2 + lam            # lateral (y) beam half-width at the weld

    def add(spec, path_t, amp):
        return spec + amp[:, None] * np.exp(-2j * np.pi * fb[None] * (path_t[:, None] + t_delay))

    spec = np.zeros((len(pos), len(fb)), complex)
    P = len(pos)
    ones = np.ones(P)
    # lateral wave (+) and back wall (−): attenuated by surface / root breaking cracks
    lat_block, bw_block = np.zeros(P), np.zeros(P)
    tips = []
    for cr in cracks:
        ny = max(2, int(np.ceil((cr.y1 - cr.y0) / (lam / 2))))
        yy = np.linspace(cr.y0, cr.y1, ny)
        for yv in yy:
            wy = np.exp(-0.5 * ((pos[:, 1] - yv) / yw) ** 2) * (cr.y1 - cr.y0) / ny / (
                np.sqrt(2 * np.pi) * yw)
            xo = pos[:, 0] - cr.x
            if cr.top <= 1e-6:
                lat_block += wy * np.exp(-0.5 * (xo / (S * 0.5)) ** 2) * 2.5
            else:
                tips.append((xo, cr.top, wy, -1.0))
            if cr.bottom >= T - 1e-6:
                bw_block += wy * np.exp(-0.5 * (xo / (S * 0.5)) ** 2) * 2.5
            else:
                tips.append((xo, cr.bottom, wy, +1.0))
    # lateral wave: creeping along the surface, weak because it leaves the probes at grazing
    spec = add(spec, 2 * S / c * ones, 0.3 * np.clip(1 - lat_block, 0, 1))
    r_bw = 2 * np.sqrt(S ** 2 + T ** 2) * ones          # flat back wall: independent of x
    ang_bw = np.arctan2(S, T)
    spec = add(spec, r_bw / c, -np.clip(1 - bw_block, 0, 1) * beam(ang_bw) * (2 * T / r_bw))
    for xo, d, wy, sgn in tips:
        r1 = np.sqrt((S + xo) ** 2 + d ** 2)        # transmitter at x = −S
        r2 = np.sqrt((S - xo) ** 2 + d ** 2)
        a1, a2 = np.arctan2(S + xo, d), np.arctan2(S - xo, d)
        amp = sgn * 0.12 * beam(a1) * beam(a2) * np.sqrt(2 * T / (r1 + r2)) * wy * 6
        spec = add(spec, (r1 + r2) / c, amp)
    spec *= Pw[None]
    data = sg.irfft_band(spec, idx, n)
    ref = sg.irfft_band((Pw * np.exp(-2j * np.pi * fb * t_delay))[None], idx, n)[0]
    data /= np.max(np.abs(ref))
    rng = np.random.default_rng(setup.seed)
    data += setup.noise * rng.standard_normal(data.shape)
    return dict(t=t, data=data, pos=pos, t_delay=t_delay,
                t_lat=setup.t_lateral() + t_delay, t_bw=setup.t_backwall() + t_delay)


def tofd_size_ascan(t, ascan, setup: TOFDSetup, t_delay=0.5e-6, min_sep=None):
    """Pick the two strongest diffraction signals between the lateral wave and the back wall
    and return (top depth, bottom depth, height) via d = √((ct/2)² − S²)."""
    env = sg.envelope(ascan)
    tl = setup.t_lateral() + t_delay + 1.5 / setup.f0
    tb = setup.t_backwall() + t_delay - 1.5 / setup.f0
    sel = np.nonzero((t > tl) & (t < tb))[0]
    if len(sel) < 3:
        return None
    e = env[sel]
    min_sep = min_sep or int(1.2 / setup.f0 * setup.fs)
    i1 = int(np.argmax(e))
    e2 = e.copy()
    e2[max(0, i1 - min_sep):i1 + min_sep] = 0
    i2 = int(np.argmax(e2))
    ts = sorted([t[sel][i1], t[sel][i2]])
    d = tofd_depth(np.array(ts) - t_delay, setup)
    return float(d[0]), float(d[1]), float(d[1] - d[0])
