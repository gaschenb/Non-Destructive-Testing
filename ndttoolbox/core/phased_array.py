"""Linear phased arrays in 2-D (x–z plane): delay laws, beam fields, full-matrix capture
simulation and imaging (TFM, phase-coherence weighting, plane-wave imaging, sectorial scans).

Ray model
---------
Element e sits at (x_e, z_e) (z < 0 above the part for wedges and immersion). The travel time
to a point P in the part is found from Fermat's principle on the flat entry surface z = 0,
    t(ξ) = √((ξ−x_e)² + z_e²)/c₁ + √((x−ξ)² + z²)/c₂ ,   ∂t/∂ξ = 0  (Snell's law),
solved by safeguarded Newton iterations for all (element, point) pairs at once. The 2-D ray
amplitude is the element directivity sinc(k w sinθ/2)·cosθ times the interface energy
transmission and the geometric spreading 1/√L with L = d₁ + d₂ (c₂/c₁)(cos²θ₁/cos²θ₂).

Scatterers are points (side-drilled holes), dense point chains (cracks, back wall — the
coherent sum reproduces specular reflection and tip diffraction) and random grain scatterers.
FMC signals are assembled per frequency as S(ω) = E(ω) diag(A) E(ω)ᵀ with
E_eq = amplitude · e^{−iω t_eq − α d_eq}, i.e. one batched matrix product.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.fft as sfft

from . import signals as sg
from .materials import get
from .ultrasonics import fluid_solid, solid_solid


@dataclass
class ArraySetup:
    material: str = "steel"
    thickness: float = 40e-3
    n_el: int = 32
    pitch: float = 0.6e-3
    width: float = 0.5e-3
    f0: float = 5e6
    bw: float = 0.6
    fs: float = 50e6
    coupling: str = "contact"          # 'contact' | 'wedge' | 'immersion'
    wedge_material: str = "rexolite"
    wedge_angle: float = 36.0          # degrees
    wedge_height: float = 8e-3         # height of the first element above the part
    water_path: float = 15e-3
    x0: float = 0.0                    # array centre (contact/immersion) or first element (wedge)
    t_max: float | None = None
    noise: float = 3e-3
    grain_scale: float = 1.0
    n_grains: int = 600
    seed: int = 0

    @property
    def mat(self):
        return get(self.material)

    @property
    def c(self):
        return self.mat.cL

    @property
    def medium1(self):
        if self.coupling == "wedge":
            return get(self.wedge_material)
        if self.coupling == "immersion":
            return get("water")
        return None

    def elements(self):
        """Element centres (N,2) and the element normal angle (rad, from +z)."""
        i = np.arange(self.n_el)
        if self.coupling == "wedge":
            w = np.radians(self.wedge_angle)
            x = self.x0 + i * self.pitch * np.cos(w)
            z = -(self.wedge_height + i * self.pitch * np.sin(w))
            return np.stack([x, z], -1), w
        xs = self.x0 + (i - (self.n_el - 1) / 2) * self.pitch
        z = np.full(self.n_el, -self.water_path if self.coupling == "immersion" else 0.0)
        return np.stack([xs, z], -1), 0.0

    def aperture(self):
        return (self.n_el - 1) * self.pitch + self.width

    def n_samples(self):
        if self.t_max is None:
            el, _ = self.elements()
            m1 = self.medium1
            t1 = 0 if m1 is None else 2 * np.max(-el[:, 1]) / m1.cL * 1.6
            tm = t1 + 2.6 * self.thickness / self.c + 6 / (self.bw * self.f0)
        else:
            tm = self.t_max
        return int(np.ceil(tm * self.fs / 2) * 2)


# =========================================================================== rays
def ray_paths(setup: ArraySetup, px, pz, iters=12):
    """Travel time, amplitude and in-part path length from every element to every point.
    px, pz: (G,) points in the part (z > 0). Returns dict(t (N,G), amp (N,G), d2 (N,G),
    theta2 (N,G) refraction angle, theta_el (N,G) angle at the element)."""
    el, nrm = setup.elements()
    xe, ze = el[:, 0:1], el[:, 1:2]
    px, pz = np.asarray(px, float)[None], np.asarray(pz, float)[None]
    c2 = setup.c
    k = 2 * np.pi * setup.f0
    m1 = setup.medium1
    if m1 is None:
        dx, dz = px - xe, pz - ze
        d = np.hypot(dx, dz)
        th = np.arctan2(dx, dz)
        L = d
        T = np.ones_like(d)
        th_el = th - nrm
        return dict(t=d / c2, amp=_directivity(setup, th_el, c2) * T / np.sqrt(np.maximum(L, 1e-6)),
                    d2=d, theta2=th, theta_el=th_el, xi=None)
    c1 = m1.cL
    h = -ze                                      # height above the surface (>0)
    # Newton on g(ξ) = (ξ−xe)/(c1 d1) − (x−ξ)/(c2 d2) = 0, bracket [min, max]
    lo = np.minimum(xe, px) * np.ones_like(px * xe)
    hi = np.maximum(xe, px) * np.ones_like(px * xe)
    xi = xe + (px - xe) * h / (h + pz * c1 / c2)          # paraxial start
    for _ in range(iters):
        d1 = np.hypot(xi - xe, h)
        d2 = np.hypot(px - xi, pz)
        g = (xi - xe) / (c1 * d1) - (px - xi) / (c2 * d2)
        gp = h ** 2 / (c1 * d1 ** 3) + pz ** 2 / (c2 * d2 ** 3)
        lo = np.where(g < 0, xi, lo)
        hi = np.where(g > 0, xi, hi)
        xn = xi - g / gp
        bad = (xn <= lo) | (xn >= hi)
        xi = np.where(bad, 0.5 * (lo + hi), xn)
    d1 = np.hypot(xi - xe, h)
    d2 = np.hypot(px - xi, pz)
    t = d1 / c1 + d2 / c2
    th1 = np.arctan2(xi - xe, h)
    th2 = np.arctan2(px - xi, pz)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = d1 + d2 * (c2 / c1) * np.cos(th1) ** 2 / np.maximum(np.cos(th2) ** 2, 1e-6)
    Tx = _transmission(setup, np.abs(th1))
    th_el = th1 - nrm
    amp = _directivity(setup, th_el, c1) * Tx / np.sqrt(np.maximum(L, 1e-6))
    return dict(t=t, amp=amp, d2=d2, theta2=th2, theta_el=th_el, xi=xi)


def _directivity(setup, th, c):
    k = 2 * np.pi * setup.f0 / c
    return np.sinc(k * setup.width * np.sin(th) / (2 * np.pi)) * np.clip(np.cos(th), 0, None)


_T_CACHE = {}


def _transmission(setup, th1):
    """Amplitude (√energy) transmission into the part L wave vs incidence angle, tabulated."""
    key = (setup.coupling, setup.wedge_material, setup.material)
    if key not in _T_CACHE:
        ang = np.linspace(0, np.pi / 2 * 0.999, 400)
        if setup.coupling == "immersion":
            e = fluid_solid(ang, "water", setup.material)["EL"]
        else:
            e = np.real(solid_solid(ang, setup.wedge_material, setup.material)["ETp"])
        _T_CACHE[key] = (ang, np.sqrt(np.clip(e, 0, None)))
    ang, tv = _T_CACHE[key]
    return np.interp(th1, ang, tv)


# =========================================================================== delay laws
def delay_law(setup: ArraySetup, angle_deg=0.0, focus=None):
    """Transmit delays (s, ≥ 0) steering to `angle_deg` in the part (refracted angle) and
    focusing at depth `focus` along that direction (None: plane-wave steering).
    The beam origin is the array centre projected onto the surface."""
    el, _ = setup.elements()
    th = np.radians(angle_deg)
    x_ref = _beam_origin(setup)
    if focus is None:
        # far focus → plane wave: focus very deep
        F = 50.0
    else:
        F = focus
    fx, fz = x_ref + F * np.sin(th), F * np.cos(th)
    r = ray_paths(setup, np.array([fx]), np.array([fz]))
    t = r["t"][:, 0]
    return t.max() - t


def _beam_origin(setup):
    el, _ = setup.elements()
    if setup.coupling == "contact":
        return float(np.mean(el[:, 0]))
    # where the central ray of the wedge/immersion array meets the surface
    m1 = setup.medium1
    xc, zc = np.mean(el, axis=0)
    if setup.coupling == "wedge":
        w = np.radians(setup.wedge_angle)
        return float(xc + (-zc) * np.tan(w))
    return float(xc)


def grating_lobe_angles(pitch, f, c, steer_deg=0.0):
    """Grating-lobe directions sinθ_g = sinθ_s ± mλ/p that exist (|sinθ_g| ≤ 1), degrees."""
    lam = c / f
    s0 = np.sin(np.radians(steer_deg))
    out = []
    for m in range(1, 6):
        for sgn in (-1, 1):
            s = s0 + sgn * m * lam / pitch
            if abs(s) <= 1:
                out.append(float(np.degrees(np.arcsin(s))))
    return sorted(out)


def array_near_field(setup: ArraySetup):
    """Near-field length of the active aperture (linear array), N = A²/(4λ)."""
    lam = setup.c / setup.f0
    return setup.aperture() ** 2 / (4 * lam)


# =========================================================================== beam field
def beam_field(setup: ArraySetup, x, z, delays=None, mode="pulsed", n_freq=24, n_tau=48):
    """Transmitted field in the part on the grid x × z. mode 'cw': |p| at f0; 'pulsed': peak
    envelope over time of the broadband pulse. Returns (nz, nx) array normalised to max 1."""
    X, Z = np.meshgrid(x, z)
    r = ray_paths(setup, X.ravel(), Z.ravel())
    N = setup.n_el
    d = np.zeros(N) if delays is None else np.asarray(delays)
    tt = r["t"] + d[:, None]
    m = setup.mat
    if mode == "cw":
        w = 2 * np.pi * setup.f0
        att = np.exp(-m.alpha(setup.f0) * r["d2"])
        p = np.abs(np.sum(r["amp"] * att * np.exp(-1j * w * tt), axis=0))
    else:
        sf = setup.bw * setup.f0 / (2 * np.sqrt(2 * np.log(2)))
        fr = np.linspace(max(setup.f0 - 3 * sf, 0.05 * setup.f0), setup.f0 + 3 * sf, n_freq)
        Pw = sg.gaussian_spectrum(fr, setup.f0, setup.bw)
        tref = np.median(tt, axis=0)
        span = 3.0 / (setup.bw * setup.f0)
        taus = np.linspace(-span, span, n_tau)
        p = np.zeros(X.size)
        att_w = np.exp(-m.alpha(fr)[:, None, None] * r["d2"][None])
        for g0 in range(0, X.size, 4096):
            sl = slice(g0, g0 + 4096)
            dt = tt[:, sl] - tref[None, sl]                                  # (N,g)
            ph = np.exp(-2j * np.pi * fr[:, None, None] * dt[None])          # (F,N,g)
            spec = np.sum(r["amp"][None, :, sl] * att_w[:, :, sl] * ph, axis=1) * Pw[:, None]
            sig = np.einsum("fg,tf->gt", spec, np.exp(2j * np.pi * taus[:, None] * fr[None]))
            p[sl] = np.max(np.abs(sig), axis=1)
    p = p.reshape(X.shape)
    return p / max(p.max(), 1e-300)


# =========================================================================== scatterers
@dataclass
class Scatterer:
    """kind: 'sdh' (side-drilled hole, radius = size/2), 'crack' (segment from (x,z) of length
    `size` at `angle` degrees from vertical), 'pore', 'backwall'."""
    kind: str
    x: float = 0.0
    z: float = 0.02
    size: float = 1.5e-3
    angle: float = 0.0
    strength: float = 1.0


def scatterer_points(setup: ArraySetup, scat, rng):
    """Point cloud (x, z, complex amplitude) for all scatterers + back wall + grain noise."""
    lam = setup.c / setup.f0
    xs, zs, amps = [], [], []
    for s in scat:
        if s.kind in ("sdh", "pore"):
            # 2-D cylinder backscatter ∝ √(r λ); pores slightly weaker (3-D)
            a = -np.sqrt(s.size / 2 / lam) * 0.5 * (0.6 if s.kind == "pore" else 1.0)
            xs.append([s.x]); zs.append([s.z - s.size / 2]); amps.append([a * s.strength])
        elif s.kind == "crack":
            n = max(3, int(np.ceil(s.size / (lam / 5))))
            u = np.linspace(0, s.size, n)
            a = np.radians(s.angle)
            xs.append(s.x + u * np.sin(a)); zs.append(s.z + u * np.cos(a))
            amps.append(np.full(n, -0.5 * s.strength * (lam / 5) / lam * 2.0))
    # back wall: dense chain, tapered over its ends
    el, _ = setup.elements()
    x0 = _beam_origin(setup)
    span = max(setup.aperture() * 2.5, 2.2 * setup.thickness)
    n = int(np.ceil(span / (lam / 5)))
    xb = x0 + np.linspace(-span / 2, span / 2, n)
    taper = np.hanning(n) ** 0.5
    xs.append(xb); zs.append(np.full(n, setup.thickness))
    amps.append(-1.0 * taper * (lam / 5) / lam * 2.0)
    # grains
    g = setup.grain_scale * (setup.mat.grain_um * 1e-6 / lam) ** 1.5
    if g > 0 and setup.n_grains > 0:
        ng = setup.n_grains
        gx = x0 + rng.uniform(-span / 2, span / 2, ng)
        gz = rng.uniform(0.6 * lam, setup.thickness - 0.5 * lam, ng)
        xs.append(gx); zs.append(gz)
        amps.append(rng.standard_normal(ng) * g * 2.0)
    return (np.concatenate(xs), np.concatenate(zs),
            np.concatenate([np.asarray(a, float) for a in amps]))


# =========================================================================== FMC
def simulate_fmc(setup: ArraySetup, scatterers=()):
    """Full-matrix capture: returns dict(t, fmc (N,N,n) real, analytic (N,N,n) complex)."""
    rng = np.random.default_rng(setup.seed)
    n = setup.n_samples()
    t = np.arange(n) / setup.fs
    f, idx = sg.band(setup.f0, setup.bw, setup.fs, n, floor=3e-4)
    fb = f[idx]
    Pw = sg.gaussian_spectrum(fb, setup.f0, setup.bw)
    t_delay = 4.0 / (2 * np.pi * setup.bw * setup.f0 / (2 * np.sqrt(2 * np.log(2))))
    qx, qz, qa = scatterer_points(setup, scatterers, rng)
    r = ray_paths(setup, qx, qz)
    m = setup.mat
    N = setup.n_el
    S = np.zeros((len(fb), N, N), complex)
    for f0 in range(0, len(fb), 32):
        fsl = slice(f0, f0 + 32)
        w = 2 * np.pi * fb[fsl, None, None]
        E = r["amp"][None] * np.exp(-1j * w * r["t"][None] - m.alpha(fb[fsl])[:, None, None]
                                    * r["d2"][None])                       # (F,N,Q)
        S[fsl] = np.einsum("fnq,fmq->fnm", E * qa[None, None], E)
    S *= (Pw * np.exp(-2j * np.pi * fb * t_delay))[:, None, None]
    S = np.moveaxis(S, 0, -1)                                              # (N,N,F)
    fmc = sg.irfft_band(S, idx, n)
    ref = sg.irfft_band((Pw * np.exp(-2j * np.pi * fb * t_delay))[None], idx, n)[0]
    scale = np.max(np.abs(ref))
    fmc /= scale
    fmc += setup.noise * np.max(np.abs(fmc)) * rng.standard_normal(fmc.shape)
    an = sg.analytic(fmc)
    return dict(t=t, fmc=fmc, analytic=an, t_delay=t_delay, f=fb)


# =========================================================================== imaging
def image_grid(setup, nx=140, nz=140, x_range=None, z_range=None):
    x0 = _beam_origin(setup)
    xr = x_range or (x0 - max(setup.aperture(), setup.thickness) * 0.9,
                     x0 + max(setup.aperture(), setup.thickness) * 0.9)
    if setup.coupling == "wedge":
        xr = x_range or (x0 - 0.2 * setup.thickness, x0 + 1.6 * setup.thickness)
    zr = z_range or (1e-3, setup.thickness * 1.08)
    return np.linspace(*xr, nx), np.linspace(*zr, nz)


def tfm(setup, data, x, z, coherence=False, tx=None):
    """Total focusing method: I(P) = |Σᵢⱼ hᵢⱼ(tᵢ(P) + tⱼ(P))| on the grid x × z (nz, nx).
    With coherence=True also returns the phase-coherence factor
    PCF(P) = |Σ e^{iφᵢⱼ}| / N² (Camacho et al. 2009 style instantaneous-phase coherence)."""
    X, Z = np.meshgrid(x, z)
    r = ray_paths(setup, X.ravel(), Z.ravel())
    tt = r["t"]
    an = data["analytic"]
    t = data["t"]
    dt = t[1] - t[0]
    td = data["t_delay"]
    N = an.shape[0]
    nt = an.shape[2]
    out = np.zeros(X.size, complex)
    coh = np.zeros(X.size, complex) if coherence else None
    txs = range(N) if tx is None else tx
    for i in txs:
        s = (tt[i][None] + tt + td) / dt                     # (N rx, G)
        k = np.floor(s).astype(np.int64)
        fr = s - k
        valid = (k >= 0) & (k < nt - 1)
        k = np.clip(k, 0, nt - 2)
        rows = an[i]                                         # (N rx, nt)
        v = np.take_along_axis(rows, k, axis=1) * (1 - fr) + \
            np.take_along_axis(rows, k + 1, axis=1) * fr
        v = np.where(valid, v, 0)
        out += v.sum(axis=0)
        if coherence:
            coh += (v / np.maximum(np.abs(v), 1e-30)).sum(axis=0)
    I = np.abs(out).reshape(X.shape)
    if coherence:
        pcf = (np.abs(coh) / (N * len(list(txs)))).reshape(X.shape)
        return I, pcf
    return I


def plane_wave_image(setup, data, x, z, angles_deg=np.linspace(-20, 20, 11)):
    """Plane-wave imaging synthesised from the FMC (contact arrays): each transmission
    sᵐⱼ(t) = Σᵢ sᵢⱼ(t − τᵢ(θₘ)), τᵢ = (xᵢ − x_min) sinθₘ / c, then coherent compounding."""
    if setup.coupling != "contact":
        raise ValueError("plane-wave imaging is implemented for contact arrays")
    el, _ = setup.elements()
    c = setup.c
    an = data["analytic"]
    t = data["t"]; dt = t[1] - t[0]; nt = len(t)
    td = data["t_delay"]
    X, Z = np.meshgrid(x, z)
    r = ray_paths(setup, X.ravel(), Z.ravel())
    out = np.zeros(X.size, complex)
    A = sfft.fft(an, axis=2, workers=-1)
    w = 2 * np.pi * sfft.fftfreq(nt, dt)
    for th in np.radians(np.asarray(angles_deg, float)):
        xs = el[:, 0]
        tau = (xs - (xs.min() if th >= 0 else xs.max())) * np.sin(th) / c
        sm = sfft.ifft(np.sum(A * np.exp(-1j * w[None, None] * tau[:, None, None]), axis=0),
                       axis=1, workers=-1)                                   # (N rx, nt)
        ttx = ((X.ravel() - (xs.min() if th >= 0 else xs.max())) * np.sin(th)
               + Z.ravel() * np.cos(th)) / c
        s = (ttx[None] + r["t"] + td) / dt
        k = np.floor(s).astype(np.int64); fr = s - k
        valid = (k >= 0) & (k < nt - 1); k = np.clip(k, 0, nt - 2)
        v = np.take_along_axis(sm, k, axis=1) * (1 - fr) + np.take_along_axis(sm, k + 1, axis=1) * fr
        out += np.where(valid, v, 0).sum(axis=0)
    return np.abs(out).reshape(X.shape)


def sector_scan(setup, data, x, z, angles_deg=np.linspace(-40, 40, 81), focus=None):
    """Conventional sectorial scan synthesised from the FMC: transmit focal law per angle
    (focus None → steering only), dynamic depth focusing on receive. Pixels outside the sector
    are 0. Returns (nz, nx)."""
    an = data["analytic"]
    t = data["t"]; dt = t[1] - t[0]; nt = len(t)
    td = data["t_delay"]
    N = setup.n_el
    angs = np.asarray(angles_deg, float)
    X, Z = np.meshgrid(x, z)
    x0 = _beam_origin(setup)
    pang = np.degrees(np.arctan2(X.ravel() - x0, Z.ravel()))
    m_idx = np.round(np.interp(pang, angs, np.arange(len(angs)))).astype(int)
    inside = (pang >= angs[0] - 1e-9) & (pang <= angs[-1] + 1e-9)
    r = ray_paths(setup, X.ravel(), Z.ravel())
    A = sfft.fft(an, axis=2, workers=-1)
    w = 2 * np.pi * sfft.fftfreq(nt, dt)
    out = np.zeros(X.size, complex)
    for m, ang in enumerate(angs):
        sel = np.nonzero(inside & (m_idx == m))[0]
        if len(sel) == 0:
            continue
        d = delay_law(setup, ang, focus)
        sm = sfft.ifft(np.sum(A * np.exp(-1j * w[None, None] * d[:, None, None]), axis=0),
                       axis=1, workers=-1)                                   # (N rx, nt)
        w_tx = r["amp"][:, sel]
        ttx = np.sum(w_tx * (d[:, None] + r["t"][:, sel]), axis=0) / np.maximum(
            np.sum(w_tx, axis=0), 1e-30)                       # amplitude-weighted arrival
        s = (ttx[None] + r["t"][:, sel] + td) / dt
        k = np.floor(s).astype(np.int64); fr = s - k
        valid = (k >= 0) & (k < nt - 1); k = np.clip(k, 0, nt - 2)
        v = np.take_along_axis(sm, k, axis=1) * (1 - fr) + np.take_along_axis(sm, k + 1, axis=1) * fr
        out[sel] = np.where(valid, v, 0).sum(axis=0)
    return np.abs(out).reshape(X.shape)


def array_performance(img, x, z, target, half_width=1.5e-3):
    """Signal-to-noise of an image at a target: peak in a box around target vs rms outside."""
    X, Z = np.meshgrid(x, z)
    box = (np.abs(X - target[0]) < half_width) & (np.abs(Z - target[1]) < half_width)
    peak = img[box].max() if box.any() else np.nan
    rest = img[~box]
    return float(20 * np.log10(peak / max(np.sqrt(np.mean(rest ** 2)), 1e-300)))


def amplitude_size(img, x, z, drop_db=6.0):
    """Extent (dx, dz) of the region above max − drop dB around the global maximum."""
    from scipy.ndimage import label
    thr = img.max() * 10 ** (-drop_db / 20)
    lab, _ = label(img >= thr)
    k = lab[np.unravel_index(np.argmax(img), img.shape)]
    zz, xx = np.nonzero(lab == k)
    return float(x[xx].max() - x[xx].min()), float(z[zz].max() - z[zz].min())
