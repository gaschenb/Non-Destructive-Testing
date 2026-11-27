"""Active infrared thermography: flash / step / lock-in heating of layered plates with
subsurface defects, and the standard processing (thermal contrast, TSR, pulsed-phase).

1-D model (thermal quadrupoles, Maillet et al. 2000)
---------------------------------------------------
Each layer is the 2×2 transfer matrix [[cosh kL, sinh kL/(λk)], [λk sinh kL, cosh kL]],
k = √(s/α), an interface resistance is [[1, R], [0, 1]]. Front-surface temperature for an input
flux Q(s) with front/rear heat-transfer coefficients h_f, h_r:
    θ(s) = Z Q / (1 + h_f Z),   Z = input impedance of the stack terminated by 1/h_r,
computed with the stable recursion Z ← (Z_load + tanh(kL)/(λk)) / (λk tanh(kL) Z_load + 1).
Time domain by Gaver–Stehfest, lock-in by s = iω.

3-D model
---------
Finite volumes on a regular grid, anisotropic conductivity per cell (harmonic means on faces),
backward-Euler locally-one-dimensional splitting (x, y, z sweeps). Every sweep is a single
block-tridiagonal system solved with LAPACK banded elimination, so a 64×64×24 plate advances in a
few milliseconds per step. Delaminations are thin gaps folded into the conductivity of the cell
that contains them (series in z, parallel in-plane), back-drilled holes and inclusions replace
cell properties.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from .eddy_current import stehfest_weights
from .materials import get


# =========================================================================== 1-D quadrupoles
@dataclass
class ThermalLayer:
    thickness: float
    k: float
    rho_cp: float

    @staticmethod
    def of(material, thickness, through=True):
        m = get(material)
        k = (m.k_z if (through and m.k_z is not None) else m.k)
        return ThermalLayer(thickness, k, m.rho * m.cp)

    @property
    def alpha(self):
        return self.k / self.rho_cp


def stack_impedance(layers, s, resistances=None, h_rear=0.0):
    """Input thermal impedance Z(s) of layers (front first) with interface resistances
    R between layer i and i+1 (list of len(layers) − 1, m²K/W) and rear exchange h_rear."""
    s = np.asarray(s, complex)
    Z = np.full(s.shape, np.inf + 0j) if h_rear == 0 else np.full(s.shape, 1 / h_rear + 0j)
    res = list(resistances or [0.0] * (len(layers) - 1))
    for i in range(len(layers) - 1, -1, -1):
        L = layers[i]
        kk = np.sqrt(s / L.alpha)
        lk = L.k * kk
        e = np.exp(-2 * kk * L.thickness)
        th = (1 - e) / (1 + e)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            Zn = np.where(np.isinf(Z), 1 / (lk * th), (Z + th / lk) / (lk * th * Z + 1))
        Z = Zn
        if i > 0:
            Z = Z + res[i - 1]
    return Z


def flash_response_1d(layers, t, Q=1e4, resistances=None, h_front=0.0, h_rear=0.0,
                      pulse=None, N=16):
    """Front-surface temperature rise after a flash of Q J/m² (pulse=None) or a rectangular
    heating pulse of duration `pulse` s with total energy Q."""
    t = np.atleast_1d(np.asarray(t, float))
    V = stehfest_weights(N)
    ln2 = np.log(2)
    out = np.zeros(len(t))
    for i, ti in enumerate(t):
        s = np.arange(1, N + 1) * ln2 / ti
        Z = stack_impedance(layers, s, resistances, h_rear)
        if pulse:
            q = Q / pulse * (1 - np.exp(-s * pulse)) / s
        else:
            q = Q
        th = np.real(Z * q / (1 + h_front * Z))
        out[i] = ln2 / ti * np.sum(V * th)
    return out


def adiabatic_slab(t, L, alpha, rho_cp, Q=1e4, n_terms=200):
    """Exact front-face rise of an adiabatic slab: Q/(ρcL)[1 + 2Σ exp(−n²π²αt/L²)]."""
    t = np.atleast_1d(t)[:, None]
    n = np.arange(1, n_terms + 1)[None]
    return Q / (rho_cp * L) * (1 + 2 * np.sum(np.exp(-n ** 2 * np.pi ** 2 * alpha * t / L ** 2),
                                              axis=1))


def lockin_1d(layers, f, q0=1e3, resistances=None, h_front=0.0, h_rear=0.0):
    """Complex surface temperature amplitude for a sinusoidal flux q₀ e^{iωt}."""
    s = 2j * np.pi * np.asarray(f, float)
    Z = stack_impedance(layers, s, resistances, h_rear)
    return Z * q0 / (1 + h_front * Z)


def diffusion_length(alpha, f):
    """Thermal diffusion length μ = √(α/(πf))."""
    return np.sqrt(alpha / (np.pi * np.asarray(f, float)))


def ppt_depth(alpha, f_blind, C1=1.82):
    """Pulsed-phase depth inversion d = C₁ √(α/(π f_b)) (Ibarra-Castanedo & Maldague 2004)."""
    return C1 * diffusion_length(alpha, f_blind)


def tsr_constant():
    """κ in t* = κ L²/α, the time of the peak of d²lnT/d(ln t)² for an adiabatic slab
    (computed once from the exact series): depth d = √(α t*/κ)."""
    tau = np.logspace(-3, 1, 4000)
    T = adiabatic_slab(tau, 1.0, 1.0, 1.0)
    lt, lT = np.log(tau), np.log(T)
    d1 = np.gradient(lT, lt)
    d2 = np.gradient(d1, lt)
    return float(tau[np.argmax(d2)])


_KAPPA = None


def tsr_window(alpha, thickness, frac=0.5):
    """Latest t* that can belong to a defect rather than to the back wall (frac·κL²/α)."""
    global _KAPPA
    if _KAPPA is None:
        _KAPPA = tsr_constant()
    return frac * _KAPPA * thickness ** 2 / alpha


def tsr_depth(alpha, t_star):
    global _KAPPA
    if _KAPPA is None:
        _KAPPA = tsr_constant()
    return np.sqrt(alpha * np.asarray(t_star) / _KAPPA)


# =========================================================================== 3-D model
@dataclass
class ThermalDefect:
    """kind 'delam' (gap of thickness `gap` at `depth`), 'fbh' (back-drilled flat-bottom hole
    with remaining ligament `depth` from the front), 'inclusion' (block of `fill` between depth
    and depth + `thick`). Lateral shape: rectangle wx × wy or ellipse (shape='round')."""
    kind: str = "delam"
    x: float = 0.0
    y: float = 0.0
    depth: float = 1e-3
    wx: float = 10e-3
    wy: float = 10e-3
    gap: float = 50e-6
    thick: float = 1e-3
    fill: str = "air"
    shape: str = "round"


@dataclass
class ThermalSetup:
    material: str = "cfrp"
    thickness: float = 4e-3
    size: tuple = (60e-3, 60e-3)
    nx: int = 48
    ny: int = 48
    nz: int = 24
    heating: str = "flash"            # 'flash' | 'step'
    energy: float = 8e3               # J/m² (flash) ; step: total over `pulse`
    pulse: float = 5.0                # step duration (s)
    h_front: float = 10.0
    h_rear: float = 10.0
    t_end: float = 20.0
    n_frames: int = 200
    nonuniform: float = 0.0           # flash non-uniformity (fraction, smooth lateral variation)
    nedt: float = 0.02                # camera noise-equivalent temperature difference (K)
    view: str = "reflection"          # 'reflection' | 'transmission'
    seed: int = 0


def _grid_props(setup: ThermalSetup, defects):
    m = get(setup.material)
    nx, ny, nz = setup.nx, setup.ny, setup.nz
    dz = setup.thickness / nz
    Lx, Ly = setup.size
    x = (np.arange(nx) + 0.5) / nx * Lx - Lx / 2
    y = (np.arange(ny) + 0.5) / ny * Ly - Ly / 2
    z = (np.arange(nz) + 0.5) * dz
    kxy = np.full((ny, nx, nz), float(m.k))
    kz = np.full((ny, nx, nz), float(m.k_z if m.k_z is not None else m.k))
    rc = np.full((ny, nx, nz), float(m.rho * m.cp))
    Y, X = np.meshgrid(y, x, indexing="ij")
    air = get("air")
    for d in defects:
        if d.shape == "round":
            lat = ((X - d.x) / (d.wx / 2)) ** 2 + ((Y - d.y) / (d.wy / 2)) ** 2 <= 1
        else:
            lat = (np.abs(X - d.x) <= d.wx / 2) & (np.abs(Y - d.y) <= d.wy / 2)
        if d.kind == "delam":
            iz = min(nz - 1, int(d.depth / dz))
            g = min(d.gap, 0.95 * dz)
            kh = kz[..., iz][lat]
            kz[..., iz][lat] = dz / (g / air.k + (dz - g) / kh)
            kxy[..., iz][lat] = (g * air.k + (dz - g) * kxy[..., iz][lat]) / dz
        elif d.kind in ("fbh", "inclusion"):
            f = get(d.fill)
            z0 = d.depth
            z1 = setup.thickness if d.kind == "fbh" else d.depth + d.thick
            zsel = (z >= z0) & (z <= z1 + 1e-12)
            sel = lat[..., None] & zsel[None, None]
            kxy[sel] = f.k
            kz[sel] = f.k_z if f.k_z is not None else f.k
            rc[sel] = f.rho * f.cp
    return x, y, z, dz, kxy, kz, rc


def _face_conductance(k, h, axis):
    """Harmonic-mean conductance per unit area between neighbouring cells (W/m²K)/m → k/h²."""
    a = np.take(k, range(k.shape[axis] - 1), axis=axis)
    b = np.take(k, range(1, k.shape[axis]), axis=axis)
    return 2 * a * b / (a + b) / h ** 2


class ThermalSolver:
    """Backward-Euler LOD solver; `run()` returns surface temperature frames."""

    def __init__(self, setup: ThermalSetup, defects=()):
        self.s = setup
        self.x, self.y, self.z, self.dz, kxy, kz, self.rc = _grid_props(setup, defects)
        Lx, Ly = setup.size
        self.dx, self.dy = Lx / setup.nx, Ly / setup.ny
        self.G = [_face_conductance(kxy, self.dy, 0), _face_conductance(kxy, self.dx, 1),
                  _face_conductance(kz, self.dz, 2)]
        self._cache = {}

    def _bands(self, axis, dt):
        key = (axis, dt)
        if key in self._cache:
            return self._cache[key]
        G = self.G[axis]
        rc = self.rc
        n = rc.shape[axis]
        lower = np.zeros(rc.shape); upper = np.zeros(rc.shape)
        sl_lo = [slice(None)] * 3; sl_lo[axis] = slice(1, None)
        sl_up = [slice(None)] * 3; sl_up[axis] = slice(0, n - 1)
        lower[tuple(sl_lo)] = -dt * G / rc[tuple(sl_lo)]      # coefficient of T_{i−1}
        upper[tuple(sl_up)] = -dt * G / rc[tuple(sl_up)]      # coefficient of T_{i+1}
        diag = 1 - lower - upper
        if axis == 2:   # convective losses on front / rear faces
            diag[..., 0] += dt * self.s.h_front / (rc[..., 0] * self.dz)
            diag[..., -1] += dt * self.s.h_rear / (rc[..., -1] * self.dz)
        perm = [a for a in range(3) if a != axis] + [axis]
        lo = np.moveaxis(lower, axis, -1).ravel()
        up = np.moveaxis(upper, axis, -1).ravel()
        dg = np.moveaxis(diag, axis, -1).ravel()
        ab = np.zeros((3, dg.size))
        ab[0, 1:] = up[:-1]
        ab[1] = dg
        ab[2, :-1] = lo[1:]
        self._cache[key] = (ab, perm)
        return ab, perm

    def _sweep(self, T, axis, dt):
        ab, perm = self._bands(axis, dt)
        Tm = np.moveaxis(T, axis, -1)
        shp = Tm.shape
        sol = solve_banded((1, 1), ab, Tm.ravel(), overwrite_b=True, check_finite=False)
        return np.moveaxis(sol.reshape(shp), -1, axis)

    def step(self, T, dt, flux=None):
        if flux is not None:
            T[..., 0] += flux * dt / (self.rc[..., 0] * self.dz)
        for ax in (0, 1, 2):
            T = self._sweep(T, ax, dt)
        return T

    def run(self, progress=None):
        s = self.s
        rng = np.random.default_rng(s.seed)
        ny, nx, nz = self.rc.shape
        T = np.zeros((ny, nx, nz))
        Y, X = np.meshgrid(self.y, self.x, indexing="ij")
        Lx, Ly = s.size
        lateral = 1 + s.nonuniform * (np.cos(np.pi * X / Lx) * np.cos(np.pi * Y / Ly) - 0.5) * 2
        q_map = s.energy * lateral
        if s.heating == "flash":
            T[..., 0] = q_map / (self.rc[..., 0] * self.dz)
        dtf = s.t_end / s.n_frames
        times = np.arange(1, s.n_frames + 1) * dtf
        frames = np.empty((s.n_frames, ny, nx))
        rear = np.empty((s.n_frames, ny, nx))
        t = 0.0
        for k, tk in enumerate(times):
            # sub-steps: geometric in the first frame, ≥ 10 steps per elapsed time scale after
            if k == 0:
                edges = tk * np.concatenate([[0], np.geomspace(2e-4, 1.0, 14)])
            else:
                m = int(np.ceil(6.0 / k)) if k < 6 else 1
                edges = np.linspace(t, tk, m + 1)
            for a, b in zip(edges[:-1], edges[1:]):
                dt = float(np.round(b - a, 12))
                if dt <= 0:
                    continue
                flux = None
                if s.heating == "step" and a < s.pulse:
                    frac = (min(b, s.pulse) - a) / dt
                    flux = q_map / s.pulse * frac
                T = self.step(T, dt, flux)
            t = tk
            frames[k] = T[..., 0]
            rear[k] = T[..., -1]
            if progress is not None and k % 10 == 0:
                progress(k, s.n_frames)
        # surface temperature of the first cell centre → extrapolate to the face (½ cell)
        img = frames if s.view == "reflection" else rear
        noisy = img + s.nedt * rng.standard_normal(img.shape)
        return dict(t=times, frames=noisy, clean=img, x=self.x, y=self.y, dz=self.dz)


# =========================================================================== processing
def sound_reference(frames, mask=None):
    """Mean sound-area history (median over the frame if no mask)."""
    if mask is None:
        return np.median(frames.reshape(len(frames), -1), axis=1)
    return frames[:, mask].mean(axis=1)


def thermal_contrast(frames, ref=None, kind="absolute"):
    ref = sound_reference(frames) if ref is None else ref
    if kind == "running":
        return (frames - ref[:, None, None]) / np.maximum(ref[:, None, None], 1e-9)
    return frames - ref[:, None, None]


def tsr_fit(t, frames, degree=6, clip=1e-4, n_log=80):
    """Thermographic signal reconstruction: per-pixel polynomial in ln t of ln T, fitted on
    a log-uniform resampling (n_log points; 0 = use frames as they are) so that every decade
    carries equal weight. Returns (coeffs (deg+1, ny, nx), callable giving recon / 1st / 2nd
    log-derivatives at times)."""
    lt = np.log(t)
    lT = np.log(np.maximum(frames, clip)).reshape(len(t), -1)
    if n_log:
        lg = np.linspace(lt[0], lt[-1], n_log)
        j = np.clip(np.searchsorted(lt, lg) - 1, 0, len(lt) - 2)
        w = ((lg - lt[j]) / (lt[j + 1] - lt[j]))[:, None]
        lT = (1 - w) * lT[j] + w * lT[j + 1]
        lt = lg
    V = np.vander(lt, degree + 1)
    coef, *_ = np.linalg.lstsq(V, lT, rcond=None)
    shp = frames.shape[1:]

    def evaluate(times, order=0):
        x = np.log(np.atleast_1d(times))
        c = coef
        for _ in range(order):
            c = _polyder(c)
        Vt = np.vander(x, len(c))
        out = Vt @ c
        if order == 0:
            out = np.exp(out)
        return out.reshape((len(x),) + shp)
    return coef.reshape((degree + 1,) + shp), evaluate


def _polyder(c):
    n = len(c) - 1
    return c[:-1] * np.arange(n, 0, -1)[:, None]


def ppt(t, frames):
    """Pulsed-phase thermography: FFT along time, returns (f, phase (nf, ny, nx), amp)."""
    dt = t[1] - t[0]
    F = np.fft.rfft(frames, axis=0)
    f = np.fft.rfftfreq(len(t), dt)
    return f, np.angle(F), np.abs(F)


def blind_frequency(f, phase_def, phase_ref, tol_deg=1.0):
    """First frequency above which the defect/sound phase difference stays below tol."""
    d = np.degrees(np.unwrap(phase_def) - np.unwrap(phase_ref))
    big = np.nonzero(np.abs(d[1:]) > tol_deg)[0]
    if len(big) == 0:
        return float(f[1])
    return float(f[1:][big[-1]])


def peak_contrast(t, contrast_curve):
    i = int(np.argmax(contrast_curve))
    return float(t[i]), float(contrast_curve[i])


def defect_extent(img, x, y, level=0.5):
    """FWHM extent (wx, wy) of the region around the maximum of a contrast image."""
    from scipy.ndimage import label
    thr = img.max() * level
    lab, _ = label(img >= thr)
    k = lab[np.unravel_index(np.argmax(img), img.shape)]
    yy, xx = np.nonzero(lab == k)
    dx, dy = x[1] - x[0], y[1] - y[0]
    return float(x[xx].max() - x[xx].min() + dx), float(y[yy].max() - y[yy].min() + dy)
