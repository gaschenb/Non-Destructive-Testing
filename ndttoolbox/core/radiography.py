"""X-ray and gamma radiography, digital detector images, image-quality indicators and CT.

Attenuation
-----------
μ/ρ = (N_A/A) Σ_elements w [Z σ_KN(E) + τ(Z,E) + σ_coh(Z,E)]   [cm²/g]
with the exact Klein–Nishina cross-section σ_KN per electron, a photoelectric term
τ = C Z^n (E/100 keV)^{−m(Z)} (divided by the K-shell jump below the K edge) and a coherent
term C_c Z^{n_c} (E/100)^{−m_c}. The six constants were fitted to NIST XCOM totals for water,
Al, Fe and Pb (20 keV–1 MeV); residuals are ≤ 5 % for Z ≤ 26 and ≤ 12 % near the Pb K edge.

Sources: tungsten-anode bremsstrahlung (Kramers, N(E) ∝ Z(E₀ − E)/E per unit energy) with
characteristic K lines above 69.5 kV, inherent and added filtration; Ir-192, Se-75, Co-60
gamma lines. Detector: CsI-scintillator DDA, energy-integrating, absorption η(E) = 1 − e^{−μt},
quantum (Poisson) noise, Gaussian unsharpness from focal spot and detector, scatter through a
linear build-up factor B = 1 + kμx spread by a broad kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .materials import ELEMENTS, get

NA = 6.02214076e23
R_E = 2.8179403262e-13            # classical electron radius, cm
_P = (-10.0103837, 3.97053021, 3.0222325, 0.1102084, -12.54227375, 4.14504974, 1.67440604,
      9.46825401)


def klein_nishina(E):
    """Total Klein–Nishina cross-section per electron [cm²], E in keV."""
    k = np.asarray(E, float) / 510.998950
    l = np.log1p(2 * k)
    a = (1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - l / k) + l / (2 * k) \
        - (1 + 3 * k) / (1 + 2 * k) ** 2
    return 2 * np.pi * R_E ** 2 * a


def mu_rho_element(sym, E):
    Z, A, EK, _ = ELEMENTS[sym]
    E = np.asarray(E, float)
    lC, n, m0, m1, lCc, nc, mc, jr = _P
    m = m0 + m1 * np.log(Z / 26)
    tau = np.exp(lC) * Z ** n * (E / 100) ** (-m) * 1e-24
    tau = np.where(E < EK, tau / jr, tau)
    coh = np.exp(lCc) * Z ** nc * (E / 100) ** (-mc) * 1e-24
    return NA / A * (Z * klein_nishina(E) + tau + coh)


def mu_rho(material, E):
    """Mass attenuation coefficient μ/ρ [cm²/g] of a material (composition by mass)."""
    m = get(material)
    return sum(w * mu_rho_element(s, E) for s, w in m.composition.items())


def mu(material, E):
    """Linear attenuation coefficient [1/m]."""
    m = get(material)
    return mu_rho(m, E) * m.rho / 1000 * 100          # (cm²/g)(g/cm³) → 1/cm → 1/m


def half_value_layer(material, E):
    return np.log(2) / mu(material, E)


# =========================================================================== sources
W_LINES = [(57.98, 0.30), (59.32, 0.52), (67.24, 0.14), (69.07, 0.04)]   # keV, relative
GAMMA = {
    "Ir-192": [(295.96, 0.287), (308.46, 0.297), (316.51, 0.829), (468.07, 0.478),
               (588.58, 0.045), (604.41, 0.082), (612.47, 0.053)],
    "Se-75": [(121.1, 0.172), (136.0, 0.585), (264.7, 0.589), (279.5, 0.250), (400.7, 0.115)],
    "Co-60": [(1173.2, 0.9985), (1332.5, 0.9998)],
}


@dataclass
class Source:
    kind: str = "xray"               # 'xray' | 'Ir-192' | 'Se-75' | 'Co-60'
    kv: float = 200.0
    ma: float = 5.0
    exposure_s: float = 10.0
    activity_ci: float = 50.0
    focal_spot: float = 1.0e-3       # m (effective size)
    filter_material: str = "copper"
    filter_mm: float = 0.5
    inherent_al_mm: float = 3.0

    def spectrum(self, n=160):
        """Photon fluence per energy bin at 1 m [photons / m²] for the whole exposure.
        Returns (E keV, N)."""
        if self.kind == "xray":
            E = np.linspace(5, self.kv, n)
            dE = E[1] - E[0]
            # Kramers: energy fluence ∝ Z(E0 − E) → photon number ∝ Z(E0 − E)/E
            N = 74 * np.clip(self.kv - E, 0, None) / E * dE
            if self.kv > 69.53:
                strength = 0.11 * (self.kv / 69.53 - 1) ** 1.63 * np.sum(N) / 1.0
                for e, w in W_LINES:
                    N[np.argmin(np.abs(E - e))] += strength * w
            # output calibration: ≈ 1e6 photons/(mm² mAs) at 1 m behind 3 mm Al at 100 kV
            # (≈ 50 µGy/mAs), scaling ∝ kV² before filtration
            N *= 2.0e12 / np.sum(N) * (self.kv / 100) ** 2 * self.ma * self.exposure_s
            # anode self-filtration + inherent + added filters
            N *= np.exp(-mu("aluminium-6061", E) * self.inherent_al_mm * 1e-3)
            if self.filter_mm > 0:
                N *= np.exp(-mu(self.filter_material, E) * self.filter_mm * 1e-3)
            return E, N
        lines = GAMMA[self.kind]
        E = np.array([e for e, _ in lines])
        y = np.array([w for _, w in lines])
        decays = self.activity_ci * 3.7e10 * self.exposure_s
        return E, decays * y / (4 * np.pi)            # per m² at 1 m

    def mean_energy(self):
        E, N = self.spectrum()
        return float(np.sum(E * N) / np.sum(N))


@dataclass
class Detector:
    scint_material: str = "csi"
    scint_um: float = 500.0
    pixel: float = 0.1e-3
    srb: float = 0.13e-3             # basic spatial resolution (≈ half the unsharpness)
    swank: float = 0.85
    electronic_noise: float = 20.0   # grey values rms

    def absorption(self, E):
        return 1 - np.exp(-mu(self.scint_material, E) * self.scint_um * 1e-6)


@dataclass
class Geometry:
    sod: float = 0.7                 # source–object distance (m)
    odd: float = 0.05                # object–detector distance (m)

    @property
    def sdd(self):
        return self.sod + self.odd

    @property
    def magnification(self):
        return self.sdd / self.sod

    def geometric_unsharpness(self, focal_spot):
        return focal_spot * self.odd / self.sod


def total_unsharpness(ug, ud):
    """EN ISO 17636-2 combination U_T = (U_g³ + U_d³)^{1/3} (in the detector plane)."""
    return (ug ** 3 + ud ** 3) ** (1 / 3)


# =========================================================================== objects
WIRE_IQI = [3.20, 2.50, 2.00, 1.60, 1.25, 1.00, 0.80, 0.63, 0.50, 0.40, 0.32, 0.25, 0.20,
            0.16, 0.125, 0.100, 0.080, 0.063, 0.050]          # ISO 19232-1 W1 … W19 (mm)


@dataclass
class RTFlaw:
    """kind: 'pore' (sphere), 'slag' (sphere of `fill`), 'tungsten', 'crack' (planar, height
    `size2`, opening `width`, tilt to beam `angle`), 'lof' (lack of fusion along the bevel),
    'undercut' (groove at the weld toe). Positions in the object plane (m)."""
    kind: str
    x: float = 0.0
    y: float = 0.0
    size: float = 1e-3               # diameter or length
    size2: float = 3e-3              # crack height (through-thickness extent)
    width: float = 0.05e-3           # crack opening
    angle: float = 0.0               # degrees between crack plane and beam
    fill: str = "glass-slag"


@dataclass
class RTObject:
    """A plate (optionally welded) seen along the beam. Thickness maps are built on a grid."""
    material: str = "steel"
    thickness: float = 12e-3
    weld: bool = True
    weld_width: float = 16e-3
    cap: float = 2.0e-3              # reinforcement height
    flaws: list = field(default_factory=list)
    iqi: str | None = "W10-W16"      # wire set placed on the source side, wires along y
    step_wedge: list | None = None   # list of step thicknesses (replaces the weld model)
    size: tuple = (60e-3, 40e-3)     # field of view (x, y)

    def maps(self, nx, ny):
        """Path lengths through each material: dict material → (ny, nx) array (m)."""
        Lx, Ly = self.size
        x = (np.arange(nx) + 0.5) / nx * Lx - Lx / 2
        y = (np.arange(ny) + 0.5) / ny * Ly - Ly / 2
        X, Y = np.meshgrid(x, y)
        base = np.full(X.shape, self.thickness)
        if self.step_wedge:
            k = np.clip(((X + Lx / 2) / Lx * len(self.step_wedge)).astype(int), 0,
                        len(self.step_wedge) - 1)
            base = np.asarray(self.step_wedge)[k]
        elif self.weld:
            u = X / (self.weld_width / 2)
            base = base + self.cap * np.clip(1 - u ** 2, 0, None)
        maps = {self.material: base.copy()}
        for fl in self.flaws:
            r = fl.size / 2
            rho2 = (X - fl.x) ** 2 + (Y - fl.y) ** 2
            if fl.kind in ("pore", "slag", "tungsten"):
                chord = 2 * np.sqrt(np.clip(r ** 2 - rho2, 0, None))
                maps[self.material] -= chord
                if fl.kind != "pore":
                    key = "tungsten" if fl.kind == "tungsten" else fl.fill
                    maps[key] = maps.get(key, 0) + chord
            elif fl.kind in ("crack", "lof"):
                ang = np.radians(fl.angle if fl.kind == "crack" else 30.0)
                h, w = fl.size2, fl.width
                proj = h * np.sin(ang) + w * np.cos(ang)        # projected width
                deficit = w * h / max(proj, 1e-9)
                # crack runs along y, slightly wavy (cracks are not straight)
                xc = fl.x + 0.15e-3 * np.sin(Y / 1.7e-3) * (fl.kind == "crack")
                inside = (np.abs(X - xc) <= proj / 2 + 1e-12) & (np.abs(Y - fl.y) <= fl.size / 2)
                taper = np.sqrt(np.clip(1 - ((Y - fl.y) / (fl.size / 2)) ** 2, 0, None))
                maps[self.material] -= deficit * inside * taper
            elif fl.kind == "undercut":
                g = np.exp(-0.5 * ((X - fl.x) / (fl.size / 2)) ** 2)
                maps[self.material] -= fl.size2 * g * (np.abs(Y - fl.y) < 15e-3)
        for num, d, xw, yw, half in self.iqi_wires():
            # wire lies across the weld (along x) at height y = yw
            inside_x = np.abs(X - xw) < half
            maps[self.material] = maps[self.material] + _chord_avg(Y - yw, d / 2,
                                                                   y[1] - y[0]) * inside_x
        maps[self.material] = np.clip(maps[self.material], 0, None)
        return maps, x, y

    def iqi_wires(self):
        """(number, diameter, x centre, y position, half length) of each IQI wire. The wires lie
        across the weld on the source side, stacked in y near the top of the field of view."""
        if not self.iqi:
            return []
        Lx, Ly = self.size
        lo, hi = [int(s[1:]) for s in self.iqi.split("-")]
        nums = list(range(lo, hi + 1))
        spacing = min(2.5e-3, 0.4 * Ly / max(len(nums), 1))
        y0 = Ly / 2 - 2e-3 - spacing * (len(nums) - 1)
        half = min(12.5e-3, 0.45 * Lx)
        return [(num, WIRE_IQI[num - 1] * 1e-3, 0.0, y0 + j * spacing, half)
                for j, num in enumerate(nums)]


def _chord_avg(u, r, p):
    """Pixel-averaged chord length of a cylinder of radius r: (1/p)∫ 2√(r² − v²) dv over the
    pixel [u − p/2, u + p/2] (exact; wires thinner than a pixel keep their area πr²)."""
    def F(v):
        v = np.clip(v, -r, r)
        return v * np.sqrt(np.clip(r ** 2 - v ** 2, 0, None)) + r ** 2 * np.arcsin(v / r)
    return (F(u + p / 2) - F(u - p / 2)) / p


# extra materials used by the radiography models
from . import materials as _mats   # noqa: E402

_mats.MATERIALS.setdefault("glass-slag", _mats.Material(
    "glass-slag", rho=2600, cL=5500, cT=3300, composition={"O": 0.50, "Si": 0.30, "Ca": 0.12,
                                                            "Mn": 0.08}))


# =========================================================================== imaging
@dataclass
class RTSetup:
    source: Source = field(default_factory=Source)
    detector: Detector = field(default_factory=Detector)
    geometry: Geometry = field(default_factory=Geometry)
    scatter_k: float = 0.6           # build-up B = 1 + k μx (at the mean energy)
    seed: int = 0


def radiograph(setup: RTSetup, obj: RTObject, n_e=80, noise=True, gain=None):
    """Simulate a digital radiograph. Returns dict(image, clean, primary, scatter, x, y, ...)
    with grey values proportional to absorbed energy (16-bit-like scale). `gain` fixes the
    grey-value scale (e.g. that of another radiograph, returned as out["gain"])."""
    src, det, geo = setup.source, setup.detector, setup.geometry
    M = geo.magnification
    p_obj = det.pixel / M                          # pixel size projected into the object
    nx = int(round(obj.size[0] / p_obj)); ny = int(round(obj.size[1] / p_obj))
    maps, x, y = obj.maps(nx, ny)
    E, N = src.spectrum(n_e)
    eta = det.absorption(E)
    fluence = N / geo.sdd ** 2 * det.pixel ** 2          # photons per pixel (no object)
    mus = {m: mu(m, E) for m in maps}
    att = np.zeros((len(E),) + maps[obj.material].shape)
    for m, L in maps.items():
        att += mus[m][:, None, None] * L[None]
    T = np.exp(-att)                                     # (E, ny, nx)
    w = fluence * eta * E                                # absorbed energy per photon bin
    primary = np.tensordot(w, T, axes=(0, 0))
    photons = np.tensordot(fluence * eta, T, axes=(0, 0))
    # scatter: B − 1 = k μ̄x with μ̄ at the spectrum-weighted transmitted energy
    Ebar = np.sum(w * np.mean(T, axis=(1, 2)) * E) / max(np.sum(w * np.mean(T, axis=(1, 2))), 1e-30)
    mbar = mu(obj.material, Ebar)
    spr = setup.scatter_k * mbar * maps[obj.material]
    scatter = gaussian_filter(primary * spr, 12e-3 / p_obj, mode="nearest")
    # unsharpness in the detector plane → object-plane pixels
    ug = geo.geometric_unsharpness(src.focal_spot)
    ud = 2 * det.srb
    ut = total_unsharpness(ug, ud)
    sig_px = ut / 2.355 / det.pixel
    clean = gaussian_filter(primary + scatter, sig_px, mode="nearest")
    if gain is None:
        gain = 40000.0 / max(np.percentile(primary + scatter, 99), 1e-30)
    out = dict(clean=clean * gain, primary=primary * gain, scatter=scatter * gain, x=x, y=y,
               maps=maps, E=E, spectrum=N, eta=eta, unsharpness=ut, ug=ug, ud=ud,
               magnification=M, p_obj=p_obj, E_mean_transmitted=float(Ebar), spr=spr,
               gain=gain)
    if noise:
        rng = np.random.default_rng(setup.seed)
        # energy-integrating detector: variance = Σ N E² (per pixel) / Swank
        en2 = np.tensordot(fluence * eta * E ** 2, T, axes=(0, 0))
        var = (en2 * (1 + spr)) / det.swank
        var_s = gaussian_filter(var, sig_px * 0.7, mode="nearest")
        img = clean * gain + rng.standard_normal(clean.shape) * np.sqrt(var_s) * gain \
            / np.sqrt(max(1.0, 4 * np.pi * (sig_px * 0.7) ** 2)) \
            + det.electronic_noise * rng.standard_normal(clean.shape)
        out["image"] = img
        out["photons"] = photons
    else:
        out["image"] = out["clean"]
    return out


def snr_normalized(img, region, srb):
    """SNR_N = SNR_measured · 88.6 µm / SRb (EN ISO 17636-2) in a (y-slice, x-slice) region."""
    patch = img[region]
    from scipy.ndimage import uniform_filter
    hp = patch - uniform_filter(patch, 9)          # remove low-frequency structure
    snr = np.mean(patch) / max(np.std(hp) * np.sqrt(81 / 80), 1e-30)
    return float(snr * 88.6e-6 / srb), float(snr)


def wire_visibility(res, obj: RTObject, threshold=5.0, length=6e-3):
    """Detectability index of every IQI wire, d' = ΔI/σ · √n_px, with ΔI the peak contrast of
    the noise-free blurred wire profile (averaged over `length` along the wire), σ the pixel
    noise next to it and n_px the pixels of the wire's projected area (width ≈ max(d,
    unsharpness) × length); visible if d' ≥ 5 (Rose criterion). Returns (list of dicts,
    number of the thinnest visible wire)."""
    x, y = res["x"], res["y"]
    clean, img = res["clean"], res["image"]
    p = res["p_obj"]
    out, best = [], None
    noise_map = img - clean
    for num, d, xw, yw, half in obj.iqi_wires():
        ix = slice(np.searchsorted(x, xw - length / 2), np.searchsorted(x, xw + length / 2))
        iy = int(np.argmin(np.abs(y - yw)))
        k = max(2, int(round(0.9e-3 / p)))
        prof = clean[:, ix].mean(axis=1)
        bg = 0.5 * (prof[max(iy - k, 0)] + prof[min(iy + k, len(prof) - 1)])
        dI = abs(prof[iy] - bg)
        noise = np.std(noise_map[max(iy - k, 0):iy + k + 1, ix])
        w_eff = max(d, res["unsharpness"] / res["magnification"])
        npx = max(1.0, w_eff * length / p ** 2)
        dprime = dI / max(noise, 1e-30) * np.sqrt(npx)
        vis = dprime >= threshold
        out.append(dict(wire=f"W{num}", diameter_mm=d * 1e3, contrast=float(dI),
                        dprime=float(dprime), visible=bool(vis)))
        if vis:
            best = num
    return out, best


def contrast_sensitivity(res, obj: RTObject):
    """Thinnest visible wire diameter as a percentage of the penetrated thickness."""
    wires, best = wire_visibility(res, obj)
    if best is None:
        return None
    return WIRE_IQI[best - 1] * 1e-3 / (obj.thickness + (obj.cap if obj.weld else 0)) * 100


# =========================================================================== CT
def phantom(kind="pipe", n=256, fov=0.12):
    """2-D CT phantoms: dict material → (n, n) indicator (fractional) maps and the pixel size."""
    x = (np.arange(n) + 0.5) / n * fov - fov / 2
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    maps = {}
    if kind == "pipe":
        steel = ((R < 0.05) & (R > 0.042)).astype(float)
        # corrosion pits on the inner wall and a crack-like notch on the outer wall
        for ang, depth in [(30, 2.5e-3), (120, 1.5e-3), (220, 3.5e-3)]:
            a = np.radians(ang)
            cx, cy = 0.042 * np.cos(a), 0.042 * np.sin(a)
            steel[np.hypot(X - cx, Y - cy) < depth] = 0
        th = np.arctan2(Y, X)
        steel[(np.abs(th - np.radians(300)) < 0.012) & (R > 0.047)] = 0
        water = (R <= 0.042).astype(float) * (steel == 0)
        water[R > 0.042] = 0
        for ang, depth in [(30, 2.5e-3), (120, 1.5e-3), (220, 3.5e-3)]:
            a = np.radians(ang)
            cx, cy = 0.042 * np.cos(a), 0.042 * np.sin(a)
            water[(np.hypot(X - cx, Y - cy) < depth) & (R < 0.05)] = 1
        maps = {"steel": steel, "water": water}
    elif kind == "casting":
        al = ((np.abs(X) < 0.045) & (np.abs(Y) < 0.03)).astype(float)
        al[np.hypot(X + 0.02, Y) < 0.012] = 0                      # cored hole
        rng = np.random.default_rng(3)
        for _ in range(12):                                      # gas porosity
            cx, cy = rng.uniform(-0.04, 0.04), rng.uniform(-0.025, 0.025)
            al[np.hypot(X - cx, Y - cy) < rng.uniform(0.4e-3, 1.6e-3)] = 0
        steel = (np.hypot(X - 0.025, Y + 0.01) < 4e-3).astype(float)   # steel insert
        al[steel > 0] = 0
        maps = {"aluminium-6061": al, "steel": steel}
    elif kind == "composite":
        cf = ((np.abs(X) < 0.05) & (np.abs(Y) < 0.006)).astype(float)
        for cx in (-0.03, 0.0, 0.025):                              # delaminations / voids
            cf[(np.abs(X - cx) < 4e-3) & (np.abs(Y - 0.002 * np.sign(cx + 1e-9)) < 0.25e-3)] = 0
        ti = ((np.abs(X - 0.04) < 2.5e-3) & (np.abs(Y) < 0.009)).astype(float)
        cf[ti > 0] = 0
        maps = {"cfrp": cf, "titanium-6al4v": ti}
    else:
        raise ValueError(kind)
    return maps, fov / n


def radon(img, angles_deg):
    """Parallel-beam projections (line integrals in pixel units) by image rotation."""
    from scipy.ndimage import rotate
    out = np.empty((len(angles_deg), img.shape[1]))
    for i, a in enumerate(angles_deg):
        out[i] = rotate(img, -a, reshape=False, order=1, mode="constant").sum(axis=0)
    return out


def ct_sinogram(maps, px, source: Source, angles_deg, detector: Detector | None = None,
                photons_per_ray=2e5, seed=0, mono_kev=None, paths=None):
    """Polychromatic (or monochromatic) parallel-beam sinogram p = −ln(I/I₀) with Poisson
    noise; also returns the per-material path-length sinograms."""
    det = detector or Detector()
    L = paths if paths is not None else {m: radon(v, angles_deg) * px for m, v in maps.items()}
    if mono_kev is not None:
        E = np.array([float(mono_kev)]); w = np.array([1.0])
    else:
        E, N = source.spectrum(100)
        w = N * det.absorption(E) * E
        w = w / w.sum()
    att = 0
    for m, l in L.items():
        att = att + mu(m, E)[:, None, None] * l[None]
    I = np.tensordot(w, np.exp(-att), axes=(0, 0))
    rng = np.random.default_rng(seed)
    if photons_per_ray:
        counts = rng.poisson(np.clip(I * photons_per_ray, 0, None)).astype(float)
        I = np.maximum(counts, 1.0) / photons_per_ray
    return -np.log(I), L, (E, w)


def fbp(sino, angles_deg, px, filt="hann"):
    """Filtered back-projection (ramp × window), returning μ in 1/m."""
    n_a, n = sino.shape
    pad = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.fftfreq(pad)
    # ramp from the band-limited spatial Ram-Lak kernel (Kak & Slaney): h(0) = 1/4,
    # h(n odd) = −1/(πn)²; avoids the DC bias of sampling |f| directly
    nn = np.concatenate([np.arange(1, pad // 2 + 1, 2), np.arange(pad // 2 - 1, 0, -2)])
    h = np.zeros(pad)
    h[0] = 0.25
    h[1::2] = -1 / (np.pi * nn) ** 2
    H = 2 * np.real(np.fft.fft(h))
    if filt == "hann":
        H *= 0.5 * (1 + np.cos(2 * np.pi * f))
    elif filt == "shepp-logan":
        H *= np.sinc(f)
    q = np.real(np.fft.ifft(np.fft.fft(sino, pad, axis=1) * H, axis=1))[:, :n]
    c = (np.arange(n) - (n - 1) / 2)
    X, Y = np.meshgrid(c, c)
    img = np.zeros((n, n))
    for i, a in enumerate(np.radians(angles_deg)):
        s = X * np.cos(a) + Y * np.sin(a) + (n - 1) / 2
        img += np.interp(s.ravel(), np.arange(n), q[i], left=0, right=0).reshape(n, n)
    return img * np.pi / (2 * n_a) / px


def effective_energy(material, ew, length):
    """Mean energy of the detected spectrum after `length` of `material` (beam-hardened)."""
    E, w = ew
    t = w * np.exp(-mu(material, E) * length)
    return float(np.sum(t * E) / np.sum(t))


def beam_hardening_correction(sino, material, ew, max_len=0.2, order=4, mu_ref=None):
    """Linearisation: map polychromatic p to μ_ref·L of `material` (default: the
    detector-weighted mean μ), using a polynomial fit of the forward relation p(L)."""
    E, w = ew
    L = np.linspace(0, max_len, 400)
    p = -np.log(np.tensordot(w, np.exp(-mu(material, E)[:, None] * L[None]), axes=(0, 0)))
    m_eff = np.sum(w * mu(material, E)) if mu_ref is None else mu_ref
    coef = np.polyfit(p, m_eff * L, order)
    return np.polyval(coef, sino)


def cupping(img, mask):
    """Cupping index: (edge mean − centre mean)/edge mean inside a (convex) material mask."""
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(mask)
    if d.max() <= 2:
        return 0.0
    edge = img[(d > 1) & (d <= 0.25 * d.max())].mean()
    centre = img[d >= 0.75 * d.max()].mean()
    return float((edge - centre) / edge)
