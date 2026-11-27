"""Material database: elastic, electromagnetic, thermal and X-ray properties.

Units are SI throughout (m, s, kg, S/m, W/(m K), J/(kg K)) except where a name says otherwise
(``atten_db_mm`` is dB/mm at 1 MHz, photon energies are keV).

Ultrasonic attenuation follows a power law α(f) = α₁ (f / 1 MHz)ⁿ [Np/m]; the dispersion that
causality imposes on it (Kramers–Kronig, Szabo 1994) is applied in :mod:`ndttoolbox.core.signals`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

NP_PER_DB = np.log(10) / 20.0          # 1 dB = 0.1151 Np
MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class Material:
    name: str
    rho: float                 # density kg/m³
    cL: float                  # longitudinal velocity m/s
    cT: float                  # shear velocity m/s (0 for fluids)
    atten_db_mm: float = 0.0   # attenuation at 1 MHz, dB/mm
    atten_n: float = 1.0       # attenuation frequency exponent
    sigma: float = 0.0         # electrical conductivity S/m
    mur: float = 1.0           # relative permeability
    k: float = 1.0             # thermal conductivity W/(m K) (in-plane for laminates)
    k_z: float | None = None   # through-thickness conductivity (anisotropic laminates)
    cp: float = 1000.0         # specific heat J/(kg K)
    emissivity: float = 0.9
    composition: dict = field(default_factory=dict)   # element -> mass fraction (radiography)
    grain_um: float = 0.0      # mean grain size (ultrasonic backscatter noise)

    # ---- derived acoustic quantities
    @property
    def ZL(self):
        """Longitudinal acoustic impedance ρ c_L [Rayl]."""
        return self.rho * self.cL

    @property
    def ZT(self):
        return self.rho * self.cT

    @property
    def is_fluid(self):
        return self.cT <= 0

    @property
    def poisson(self):
        if self.is_fluid:
            return 0.5
        r = (self.cL / self.cT) ** 2
        return (r - 2) / (2 * (r - 1))

    @property
    def youngs(self):
        nu = self.poisson
        return 2 * self.rho * self.cT ** 2 * (1 + nu)

    def alpha(self, f):
        """Attenuation α(f) in Np/m for frequency f in Hz."""
        f = np.asarray(f, float)
        a1 = self.atten_db_mm * 1e3 * NP_PER_DB
        return a1 * (np.abs(f) / 1e6) ** self.atten_n

    # ---- thermal
    @property
    def diffusivity(self):
        return self.k / (self.rho * self.cp)

    @property
    def diffusivity_z(self):
        return (self.k_z if self.k_z is not None else self.k) / (self.rho * self.cp)

    @property
    def effusivity(self):
        return np.sqrt((self.k_z or self.k) * self.rho * self.cp)

    # ---- electromagnetic
    def skin_depth(self, f):
        """Standard depth of penetration δ = 1/√(π f μ σ)."""
        return 1.0 / np.sqrt(np.pi * np.asarray(f, float) * MU0 * self.mur * self.sigma)

    @property
    def iacs(self):
        """Conductivity in % IACS (100 % = 58.0 MS/m)."""
        return 100 * self.sigma / 5.8e7

    def with_(self, **kw):
        return replace(self, **kw)


def _m(name, **kw):
    return Material(name=name, **kw)


MATERIALS: dict[str, Material] = {m.name: m for m in [
    _m("steel", rho=7850, cL=5920, cT=3230, atten_db_mm=0.0025, atten_n=1.6, sigma=5.8e6, mur=100,
       k=50, cp=490, emissivity=0.95, composition={"Fe": 0.98, "C": 0.002, "Mn": 0.018},
       grain_um=25),
    _m("stainless-304", rho=7900, cL=5740, cT=3100, atten_db_mm=0.003, atten_n=2.0, sigma=1.39e6,
       k=16.2, cp=500, emissivity=0.95,
       composition={"Fe": 0.70, "Cr": 0.19, "Ni": 0.09, "Mn": 0.02}, grain_um=60),
    _m("austenitic-weld", rho=7900, cL=5650, cT=3050, atten_db_mm=0.012, atten_n=2.0,
       sigma=1.3e6, k=16, cp=500, composition={"Fe": 0.68, "Cr": 0.20, "Ni": 0.10, "Mn": 0.02},
       grain_um=250),
    _m("aluminium-2024", rho=2780, cL=6370, cT=3160, atten_db_mm=0.0015, atten_n=1.5,
       sigma=1.74e7, k=121, cp=875, emissivity=0.95,
       composition={"Al": 0.935, "Cu": 0.045, "Mg": 0.015, "Mn": 0.005}, grain_um=40),
    _m("aluminium-6061", rho=2700, cL=6320, cT=3130, atten_db_mm=0.0015, atten_n=1.5,
       sigma=2.5e7, k=167, cp=896, emissivity=0.95,
       composition={"Al": 0.98, "Mg": 0.01, "Si": 0.006, "Cu": 0.004}, grain_um=40),
    _m("titanium-6al4v", rho=4430, cL=6100, cT=3120, atten_db_mm=0.0012, atten_n=1.8,
       sigma=5.8e5, k=6.7, cp=526, emissivity=0.95,
       composition={"Ti": 0.90, "Al": 0.06, "V": 0.04}, grain_um=30),
    _m("inconel-625", rho=8440, cL=5820, cT=3020, atten_db_mm=0.004, atten_n=2.0, sigma=7.7e5,
       k=9.8, cp=410, emissivity=0.95,
       composition={"Ni": 0.61, "Cr": 0.215, "Mo": 0.09, "Fe": 0.05, "Nb": 0.035}, grain_um=80),
    _m("copper", rho=8960, cL=4660, cT=2260, atten_db_mm=0.004, atten_n=1.5, sigma=5.8e7,
       k=400, cp=385, emissivity=0.95, composition={"Cu": 1.0}),
    _m("cfrp", rho=1550, cL=2950, cT=1500, atten_db_mm=0.08, atten_n=1.3, sigma=1.5e4,
       k=3.5, k_z=0.65, cp=1100, emissivity=0.92, composition={"C": 0.76, "O": 0.12, "H": 0.08,
                                                               "N": 0.04}),
    _m("gfrp", rho=1900, cL=3000, cT=1550, atten_db_mm=0.10, atten_n=1.3, sigma=1e-6,
       k=0.9, k_z=0.35, cp=1000, emissivity=0.92,
       composition={"O": 0.45, "Si": 0.20, "C": 0.20, "Ca": 0.08, "Al": 0.04, "H": 0.03}),
    _m("pmma", rho=1180, cL=2730, cT=1430, atten_db_mm=0.05, atten_n=1.0, k=0.19, cp=1470,
       composition={"C": 0.600, "O": 0.320, "H": 0.080}),
    _m("rexolite", rho=1050, cL=2337, cT=1157, atten_db_mm=0.02, atten_n=1.0, k=0.15, cp=1300,
       composition={"C": 0.923, "H": 0.077}),
    _m("water", rho=1000, cL=1483, cT=0, atten_db_mm=0.000217, atten_n=2.0, sigma=5e-3, k=0.6,
       cp=4182, composition={"H": 0.112, "O": 0.888}),
    _m("air", rho=1.2, cL=343, cT=0, atten_db_mm=0.16, atten_n=1.5, k=0.026, cp=1005,
       composition={"N": 0.755, "O": 0.232, "Ar": 0.013}),
    _m("epoxy", rho=1170, cL=2600, cT=1100, atten_db_mm=0.1, atten_n=1.1, k=0.2, cp=1200,
       composition={"C": 0.72, "O": 0.16, "H": 0.07, "N": 0.05}),
    _m("tungsten", rho=19300, cL=5220, cT=2890, sigma=1.8e7, k=173, cp=134,
       composition={"W": 1.0}),
    _m("lead", rho=11340, cL=2160, cT=700, sigma=4.8e6, k=35, cp=128, composition={"Pb": 1.0}),
    _m("csi", rho=4510, cL=3300, cT=1600, k=1.1, cp=200, composition={"Cs": 0.5115, "I": 0.4885}),
]}


def get(name) -> Material:
    if isinstance(name, Material):
        return name
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {', '.join(MATERIALS)}") from None


# --------------------------------------------------------------------------- elements (X-ray)
# Z, atomic mass, K-edge (keV), K-shell jump ratio
ELEMENTS = {
    "H": (1, 1.008, 0.0, 1.0), "C": (6, 12.011, 0.284, 1.0), "N": (7, 14.007, 0.410, 1.0),
    "O": (8, 15.999, 0.543, 1.0), "Mg": (12, 24.305, 1.30, 1.0), "Al": (13, 26.982, 1.56, 1.0),
    "Si": (14, 28.086, 1.84, 1.0), "Ar": (18, 39.948, 3.21, 1.0), "Ca": (20, 40.078, 4.04, 1.0),
    "Ti": (22, 47.867, 4.97, 8.5), "V": (23, 50.942, 5.47, 8.4), "Cr": (24, 51.996, 5.99, 8.3),
    "Mn": (25, 54.938, 6.54, 8.2), "Fe": (26, 55.845, 7.11, 8.1), "Ni": (28, 58.693, 8.33, 7.9),
    "Cu": (29, 63.546, 8.98, 7.8), "Nb": (41, 92.906, 18.99, 6.9), "Mo": (42, 95.95, 20.0, 6.8),
    "I": (53, 126.90, 33.17, 6.0), "Cs": (55, 132.91, 35.99, 5.9), "W": (74, 183.84, 69.53, 5.0),
    "Pb": (82, 207.2, 88.00, 4.9),
}
