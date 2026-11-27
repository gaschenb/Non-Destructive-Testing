"""Industrial computed tomography: polychromatic sinograms, FBP, beam hardening, noise."""
from __future__ import annotations

import threading

import numpy as np

from ..core import radiography as rt
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

SCENARIOS = [
    dict(name="Steel pipe with corrosion pits",
         desc="Water-filled 100 mm steel pipe, 8 mm wall, inner-wall pits and an outer notch. "
              "300 kV with 1 mm Cu: beam hardening produces cupping and streaks; linearisation "
              "removes most of it.",
         values=dict(phantom="pipe", kv=300, filt=1.0, n=256, angles=360, photons=5, recon="hann",
                     bhc=0, mono=0)),
    dict(name="Same pipe, beam-hardening corrected",
         desc="Polynomial linearisation of the steel p(L) curve to μ at the beam-hardened effective energy.",
         values=dict(phantom="pipe", kv=300, filt=1.0, n=256, angles=360, photons=5, recon="hann",
                     bhc=1, mono=0)),
    dict(name="Aluminium casting with porosity",
         desc="Al casting with a cored hole, gas pores of 0.8–3 mm and a steel insert that "
              "throws metal streaks at 160 kV.",
         values=dict(phantom="casting", kv=160, filt=0.5, n=256, angles=360, photons=5,
                     recon="hann", bhc=0, mono=0)),
    dict(name="CFRP panel with Ti insert",
         desc="12 mm CFRP laminate with 0.5 mm delaminations and a titanium fastener at 80 kV.",
         values=dict(phantom="composite", kv=80, filt=0.0, n=256, angles=360, photons=5,
                     recon="hann", bhc=0, mono=0)),
    dict(name="Angular undersampling",
         desc="Only 45 projections: streak artefacts appear beyond r ≈ N_θ·Δx/π·2.",
         values=dict(phantom="pipe", kv=300, filt=1.0, n=256, angles=45, photons=5, recon="hann",
                     bhc=1, mono=0)),
    dict(name="Photon-starved scan",
         desc="10³ photons per ray: noise dominates. Compare the ramp and Hann filters.",
         values=dict(phantom="casting", kv=160, filt=0.5, n=256, angles=360, photons=3,
                     recon="ramp", bhc=0, mono=0)),
    dict(name="Monochromatic reference",
         desc="Synchrotron-like monochromatic beam: no beam hardening, no cupping.",
         values=dict(phantom="pipe", kv=300, filt=1.0, n=256, angles=360, photons=5, recon="hann",
                     bhc=0, mono=1)),
]

_PRIMARY = {"pipe": "steel", "casting": "aluminium-6061", "composite": "cfrp"}


class CTPage(Workspace):
    title = "Computed tomography"
    SCENARIOS = SCENARIOS
    CHOICES = [
        C("phantom", "Object", [("Steel pipe (water filled)", "pipe"), ("Aluminium casting", "casting"),
                                ("CFRP panel with Ti insert", "composite")], "pipe"),
        C("recon", "FBP filter", [("Ramp (Ram-Lak)", "ramp"), ("Shepp-Logan", "shepp-logan"),
                                  ("Hann", "hann")], "hann"),
        C("bhc", "Beam-hardening correction", [("Off", 0), ("Linearisation", 1)], 0),
        C("mono", "Spectrum", [("Polychromatic tube", 0), ("Monochromatic (mean energy)", 1)], 0),
        C("n", "Grid (pixels)", [("128", 128), ("192", 192), ("256", 256), ("384", 384)], 256),
    ]
    PARAMS = [
        P("kv", "Tube voltage (kV)", 40, 450, 300, 1, decimals=0),
        P("filt", "Cu filter (mm)", 0, 4, 1.0, 1, decimals=2),
        P("angles", "Projections", 12, 720, 360, 1, integer=True),
        P("photons", "log₁₀ photons per ray", 2, 7, 5, 1, decimals=2),
    ]

    def __init__(self, *a, **k):
        self._cache = {}
        self._lock = threading.Lock()
        super().__init__(*a, **k)

    def build_views(self):
        self.w_rec = W.ImagePanel("Reconstruction μ (1/cm)", "x (mm)", "y (mm)", cmap="CET-L1",
                                  invert_y=False)
        self.w_err = W.ImagePanel("Reconstruction − truth (1/cm)", "x (mm)", "y (mm)",
                                  invert_y=False, bipolar=True)
        self.w_sino = W.ImagePanel("Sinogram  p = −ln(I/I₀)", "detector (mm)", "angle (°)",
                                   cmap="viridis", invert_y=False)
        self.w_sino.pi.setAspectLocked(False)
        self.w_prof = W.plot("Profile through the centre", "x (mm)", "μ (1/cm)", legend=True)
        self.w_bh = W.plot("Beam hardening: p vs path length", "path length (mm)",
                           "p = −ln(I/I₀)", legend=True)
        return [("Reconstruction", self.w_rec), ("Error", self.w_err), ("Sinogram", self.w_sino),
                ("Profile", self.w_prof), ("Beam hardening", self.w_bh)]

    VIEW_HELP = {
        "Reconstruction": "Filtered back-projection of the simulated polychromatic sinogram: "
                          "linear attenuation μ in 1/cm at the effective energy. Hover for the "
                          "value at any pixel.",
        "Error": "Reconstruction minus the true μ at the effective energy: cupping (beam "
                 "hardening), streaks (metal, under-sampling) and noise show up here.",
        "Sinogram": "Line integrals p = −ln(I/I₀) for every detector position and view angle; "
                    "each point of the object traces a sinusoid.",
        "Profile": "Reconstructed and true μ along the horizontal line through the centre.",
        "Beam hardening": "Measured p versus path length through the main material: "
                          "polychromatic (concave), monochromatic at the effective energy "
                          "(straight), and after the linearisation correction.",
    }

    def equations(self):
        return [
            W.eq("Polychromatic projection",
                 r"$p(s,\theta)=-\ln\!\int w(E)\,\exp\!\left(-\!\int\mu(E,\ell)\,d\ell\right)dE$",
                 r"$w\propto N(E)\,\eta(E)\,E,\qquad\sigma_p\approx\dfrac{1}{\sqrt{N_0\,e^{-p}}}$"),
            W.eq("Filtered back-projection",
                 r"$\mu(x,y)=\int_0^{\pi}(p*h)(x\cos\theta+y\sin\theta,\,\theta)\,d\theta$",
                 r"$H(f)=|f|\,W(f),\qquad W\in\left\{1,\ \mathrm{sinc}\,f,\ "
                 r"\frac{1}{2}(1+\cos 2\pi f)\right\}$"),
            W.eq("Beam hardening",
                 r"$p(L)$ concave;   $p\ \mapsto\ \bar{\mu}L=\sum_k c_k\,p^k$",
                 "Low energies are removed first; the polynomial is fitted to the forward model."),
            W.eq("Angular sampling",
                 r"$N_\theta\gtrsim\dfrac{\pi}{2}\,N_{det}$"),
        ]

    def paths(self, kind, n, na):
        key = (kind, n, na)
        with self._lock:
            if key not in self._cache:
                maps, px = rt.phantom(kind, n)
                a = np.linspace(0, 180, na, endpoint=False)
                L = {m: rt.radon(v, a) * px for m, v in maps.items()}
                if len(self._cache) > 6:
                    self._cache.clear()
                self._cache[key] = (maps, px, a, L)
            return self._cache[key]

    def compute(self, p, heavy, ctx):
        kind, n, na = p["phantom"], int(p["n"]), int(p["angles"])
        maps, px, a, L = self.paths(kind, n, na)
        ctx.check()
        src = rt.Source(kind="xray", kv=p["kv"], filter_mm=p["filt"])
        E, N = src.spectrum(100)
        w = N * rt.Detector().absorption(E) * E
        w = w / w.sum()
        prim = _PRIMARY[kind]
        lp = L[prim][L[prim] > 0]
        l_ref = float(np.median(lp)) if lp.size else 0.0
        e_ref = rt.effective_energy(prim, (E, w), l_ref)      # beam-hardened reference energy
        sino, _, ew = rt.ct_sinogram(maps, px, src, a, photons_per_ray=10 ** p["photons"],
                                     mono_kev=e_ref if p["mono"] else None, paths=L)
        if p["bhc"] and not p["mono"]:
            sino = rt.beam_hardening_correction(sino, prim, (E, w), max_len=0.2,
                                                mu_ref=float(rt.mu(prim, e_ref)))
        ctx.check()
        img = rt.fbp(sino, a, px, p["recon"])
        truth = sum(float(rt.mu(m, e_ref)) * v for m, v in maps.items())
        e_mean = float(np.sum(w * E))
        # beam-hardening curves
        Ls = np.linspace(0, 0.1, 200)
        p_poly = -np.log(np.tensordot(w, np.exp(-rt.mu(prim, E)[:, None] * Ls[None]), axes=(0, 0)))
        m_eff = float(rt.mu(prim, e_ref))
        mask = maps[prim] > 0.5
        return dict(img=img, truth=truth, sino=sino, a=a, px=px, n=n, maps=maps, prim=prim,
                    e_mean=e_mean, e_ref=e_ref, l_ref=l_ref, bh=(Ls, p_poly, m_eff), mask=mask,
                    cup=rt.cupping(img, mask) if kind != "pipe" else None)

    def show_result(self, r):
        n, px = r["n"], r["px"]
        x = ((np.arange(n) + 0.5) * px - n * px / 2) * 1e3
        img, truth = r["img"] / 100, r["truth"] / 100
        hi = 1.5 * float(np.mean(truth[r["mask"]])) if r["mask"].any() else 1.0
        self.w_rec.set(img, x, x, levels=(-0.05 * hi, hi))
        err = img - truth
        m = np.percentile(np.abs(err), 99) or 1
        self.w_err.set(err, x, x, levels=(-m, m))
        s = x
        self.w_sino.set(r["sino"], s, r["a"], levels=(0, np.percentile(r["sino"], 99.8)))
        pr = self.w_prof.getPlotItem(); pr.clear()
        if pr.legend is not None:
            pr.legend.clear()
        row = n // 2
        W.curve(pr, x, img[row], 0, 1.3, name="reconstruction")
        W.curve(pr, x, truth[row], 1, 1.8, name="truth at the effective energy", style="dash")
        pb = self.w_bh.getPlotItem(); pb.clear()
        if pb.legend is not None:
            pb.legend.clear()
        Ls, pp, me = r["bh"]
        W.curve(pb, Ls * 1e3, pp, 0, 2, name=f"polychromatic ({r['prim']})")
        W.curve(pb, Ls * 1e3, me * Ls, 1, 1.6, name="μ(E_eff)·L (no hardening)", style="dash")

    def summary(self, r):
        img, truth, mask = r["img"], r["truth"], r["mask"]
        rmse = float(np.sqrt(np.mean((img - truth) ** 2))) / 100
        bg = truth == 0
        noise = float(np.std(img[bg])) / 100 if bg.any() else 0
        mu_m = float(np.mean(img[mask])) / 100
        mu_t = float(np.mean(truth[mask])) / 100
        Ls, pp, me = r["bh"]
        rows = [("mean detected energy (open beam)", W.fmt(r["e_mean"], "keV")),
                (f"effective energy (after {r['l_ref'] * 1e3:.0f} mm)", W.fmt(r["e_ref"], "keV")),
                (f"μ {r['prim']} at E_eff", W.fmt(mu_t, "1/cm")),
                (f"μ {r['prim']} (reconstructed)", W.fmt(mu_m, "1/cm")),
                ("bias", W.fmt((mu_m / mu_t - 1) * 100, "%") if mu_t else "—"),
                ("noise σ in air", W.fmt(noise, "1/cm")),
                ("RMSE", W.fmt(rmse, "1/cm")),
                ("p / μ(E_eff)·L at 50 mm", W.fmt(pp[100] / (me * Ls[100]))),
                ("pixel", W.fmt(r["px"] * 1e3, "mm")),
                ("projections / needed", f"{len(r['a'])} / {int(np.pi / 2 * r['n'])}")]
        if r["cup"] is not None:
            rows.insert(4, ("cupping index", W.fmt(r["cup"] * 100, "%")))
        return rows
