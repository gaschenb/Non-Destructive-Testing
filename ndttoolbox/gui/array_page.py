"""Phased arrays: FMC simulation, TFM / phase-coherence / plane-wave / sectorial imaging,
delay laws and beam fields."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from ..core import phased_array as pa
from ..core import signals as sg
from . import theme
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

S = pa.Scatterer


def _s(kind, x, z, size=1.5, angle=0.0):
    """Defect in display units (mm, degrees)."""
    return dict(kind=kind, x=x, z=z, size=size, angle=angle)


SCENARIOS = [
    dict(name="Side-drilled holes, contact TFM",
         desc="32-element 5 MHz linear array on 40 mm steel; Ø1.5 mm side-drilled holes at "
              "10–30 mm. TFM focuses everywhere at once.",
         values=dict(coupling="contact", n_el=32, pitch=0.6, f0=5, thickness=40,
                      method="tfm", steer=0, focus=20, dr=40),
         defects=[_s("sdh", 0, 20), _s("sdh", -8, 10), _s("sdh", 8, 15), _s("sdh", -5, 30),
               _s("sdh", 10, 30)]),
    dict(name="Vertical crack: TFM vs phase coherence",
         desc="6 mm back-wall-breaking crack. Its faces are invisible to direct TFM; the tip "
              "diffracts. Phase-coherence weighting suppresses grain noise and sharpens the tip.",
         values=dict(coupling="contact", n_el=32, pitch=0.6, f0=5, thickness=25,
                      method="pcf", steer=0, focus=19, dr=35, grain=2),
         defects=[_s("crack", 3, 19, 6, 0)]),
    dict(name="Wedge sectorial scan of a weld",
         desc="16-element 5 MHz array on a 36° Rexolite wedge, 20 mm steel. Sectorial scan "
              "40°–70° (Fermat delays through the wedge) and TFM of the same FMC.",
         values=dict(coupling="wedge", n_el=16, pitch=0.6, f0=5, thickness=20,
                      method="sscan", steer=55, focus=15, dr=35,
                     wangle=36),
         defects=[_s("sdh", 14, 12), _s("sdh", 22, 16), _s("crack", 8, 6, 4, 35)]),
    dict(name="Grating lobes",
         desc="Pitch 1.5 mm at 5 MHz (> λ/2): steering to 30° creates a grating lobe at "
              "sinθ_g = sinθ_s − λ/p. Look at the beam tab.",
         values=dict(coupling="contact", n_el=16, pitch=1.5, f0=5, thickness=40,
                      method="tfm", steer=30, focus=25, dr=40),
         defects=[_s("sdh", 8, 20)]),
    dict(name="Plane-wave compounding",
         desc="11 plane waves (±20°) synthesised from the FMC and compounded: TFM-like quality "
              "from far fewer transmissions.",
         values=dict(coupling="contact", n_el=32, pitch=0.6, f0=5, thickness=40,
                      method="pwi", steer=0, focus=20, dr=40),
         defects=[_s("sdh", 0, 20), _s("sdh", -8, 12), _s("sdh", 8, 28), _s("pore", 4, 10, 1.0)]),
    dict(name="Immersion inspection",
         desc="64-element array 15 mm above an aluminium part; Fermat refraction at the water/"
              "metal interface for every pixel.",
         values=dict(coupling="immersion", n_el=64, pitch=0.5, f0=5, thickness=30,
                      method="tfm", steer=0, focus=15, dr=40, wp=15,
                     material="aluminium-6061"),
         defects=[_s("sdh", 0, 15), _s("sdh", 6, 8), _s("sdh", -6, 22)]),
]


class ArrayPage(Workspace):
    title = "Phased array"
    SCENARIOS = SCENARIOS
    HEAVY_KEYS = ("beam_pulsed", "hires")
    run_label = "Run hi-res image + pulsed beam"
    CHOICES = [
        C("material", "Material", [("Carbon steel", "steel"), ("Austenitic weld", "austenitic-weld"),
                                   ("Aluminium 6061", "aluminium-6061"), ("Ti-6Al-4V", "titanium-6al4v")],
          "steel"),
        C("coupling", "Coupling", [("Contact", "contact"), ("Wedge", "wedge"),
                                   ("Immersion", "immersion")], "contact"),
        C("method", "Imaging", [("TFM", "tfm"), ("TFM × phase coherence", "pcf"),
                                ("Plane-wave compounding", "pwi"), ("Sectorial scan", "sscan")],
          "tfm"),
    ]
    PARAMS = [
        P("n_el", "Elements", 4, 64, 32, 1, integer=True),
        P("pitch", "Pitch (mm)", 0.2, 3, 0.6, 1e-3, decimals=2),
        P("f0", "Frequency (MHz)", 1, 10, 5, 1e6, decimals=2),
        P("thickness", "Thickness (mm)", 5, 80, 40, 1e-3, decimals=1),
        P("wangle", "Wedge angle (°)", 0, 50, 36, 1, decimals=1),
        P("wp", "Water path (mm)", 2, 50, 15, 1e-3, decimals=1),
        P("steer", "Steering (°)", -70, 80, 0, 1, decimals=1),
        P("focus", "Focal depth (mm)", 2, 80, 20, 1e-3, decimals=1),
        P("grain", "Grain noise ×", 0, 6, 1, 1, decimals=2),
        P("dr", "Dynamic range (dB)", 10, 60, 40, 1, decimals=0),
    ]

    def build_views(self):
        self.w_img = W.ImagePanel("Image (dB)", "x (mm)", "z (mm)", cmap="inferno")
        self.w_beam = W.ImagePanel("Transmitted field, CW at f₀ (dB)", "x (mm)", "z (mm)",
                                   cmap="viridis")
        self.w_delay = W.plot("Delay law", "element", "delay (ns)")
        self.w_fmc = W.ImagePanel("FMC: tx = rx (pulse-echo) B-scan", "element", "time (µs)",
                                  bipolar=True)
        self.w_fmc.pi.setAspectLocked(False)
        return [("Image", self.w_img), ("Beam", self.w_beam), ("Delay law", self.w_delay),
                ("FMC data", self.w_fmc)]

    DEFECT_NOTE = "Positions are relative to the array centre (wedge: beam exit point)."
    DEFECTS = [
        C("kind", "Type", [("Side-drilled hole", "sdh"), ("Pore", "pore"), ("Planar crack", "crack")], "sdh"),
        P("x", "x (mm)", -40, 60, 0, 1e-3, decimals=1),
        P("z", "Depth (mm, crack: upper tip)", 0.5, 80, 20, 1e-3, decimals=1),
        P("size", "Diameter / crack height (mm)", 0.3, 20, 1.5, 1e-3, decimals=2),
        P("angle", "Crack angle from vertical (°)", -70, 70, 0, 1, decimals=1),
    ]

    def describe_defect(self, d):
        name = dict(sdh="SDH", pore="pore", crack="crack")[d["kind"]]
        what = f"{d['size']:g} mm high, {d['angle']:g}°" if d["kind"] == "crack" else f"Ø{d['size']:g} mm"
        return f"{name} {what} at x {d['x']:g}, z {d['z']:g} mm"

    def defect_hidden(self, d):
        return () if d["kind"] == "crack" else ("angle",)

    VIEW_HELP = {
        "Image": "Reconstructed image in dB (0 dB = brightest pixel) over the selected dynamic "
                 "range. Dotted outlines are the true defects (numbered), squares on top are the "
                 "array elements, the dashed line is the back wall. Hover for the level at any point.",
        "Beam": "Transmitted pressure field for the current delay law (steering and focus). Run "
                "for the pulsed (broadband) beam, which suppresses grating-lobe and side-lobe "
                "fringes.",
        "Delay law": "Transmit delay applied to each element so that all wavefronts arrive at the "
                     "focal point together (Fermat path through the wedge/water when present).",
        "FMC data": "Raw full-matrix-capture data for transmitter = receiver (a synthetic B-scan "
                    "along the array): one RF trace per element versus time.",
    }

    def equations(self):
        return [
            W.eq("Fermat travel time through an interface",
                 r"$t(\xi)=\dfrac{\sqrt{(\xi-x_e)^2+z_e^2}}{c_1}+\dfrac{\sqrt{(x-\xi)^2+z^2}}{c_2}$",
                 r"$\dfrac{\partial t}{\partial\xi}=0\;\Leftrightarrow\;\dfrac{\sin\theta_1}{c_1}=\dfrac{\sin\theta_2}{c_2}$"),
            W.eq("Element directivity",
                 r"$D(\theta)=\mathrm{sinc}\!\left(\dfrac{k w\sin\theta}{2}\right)\cos\theta$"),
            W.eq("Full-matrix capture",
                 r"$\mathbf{S}(\omega)=\mathbf{E}(\omega)\,\mathrm{diag}(\mathbf{A})\,\mathbf{E}(\omega)^{T},\quad "
                 r"E_{eq}=a_{eq}\,e^{-i\omega t_{eq}-\alpha d_{eq}}$"),
            W.eq("Total focusing method and phase coherence",
                 r"$I(P)=\left|\sum_{i,j}h_{ij}(t_i(P)+t_j(P))\right|$",
                 r"$\mathrm{PCF}(P)=\dfrac{1}{N^2}\left|\sum_{i,j}e^{i\varphi_{ij}(P)}\right|$"),
            W.eq("Delay law, grating lobes, near field",
                 r"$\tau_i=\max_j t_j(F)-t_i(F)$",
                 r"$\sin\theta_g=\sin\theta_s\pm m\,\dfrac{\lambda}{p},\qquad N=\dfrac{A^2}{4\lambda}$"),
            W.eq("Defect monitor", "SNR: peak in a box around the defect versus rms of the image "
                 "elsewhere; detected when SNR ≥ 6 dB. Extent: −6 dB region around the peak."),
        ]

    def setup(self, p, hires=False):
        return pa.ArraySetup(material=p["material"],
                             thickness=p["thickness"], n_el=int(p["n_el"]), pitch=p["pitch"],
                             width=0.85 * p["pitch"], f0=p["f0"], coupling=p["coupling"],
                             wedge_angle=p["wangle"], water_path=p["wp"],
                             grain_scale=p["grain"], n_grains=600 if hires else 350,
                             fs=max(40e6, 8 * p["f0"]))

    def scatterers(self, p):
        return [S(d["kind"], d["x"], d["z"], d["size"], d["angle"] if d["kind"] == "crack" else 0.0)
                for d in p.get("_defects", []) if d["z"] < p["thickness"]]

    def _x_offset(self, s):
        return pa._beam_origin(s) if s.coupling == "wedge" else 0.0

    def compute(self, p, heavy, ctx):
        s = self.setup(p, heavy)
        sc = self.scatterers(p)
        if s.coupling == "wedge":
            x0 = pa._beam_origin(s)
            sc = [S(q.kind, q.x + x0, q.z, q.size, q.angle) for q in sc]
        out = dict(setup=s, scat=sc)
        data = pa.simulate_fmc(s, sc)
        ctx.check()
        n = 170 if heavy else 110
        x, z = pa.image_grid(s, n, n)
        method = p["method"]
        if method == "pwi" and s.coupling != "contact":
            method = "tfm"
        if method in ("tfm", "pcf"):
            I, pcf = pa.tfm(s, data, x, z, coherence=True)
            img = I * pcf if method == "pcf" else I
        elif method == "pwi":
            img = pa.plane_wave_image(s, data, x, z, np.linspace(-20, 20, 11))
        else:
            if s.coupling == "wedge":
                angs = np.linspace(40, 75, 71)
            else:
                angs = np.linspace(-50, 50, 101)
            img = pa.sector_scan(s, data, x, z, angs, focus=p["focus"])
        ctx.check()
        out["meas"] = self.measure(img, x, z, sc)
        out.update(x=x, z=z, img=img, method=method, t=data["t"], t_delay=data["t_delay"],
                   pe=np.array([data["fmc"][i, i] for i in range(s.n_el)]))
        d = pa.delay_law(s, p["steer"], p["focus"])
        out["delays"] = d
        xb, zb = pa.image_grid(s, 90 if not heavy else 150, 90 if not heavy else 150)
        out["beam"] = (xb, zb, pa.beam_field(s, xb, zb, d, mode="cw"))
        if heavy:
            out["beam_pulsed"] = (xb, zb, pa.beam_field(s, xb, zb, d, mode="pulsed"))
            out["hires"] = True
        return out

    @staticmethod
    def _centre(q):
        if q.kind == "crack":
            a = np.radians(q.angle)
            return q.x + 0.5 * q.size * np.sin(a), q.z + 0.5 * q.size * np.cos(a)
        return q.x, q.z

    def measure(self, img, x, z, sc):
        X, Z = np.meshgrid(x, z)
        ref = img.max() if img.size else 1.0
        T = self.params.values()["thickness"]
        background = (Z > 1e-3) & (Z < T - 2e-3)             # exclude surface and back wall
        for q in sc:
            cx, cz = self._centre(q)
            hw = 2e-3 + 0.6 * q.size
            background &= ~((np.abs(X - cx) < hw) & (np.abs(Z - cz) < hw))
        out = []
        for q in sc:
            cx, cz = self._centre(q)
            hw = 2e-3 + 0.6 * q.size
            box = (np.abs(X - cx) < hw) & (np.abs(Z - cz) < hw)
            if not box.any():
                out.append(None); continue
            sub = np.where(box, img, 0)
            k = np.unravel_index(np.argmax(sub), img.shape)
            rest = img[background] if background.sum() > 50 else img[~box]
            rms = float(np.sqrt(np.mean(rest ** 2))) if rest.size else 1e-30
            snr = 20 * np.log10(max(sub.max(), 1e-30) / max(rms, 1e-30))
            try:
                dx, dz = pa.amplitude_size(sub, x, z)
            except Exception:
                dx = dz = np.nan
            out.append(dict(x=float(X[k]), z=float(Z[k]), db=20 * np.log10(max(sub.max(), 1e-30) / ref),
                            snr=snr, dx=dx, dz=dz, detected=snr >= 6.0))
        return out

    def monitor(self, r):
        heads = ["#", "type", "true x, z", "peak x, z", "level dB", "SNR dB", "−6 dB extent", "status"]
        x0 = self._x_offset(r["setup"])
        rows, flags = [], []
        for i, (q, m) in enumerate(zip(r["scat"], r["meas"])):
            cx, cz = self._centre(q)
            if m is None:
                rows.append([str(i + 1), q.kind, f"{(cx - x0) * 1e3:.1f}, {cz * 1e3:.1f}", "outside image",
                             "—", "—", "—", "missed"]); flags.append(False); continue
            rows.append([str(i + 1), q.kind, f"{(cx - x0) * 1e3:.1f}, {cz * 1e3:.1f} mm",
                         f"{(m['x'] - x0) * 1e3:.1f}, {m['z'] * 1e3:.1f} mm", f"{m['db']:.1f}",
                         f"{m['snr']:.1f}", f"{m['dx'] * 1e3:.1f} × {m['dz'] * 1e3:.1f} mm",
                         "detected" if m["detected"] else "missed"])
            flags.append(m["detected"])
        return heads, rows, flags

    def show_result(self, r):
        s = r["setup"]
        p = self.params.values()
        x0 = self._x_offset(s)
        img = r["img"]
        names = dict(tfm="TFM", pcf="TFM × phase-coherence factor", pwi="Plane-wave compounding",
                     sscan="Sectorial scan")
        self.w_img.set(sg.db(img, floor=-p["dr"]), (r["x"] - x0) * 1e3, r["z"] * 1e3,
                       levels=(-p["dr"], 0), title=f"{names[r['method']]} (dB)")
        self.w_img.clear_overlays()
        el, _ = s.elements()
        self.w_img.add(pg.PlotDataItem((el[:, 0] - x0) * 1e3, el[:, 1] * 1e3,
                                       pen=None, symbol="s", symbolSize=4,
                                       symbolBrush=theme.color(5)))
        self.w_img.add(pg.InfiniteLine(pos=s.thickness * 1e3, angle=0, pen=theme.pen("data", 1, "dash")))
        for i, q in enumerate(r["scat"]):
            cx, cz = self._centre(q)
            lab = pg.TextItem(str(i + 1), color=theme.color(5), anchor=(-0.4, 1.2))
            lab.setPos((cx - x0 + 0.5 * q.size) * 1e3, (cz - 0.5 * q.size) * 1e3)
            self.w_img.add(lab)
            if q.kind == "crack":
                a = np.radians(q.angle)
                xs = (np.array([q.x, q.x + q.size * np.sin(a)]) - x0) * 1e3
                zs = np.array([q.z, q.z + q.size * np.cos(a)]) * 1e3
                self.w_img.add(pg.PlotDataItem(xs, zs, pen=theme.pen(5, 1, "dot")))
            else:
                c = pg.QtWidgets.QGraphicsEllipseItem((q.x - x0 - q.size / 2) * 1e3,
                                                      (q.z - q.size / 2) * 1e3,
                                                      q.size * 1e3, q.size * 1e3)
                c.setPen(theme.pen(5, 1, "dot"))
                self.w_img.add(c)
        zmin = min(el[:, 1].min() * 1e3, 0) - 1
        self.w_img.pi.setYRange(zmin, s.thickness * 1.1e3)
        key = "beam_pulsed" if "beam_pulsed" in r else "beam"
        xb, zb, B = r[key]
        self.w_beam.set(sg.db(B, floor=-40), (xb - x0) * 1e3, zb * 1e3, levels=(-40, 0),
                        title="Transmitted field, " + ("pulsed peak" if key == "beam_pulsed" else
                                                       "CW at f₀") + " (dB)")
        pd = self.w_delay.getPlotItem(); pd.clear()
        bg = pg.BarGraphItem(x=np.arange(1, s.n_el + 1), height=r["delays"] * 1e9, width=0.75,
                             brush=theme.color(0))
        pd.addItem(bg)
        pd.setTitle(f"Delay law: steer {p['steer']:.1f}°, focus {p['focus'] * 1e3:.1f} mm")
        t = r["t"] * 1e6
        pe = r["pe"]
        m = np.max(np.abs(pe[:, t > (r["t_delay"] * 1e6 + 1.5)])) if np.any(t > 1.5) else 1
        self.w_fmc.set(np.clip(pe.T / max(m, 1e-12), -1, 1), np.arange(1, s.n_el + 1), t,
                       levels=(-1, 1))

    def summary(self, r):
        s = r["setup"]
        lam = s.c / s.f0
        p = self.params.values()
        rows = [("wavelength λ", W.fmt(lam * 1e3, "mm")),
                ("pitch / λ", W.fmt(s.pitch / lam, "")),
                ("active aperture", W.fmt(s.aperture() * 1e3, "mm")),
                ("aperture near field", W.fmt(pa.array_near_field(s) * 1e3, "mm"))]
        gl = pa.grating_lobe_angles(s.pitch, s.f0, s.c, p["steer"])
        rows.append(("grating lobes", ", ".join(f"{g:.1f}°" for g in gl) if gl else "none"))
        x0 = self._x_offset(s)
        meas = [m for m in r.get("meas", []) if m is not None]
        if r["scat"]:
            rows.append(("defects detected (SNR ≥ 6 dB)",
                         f"{sum(m['detected'] for m in meas)} / {len(r['scat'])}"))
        rows.append(("max delay", W.fmt(r["delays"].max() * 1e9, "ns")))
        rows.append(("FMC size", f"{s.n_el}² × {len(r['t'])} samples"))
        return rows
