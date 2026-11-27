"""Eddy-current testing: impedance plane, flaw Lissajous signals, penetration, pulsed EC."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from ..core import eddy_current as ec
from ..core.materials import get
from . import theme
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

MATS = [("Aluminium 2024 (30 % IACS)", "aluminium-2024"), ("Aluminium 6061 (43 % IACS)", "aluminium-6061"),
        ("Copper", "copper"), ("Ti-6Al-4V", "titanium-6al4v"), ("Stainless 304", "stainless-304"),
        ("Inconel 625", "inconel-625"), ("Carbon steel (μr = 100)", "steel"), ("CFRP", "cfrp")]


def _f(kind, x, y, depth, size, angle=0.0):
    """Defect of an example problem, display units (mm, °)."""
    return dict(kind=kind, x=x, y=y, depth=depth, size=size, angle=angle)


SCENARIOS = [
    dict(name="Surface crack, pencil probe",
         desc="2 mm long, 1 mm deep surface-breaking fatigue crack in 2024 aluminium, 100 kHz "
              "absolute pencil probe. The flaw signal leaves the lift-off direction at a large "
              "angle — the basis of crack detection.",
         defects=[_f("crack", 0, 0, 1.0, 2.0)],
         values=dict(material="aluminium-2024", f=100, r1=0.6, r2=1.2, height=1.0, turns=150,
                     liftoff=0.15, thickness=10, rot=0, probe="abs")),
    dict(name="Subsurface voids: phase lag",
         desc="Three Ø0.8 mm voids at 0.6, 1.2 and 2.4 mm depth in 6061, 20 mm apart: the signal "
              "phase rotates by ≈ 2d/δ radians and the amplitude decays as e^(−d/δ). Compare "
              "the rows of the defect monitor.",
         defects=[_f("void", -20, 0, 0.6, 0.8), _f("void", 0, 0, 1.2, 0.8),
                  _f("void", 20, 0, 2.4, 0.8)],
         values=dict(material="aluminium-6061", f=20, r1=2, r2=4, height=2, turns=200,
                     liftoff=0.3, thickness=10, rot=0, probe="abs")),
    dict(name="Second-layer corrosion in a lap joint",
         defects=[_f("void", 0, 0, 2.6, 1.6)],
         desc="Two 1.6 mm 2024 skins (dielectric sealant between): a corrosion pit in the "
              "second layer, 5 kHz to reach it. Differential probe.",
         values=dict(material="aluminium-2024", f=5, r1=3, r2=6, height=3, turns=300,
                     liftoff=0.3, thickness=1.6, rot=0, probe="diff", layers=2), layers=2),
    dict(name="Conductivity sorting",
         desc="Impedance plane at 60 kHz: the conductivity locus separates alloys and tempers; "
              "lift-off moves along a different direction.",
         values=dict(material="aluminium-6061", f=60, r1=2, r2=4, height=2, turns=200,
                     liftoff=0.2, thickness=20, rot=0,
                     probe="abs")),
    dict(name="Thin-sheet thickness gauging",
         desc="1 mm aluminium sheet: below f ≈ δ-matched frequency the impedance point moves "
              "with thickness (thickness locus) — eddy-current thickness gauging.",
         values=dict(material="aluminium-2024", f=10, r1=2, r2=4, height=2, turns=200,
                     liftoff=0.2, thickness=1.0, rot=0,
                     probe="abs")),
    dict(name="Pulsed eddy current: wall loss under insulation",
         desc="Carbon-steel wall (σ = 5.8 MS/m, μr = 100) under 25 mm insulation. The late-time "
              "decay of the transient is set by the wall thickness (τ ∝ μσd²).",
         values=dict(material="steel", f=1, r1=10, r2=20, height=10, turns=400, liftoff=25,
                     thickness=10, rot=0, probe="abs")),
    dict(name="Ferromagnetic weld toe crack",
         desc="Carbon steel (μr = 100): very small skin depth even at 100 kHz, so only "
              "surface-breaking cracks are seen; lift-off is the dominant disturbance.",
         values=dict(material="steel", f=100, r1=1, r2=2, height=1.5, turns=150, liftoff=0.2,
                     thickness=10, rot=0, probe="abs"),
         defects=[_f("crack", 0, 0, 1.5, 4.0), _f("crack", 12, 0, 0.3, 1.5)]),
]


class EddyPage(Workspace):
    title = "Eddy current"
    SCENARIOS = SCENARIOS
    HEAVY_KEYS = ("cmap",)
    run_label = "Run C-scan of defects"
    CHOICES = [
        C("material", "Material", MATS, "aluminium-2024"),
        C("probe", "Probe", [("Absolute", "abs"), ("Differential", "diff")], "abs"),
    ]
    PARAMS = [
        P("f", "Frequency (kHz)", 0.1, 5000, 100, 1e3, log=True, decimals=2),
        P("liftoff", "Lift-off (mm)", 0.02, 40, 0.15, 1e-3, log=True, decimals=3),
        P("r1", "Coil inner radius (mm)", 0.2, 30, 0.6, 1e-3, decimals=2),
        P("r2", "Coil outer radius (mm)", 0.4, 40, 1.2, 1e-3, decimals=2),
        P("height", "Coil height (mm)", 0.2, 30, 1.0, 1e-3, decimals=2),
        P("turns", "Turns", 10, 2000, 150, 1, integer=True),
        P("thickness", "Plate thickness (mm)", 0.1, 50, 10, 1e-3, log=True, decimals=2),
        P("noise", "Instrument noise (mΩ rms)", 0.001, 10, 0.02, 1e-3, log=True, decimals=3),
        P("rot", "Display rotation (°)", -180, 180, 0, 1, decimals=0),
    ]

    def build_views(self):
        self.w_z = W.plot("Normalised impedance plane", "R/X₀", "X/X₀", legend=True)
        fw = QtWidgets.QWidget(); fv = QtWidgets.QHBoxLayout(fw); fv.setContentsMargins(0, 0, 0, 0)
        self.w_liss = W.plot("Lissajous (lift-off horizontal)", "horizontal (mΩ)",
                             "vertical (mΩ)", legend=True)
        self.w_liss.getPlotItem().setAspectLocked(True)
        self.w_trace = W.plot("Signal along the scan", "x (mm)", "mΩ", legend=True)
        fv.addWidget(self.w_liss, 1); fv.addWidget(self.w_trace, 1)
        self.w_cmap = W.ImagePanel("C-scan |ΔZ| (mΩ)", "x (mm)", "y (mm)", cmap="viridis",
                                   invert_y=False)
        self.w_pen = W.plot("Eddy-current density vs depth", "depth (mm)", "|J|/|J₀|", legend=True)
        self.w_pec = W.plot("Pulsed eddy current: transient after switch-off", "time (s)",
                            "|v(t)| (V)", legend=True)
        self.w_pec.getPlotItem().setLogMode(x=True, y=True)
        return [("Impedance plane", self.w_z), ("Flaw signal", fw), ("C-scan", self.w_cmap),
                ("Penetration", self.w_pen), ("Pulsed EC", self.w_pec)]

    DEFECT_NOTE = "x, y: position on the surface; depth: crack depth, or centre depth of a void."
    DEFECTS = [
        C("kind", "Type", [("Surface crack", "crack"), ("Embedded crack", "subcrack"),
                           ("Void / pit", "void")], "crack"),
        P("x", "x (mm)", -40, 40, 0, 1e-3, decimals=1),
        P("y", "y (mm)", -40, 40, 0, 1e-3, decimals=1),
        P("depth", "Depth (mm)", 0.05, 10, 1.0, 1e-3, decimals=2),
        P("size", "Length / diameter (mm)", 0.1, 10, 2.0, 1e-3, decimals=2),
        P("angle", "Crack normal angle from x (°)", -90, 90, 0, 1, decimals=0),
    ]

    def describe_defect(self, d):
        name = dict(crack="surface crack", subcrack="embedded crack", void="void")[d["kind"]]
        what = (f"Ø{d['size']:g} mm at {d['depth']:g} mm" if d["kind"] == "void" else
                f"{d['size']:g} mm long, {d['depth']:g} mm deep")
        return f"{name} {what}, ({d['x']:g}, {d['y']:g})"

    def defect_hidden(self, d):
        return ("angle",) if d["kind"] == "void" else ()

    VIEW_HELP = {
        "Impedance plane": "Coil impedance normalised by its reactance in air X₀: the frequency "
                           "locus (thick), the lift-off trajectory (dashed), the conductivity "
                           "and thickness loci and reference alloys. The large dot is the "
                           "operating point. Hover any curve for its value.",
        "Flaw signal": "Left: the impedance change ΔZ seen while the probe passes over each "
                       "defect (Lissajous figure), rotated so that lift-off is horizontal; the "
                       "vertical component is what separates a defect from wobble. Right: both "
                       "components along the scan line y = 0 through all defects.",
        "C-scan": "Run to map |ΔZ| over the surface around the defects (absolute probe). Numbers "
                  "mark the true defect positions.",
        "Penetration": "Eddy-current density versus depth under the coil winding, compared with "
                       "the plane-wave skin-depth law; vertical lines at 1, 2 and 3 δ.",
        "Pulsed EC": "Transient voltage after the drive current is switched off, for walls 50 – "
                     "125 % of the set thickness: the late-time decay rate measures wall "
                     "thickness.",
    }

    def equations(self):
        return [
            W.eq("Dodd–Deeds coil over a layered conductor",
                 r"$\Delta Z=\dfrac{j\omega\pi\mu_0N^2}{(r_2-r_1)^2(l_2-l_1)^2}\int_0^\infty"
                 r"\dfrac{P(\alpha)^2}{\alpha^6}\left(e^{-\alpha l_1}-e^{-\alpha l_2}\right)^2"
                 r"\phi(\alpha)\,d\alpha$",
                 r"$P(\alpha)=\int_{\alpha r_1}^{\alpha r_2}xJ_1(x)\,dx,\qquad"
                 r"\alpha_i=\sqrt{\alpha^2+j\omega\mu_0\mu_i\sigma_i}$",
                 r"$\phi_{\mathrm{half}}=\dfrac{\mu\alpha-\alpha_1}{\mu\alpha+\alpha_1}$",
                 "Layers are folded in from the bottom with the transmission-line recursion "
                 "for the surface admittance."),
            W.eq("Skin depth and phase lag",
                 r"$\delta=\dfrac{1}{\sqrt{\pi f\mu_0\mu_r\sigma}},\qquad"
                 r"|J(z)|\approx|J_0|\,e^{-z/\delta},\qquad\theta\approx\dfrac{z}{\delta}$"),
            W.eq("Defect signal by reciprocity (Auld)",
                 r"$\Delta Z=\dfrac{\sigma}{I^2}\int_V\mathbf{E}_0\cdot\mathbf{E}\,dV$",
                 r"Void:  $\mathbf{E}=\frac{3}{2}\mathbf{E}_0$",
                 r"Thin crack:  $\Delta Z\propto\frac{8}{3}a^3(\mathbf{E}_0\cdot\hat{n})^2$"),
            W.eq("Pulsed eddy current",
                 r"$v(t)=-\mathcal{L}^{-1}\!\left\{\Delta L(s)-\Delta L(\infty)\right\},"
                 r"\qquad\tau\propto\mu\sigma d^2$",
                 "Inverse Laplace transform by the Gaver–Stehfest algorithm."),
            W.eq("Defect monitor",
                 "Each defect is scanned on its own along x. Phase is measured from the lift-off "
                 "direction; SNR is the peak vertical (lift-off-free) component over the "
                 "instrument noise. Detected when SNR ≥ 6 dB."),
        ]

    # ------------------------------------------------------------------
    def coil(self, p):
        r1, r2 = min(p["r1"], p["r2"] * 0.95), p["r2"]
        return ec.Coil(r1, r2, p["liftoff"], p["height"], int(p["turns"]))

    def specimen(self, p, thickness=None):
        m = get(p["material"])
        t = thickness or p["thickness"]
        if self.scenario.get("layers") == 2:
            return ec.Specimen([ec.Layer(t, m.sigma, m.mur), ec.Layer(0.1e-3, 0.0, 1.0),
                                ec.Layer(t, m.sigma, m.mur)])
        return ec.Specimen.plate(t, m.sigma, m.mur)

    def flaw(self, p):
        return [ec.ECFlaw(d["kind"], x=d["x"], y=d["y"], depth=d["depth"], size=d["size"],
                          angle=0.0 if d["kind"] == "void" else d["angle"])
                for d in p.get("_defects", [])]

    def _grid(self, c, fl, n):
        pad = 4 * c.r2 + 1e-3
        xs = np.linspace(min(q.x for q in fl) - pad, max(q.x for q in fl) + pad, n)
        ys = np.linspace(min(q.y for q in fl) - pad, max(q.y for q in fl) + pad, n)
        return xs, ys

    def measure(self, c, sp, f, fl, p, lo_phase):
        """Scan each defect on its own: peak |ΔZ|, phase from lift-off, SNR."""
        out = []
        for q in fl:
            xs = q.x + np.linspace(-4 * c.r2 - 1e-3, 4 * c.r2 + 1e-3, 81)
            dz = ec.flaw_scan(c, sp, f, [q], xs, np.array([q.y]))
            dz = np.atleast_1d(np.squeeze(dz))
            if p["probe"] == "diff":
                dz = ec.differential(dz, xs, max(c.r2, 1e-3))
            z = ec.rotate_phase(dz, -lo_phase + 180)
            k = int(np.argmax(np.abs(z)))
            ph = ((np.degrees(np.angle(z[k])) + 180) % 360) - 180
            vert = float(np.max(np.abs(z.imag)))
            snr = 20 * np.log10(max(vert, 1e-30) / max(p["noise"], 1e-30))
            out.append(dict(xs=xs, z=z, amp=float(np.abs(z[k])), phase=ph, vert=vert, snr=snr,
                            detected=snr >= 6.0))
        return out

    def compute(self, p, heavy, ctx):
        c = self.coil(p)
        sp = self.specimen(p)
        m = get(p["material"])
        f = p["f"]
        out = dict(coil=c, spec=sp, mat=m, f=f)
        fn = ec.characteristic_frequency(c, m.sigma, m.mur)
        fl = self.flaw(p)
        out["flaws"] = fl
        if heavy:
            if fl:
                xs, ys = self._grid(c, fl, 45)
                dz = ec.flaw_scan(c, sp, f, fl, xs, ys)
                out["cmap"] = (xs, ys, dz)
            return out
        freqs = np.geomspace(max(fn * 1e-2, 1.0), fn * 1e3, 90)
        out["floc"] = (freqs, ec.normalized_impedance(c, sp, freqs))
        out["point"] = ec.normalized_impedance(c, sp, [f])
        los = np.linspace(max(c.liftoff * 0.1, 1e-5), c.liftoff + 3 * c.r2, 25)
        lo_pts = [ec.normalized_impedance(ec.Coil(c.r1, c.r2, lo, c.height, c.turns), sp, [f])
                  for lo in los]
        out["lift"] = np.array([[a[0][0], a[1][0]] for a in lo_pts])
        ctx.check()
        sig = np.geomspace(5e5, 6e7, 30)
        out["cond"] = (sig, ec.conductivity_locus(c, f, sig, m.mur, p["thickness"]))
        th = np.geomspace(0.05e-3, 20e-3, 30)
        out["thk"] = (th, ec.thickness_locus(c, f, m.sigma, th, m.mur))
        refs = {}
        for name in ("aluminium-2024", "aluminium-6061", "copper", "titanium-6al4v",
                     "stainless-304"):
            mm = get(name)
            refs[name] = ec.normalized_impedance(c, ec.Specimen.plate(p["thickness"], mm.sigma,
                                                                      mm.mur), [f])
        out["refs"] = refs
        ctx.check()
        # defect signals
        out["lo_phase"] = ec.liftoff_phase(c, sp, f)
        if fl:
            out["meas"] = self.measure(c, sp, f, fl, p, out["lo_phase"])
            ctx.check()
            xs, _ = self._grid(c, fl, 161)
            dz = ec.flaw_scan(c, sp, f, fl, xs)
            if p["probe"] == "diff":
                dz = ec.differential(dz, xs, max(c.r2, 1e-3))
            out["scan"] = (xs, dz)
        # penetration
        rbar = 0.5 * (c.r1 + c.r2)
        tot = sum(L.thickness for L in sp.layers) if sp.layers else 5 * float(ec.skin_depth(f, m.sigma, m.mur))
        dep = np.linspace(1e-6, min(tot, 6 * float(ec.skin_depth(f, m.sigma, m.mur)) + 1e-4), 60)
        J = []
        for d in dep:
            E, sg_ = ec.field_in_specimen(c, sp, f, np.array([rbar]), d)
            J.append(abs(E[0]) * sg_)
        J = np.array(J)
        out["pen"] = (dep, J / max(J[0], 1e-300))
        ctx.check()
        # pulsed EC: several wall thicknesses
        t = np.geomspace(1e-6, 5e-2 if m.mur > 1 else 5e-3, 40)
        pec = {}
        for tk in (0.5, 0.75, 1.0, 1.25):
            th_ = p["thickness"] * tk
            pec[th_] = ec.pulsed_response(c, ec.Specimen.plate(th_, m.sigma, m.mur), t)
        out["pec"] = (t, pec)
        return out

    def monitor(self, r):
        heads = ["#", "type", "x, y", "depth / δ", "peak |ΔZ| mΩ", "phase re lift-off",
                 "SNR dB", "status"]
        m_, f = r["mat"], r["f"]
        dl = float(ec.skin_depth(f, m_.sigma, m_.mur))
        names = dict(crack="surface crack", subcrack="embedded crack", void="void")
        rows, flags = [], []
        for i, (q, m) in enumerate(zip(r["flaws"], r.get("meas", []))):
            rows.append([str(i + 1), names[q.kind], f"{q.x * 1e3:.1f}, {q.y * 1e3:.1f} mm",
                         f"{q.depth / dl:.2f}", f"{m['amp'] * 1e3:.3g}", f"{m['phase']:.0f}°",
                         f"{m['snr']:.1f}", "detected" if m["detected"] else "missed"])
            flags.append(m["detected"])
        return heads, rows, flags

    def show_result(self, r):
        p = self.params.values()
        pi = self.w_z.getPlotItem(); pi.clear()
        if pi.legend is not None:
            pi.legend.clear()
        freqs, (R, X) = r["floc"]
        W.curve(pi, R, X, 0, 2, name="frequency locus")
        lo = r["lift"]
        W.curve(pi, lo[:, 0], lo[:, 1], 1, 1.6, name="lift-off", style="dash")
        sig, cl = r["cond"]
        W.curve(pi, cl[:, 0], cl[:, 1], 2, 1.6, name="conductivity 0.5–60 MS/m")
        th, tl = r["thk"]
        W.curve(pi, tl[:, 0], tl[:, 1], 4, 1.6, name="thickness 0.05–20 mm", style="dot")
        pr, px = r["point"]
        pi.addItem(pg.ScatterPlotItem(pr, px, size=11, brush=pg.mkBrush(theme.color("gate")),
                                      pen=None, name="operating point"))
        for k, (a, b) in r["refs"].items():
            pi.addItem(pg.ScatterPlotItem(a, b, size=7, symbol="d",
                                          brush=pg.mkBrush(theme.current["plot_fg"]), pen=None))
            W.text(pi, a[0], b[0], " " + k.replace("aluminium-", "Al ").replace("titanium-6al4v", "Ti64")
                   .replace("stainless-304", "304"), anchor=(0, 0.5))
        pi.addItem(pg.ScatterPlotItem([0], [1], size=8, symbol="+",
                                      brush=pg.mkBrush(theme.current["plot_fg"]), pen=None))
        # flaw signal
        pl = self.w_liss.getPlotItem(); pl.clear()
        pt = self.w_trace.getPlotItem(); pt.clear()
        if pt.legend is not None:
            pt.legend.clear()
        if pl.legend is not None:
            pl.legend.clear()
        for i, m in enumerate(r.get("meas", [])):
            z = ec.rotate_phase(m["z"], p["rot"]) * 1e3
            W.curve(pl, z.real, z.imag, i, 2, name=f"#{i + 1}")
        if r.get("meas"):
            pl.addItem(pg.ScatterPlotItem([0], [0], size=6, brush=pg.mkBrush(theme.color("fg"))))
        if "scan" in r:
            xs, dz = r["scan"]
            z = ec.rotate_phase(dz, -r["lo_phase"] + 180 + p["rot"]) * 1e3
            W.curve(pt, xs * 1e3, z.real, 0, name="horizontal")
            W.curve(pt, xs * 1e3, z.imag, 1, name="vertical")
            for i, q in enumerate(r["flaws"]):
                if abs(q.y) < 1e-9:
                    W.vline(pt, q.x * 1e3, "data", "dot", label=f"#{i + 1}")
        self.w_cmap.clear_overlays()
        if "cmap" in r:
            xs, ys, dz = r["cmap"]
            self.w_cmap.set(np.abs(dz) * 1e3, xs * 1e3, ys * 1e3)
            for i, q in enumerate(r["flaws"]):
                lab = pg.TextItem(str(i + 1), color=theme.color(5), anchor=(0.5, 0.5))
                lab.setPos(q.x * 1e3, q.y * 1e3)
                self.w_cmap.add(lab)
        pp = self.w_pen.getPlotItem(); pp.clear()
        if pp.legend is not None:
            pp.legend.clear()
        dep, J = r["pen"]
        W.curve(pp, dep * 1e3, J, 0, 2, name="model (under the winding)")
        m = r["mat"]
        dl = float(ec.skin_depth(r["f"], m.sigma, m.mur))
        W.curve(pp, dep * 1e3, np.exp(-dep / dl), 1, 1.4, name="plane wave e^(−z/δ)", style="dash")
        for k in (1, 2, 3):
            W.vline(pp, k * dl * 1e3, "data", "dot", label=f"{k}δ")
        pe = self.w_pec.getPlotItem(); pe.clear()
        if pe.legend is not None:
            pe.legend.clear()
        t, pec = r["pec"]
        for i, (th, v) in enumerate(pec.items()):
            good = np.abs(v) > 0
            W.curve(pe, t[good], np.abs(v[good]), i, name=f"wall {th * 1e3:.1f} mm")

    def summary(self, r):
        c, m, f = r["coil"], r["mat"], r["f"]
        dl = float(ec.skin_depth(f, m.sigma, m.mur))
        rows = [("conductivity", f"{m.sigma / 1e6:.3g} MS/m ({m.iacs:.1f} % IACS)"),
                ("μr", W.fmt(m.mur)),
                ("skin depth δ", W.fmt(dl * 1e3, "mm")),
                ("effective depth (3δ)", W.fmt(3 * dl * 1e3, "mm")),
                ("coil L₀", W.fmt(ec.inductance(c) * 1e6, "µH")),
                ("X₀ at f", W.fmt(float(ec.isolated_reactance(c, f)), "Ω")),
                ("f / f_c (Förster)", W.fmt(f / ec.characteristic_frequency(c, m.sigma, m.mur)))]
        pr, px = r["point"]
        rows.append(("normalised Z", f"{pr[0]:.4f} + j{px[0]:.4f}"))
        meas = r.get("meas", [])
        if meas:
            rows += [("largest defect |ΔZ|", W.fmt(max(m["amp"] for m in meas) * 1e3, "mΩ")),
                     ("defects detected (SNR ≥ 6 dB)",
                      f"{sum(m['detected'] for m in meas)} / {len(meas)}")]
        t, pec = r["pec"]
        th = list(pec)[2]
        try:
            tau = ec.pec_decay_constant(t, pec[th])
            rows.append(("PEC late decay τ (nominal wall)", W.fmt(tau * 1e3, "ms")))
        except Exception:
            pass
        return rows
