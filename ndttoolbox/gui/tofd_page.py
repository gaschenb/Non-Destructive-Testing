"""Time-of-flight diffraction: D-scans / B-scans of crack tips, depth sizing."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..core import signals as sg
from ..core import ultrasonics as ut
from . import theme
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace


def _cr(top, bottom, length, x=0.0, y=0.0):
    """Crack in display units (mm): tip depths, length along the weld, centre (x, y)."""
    return dict(top=top, height=bottom - top, length=length, x=x, y=y * 1e3 if abs(y) < 1 else y)


SCENARIOS = [
    dict(name="Embedded fatigue crack",
         desc="25 mm butt weld, 60° 5 MHz probes at 60 mm PCS. Two tip echoes of opposite "
              "phase between the lateral wave and the back wall; height = d₂ − d₁.",
         values=dict(thickness=25, pcs=60, angle=60, f0=5),
         defects=[_cr(8, 12, 20)]),
    dict(name="Surface-breaking toe crack",
         desc="Crack open to the scanning surface: no upper tip — the lateral wave is "
              "interrupted over the crack length instead.",
         values=dict(thickness=25, pcs=60, angle=60, f0=5),
         defects=[_cr(0, 6, 25)]),
    dict(name="Root crack",
         desc="Crack growing from the back wall: the back-wall echo is interrupted and only the "
              "upper tip diffracts.",
         values=dict(thickness=20, pcs=50, angle=60, f0=5),
         defects=[_cr(13, 20, 18)]),
    dict(name="Two cracks, thick section",
         desc="50 mm section with two embedded flaws at different depths; 45° 3.5 MHz probes "
              "cover the full thickness.",
         values=dict(thickness=50, pcs=100, angle=45, f0=3.5),
         defects=[_cr(12, 17, 15, y=-12e-3), _cr(32, 40, 20, y=14e-3)]),
    dict(name="Near-surface dead zone",
         desc="Shallow crack at 2–4 mm: its tip signals merge with the lateral wave. Reduce the "
              "PCS or raise the frequency to resolve them.",
         values=dict(thickness=25, pcs=70, angle=60, f0=5),
         defects=[_cr(2, 4, 20)]),
]


class TOFDPage(Workspace):
    title = "TOFD"
    SCENARIOS = SCENARIOS
    CHOICES = [C("material", "Material", [("Carbon steel", "steel"),
                                          ("Stainless 304", "stainless-304"),
                                          ("Aluminium 6061", "aluminium-6061")], "steel"),
               C("scan", "Scan", [("D-scan (along weld)", "d"), ("B-scan (across weld)", "b")], "d")]
    PARAMS = [
        P("thickness", "Thickness (mm)", 5, 120, 25, 1e-3, decimals=1),
        P("pcs", "Probe centre separation (mm)", 10, 250, 60, 1e-3, decimals=1),
        P("angle", "Probe angle (°)", 35, 75, 60, 1, decimals=1),
        P("f0", "Frequency (MHz)", 1, 15, 5, 1e6, decimals=2),
        P("ypos", "A-scan position (mm)", -60, 60, 0, 1e-3, decimals=1),
        P("noise", "Noise (dB)", -60, -10, -40, 1, decimals=0),
        P("gain", "Display gain (dB)", -20, 30, 6, 1, decimals=1),
    ]

    def build_views(self):
        self.w_a = W.plot("A-scan", "time (µs)", "amplitude")
        self.w_d = W.ImagePanel("D-scan", "scan position (mm)", "depth (mm)", bipolar=True)
        self.w_d.pi.setAspectLocked(False)
        self.w_d.set_cmap("CET-L1")
        self.w_cal = W.plot("Depth vs arrival time", "time after lateral wave (µs)", "depth (mm)",
                            legend=True)
        return [("A-scan", self.w_a), ("D-scan", self.w_d), ("Depth calibration", self.w_cal)]

    DEFECT_NOTE = "Planar cracks perpendicular to the surface (weld along y)."
    DEFECTS = [
        P("top", "Top (upper tip) depth (mm)", 0, 120, 8, 1e-3, decimals=2,
          tip="0 = surface-breaking"),
        P("height", "Through-wall height (mm)", 0.2, 60, 4, 1e-3, decimals=2,
          tip="reaching the back wall = root crack"),
        P("length", "Length along the weld (mm)", 1, 100, 20, 1e-3, decimals=1),
        P("y", "Centre along the weld y (mm)", -60, 60, 0, 1e-3, decimals=1),
        P("x", "Offset across the weld x (mm)", -30, 30, 0, 1e-3, decimals=1),
    ]

    def describe_defect(self, d):
        T = self.params.values()["thickness"] * 1e3
        kind = "surface" if d["top"] < 0.01 else "root" if d["top"] + d["height"] >= T - 0.01 else "embedded"
        return (f"{kind} crack {d['top']:g}–{d['top'] + d['height']:g} mm deep, "
                f"{d['length']:g} mm long at y {d['y']:g}")

    VIEW_HELP = {
        "A-scan": "RF signal at the selected scan position. In order: lateral wave (surface), "
                  "upper-tip diffraction, lower-tip diffraction (opposite phase), back wall, then "
                  "mode-converted shear arrivals. Dotted purple lines mark the picked tips used "
                  "for sizing.",
        "D-scan": "Grey-scale RF stack along the weld (D-scan) or across it (B-scan), with time "
                  "converted to depth. Cracks appear as arcs/lines between the lateral wave (top) "
                  "and the back wall (dashed red). Black/white = phase. The dashed line is the "
                  "A-scan position.",
        "Depth calibration": "Depth versus arrival time after the lateral wave (solid; strongly "
                             "non-linear near the surface) and the depth resolution Δd (dashed), "
                             "which explains the near-surface dead zone.",
    }

    def equations(self):
        return [
            W.eq("Arrival times (probes at ±S, tip at depth d)",
                 r"$t_{LW}=\dfrac{2S}{c},\qquad t(d)=\dfrac{2\sqrt{S^2+d^2}}{c}$",
                 r"$t_{BW}=\dfrac{2\sqrt{S^2+T^2}}{c}$"),
            W.eq("Depth from time",
                 r"$d=\sqrt{\left(\dfrac{c\,t}{2}\right)^2-S^2},\qquad h=d_2-d_1$"),
            W.eq("Off-centre (B-scan) hyperbola",
                 r"$t(x)=\dfrac{\sqrt{(S-x)^2+d^2}+\sqrt{(S+x)^2+d^2}}{c}$"),
            W.eq("Depth resolution (pulse length tₚ)",
                 r"$\Delta d\approx\dfrac{c\,t_p\,\sqrt{S^2+d^2}}{2d}$"),
            W.eq("Phase",
                 "lateral wave (+), upper tip (−), lower tip (+), back wall (−). "
                 "Length: 6 dB drop of the tip signal along the scan."),
        ]

    def setup(self, p):
        return ut.TOFDSetup(material=p["material"], thickness=p["thickness"], pcs=p["pcs"],
                            angle=p["angle"], f0=p["f0"], noise=10 ** (p["noise"] / 20))

    def cracks(self, p):
        out = []
        T = p["thickness"]
        for d in p.get("_defects", []):
            top = min(d["top"], T * 0.99)
            bot = min(top + d["height"], T)
            if top < 1e-4:
                top = 0.0
            if bot > T - 1e-4:
                bot = T
            out.append(ut.Crack(x=d["x"], y0=d["y"] - d["length"] / 2, y1=d["y"] + d["length"] / 2,
                                top=top, bottom=bot))
        return out

    def compute(self, p, heavy, ctx):
        s = self.setup(p)
        cr = self.cracks(p)
        if p["scan"] == "d":
            pos = np.linspace(-60e-3, 60e-3, 161)
            r = ut.tofd_scan(s, cr, ys=pos)
        else:
            pos = np.linspace(-40e-3, 40e-3, 161)
            r = ut.tofd_scan(s, cr, xs=pos)
        ia = int(np.argmin(np.abs(pos - p["ypos"])))
        r.update(setup=s, cracks=cr, scanpos=pos, ia=ia)
        r["sizing"] = ut.tofd_size_ascan(r["t"], r["data"][ia], s, r["t_delay"])
        # length: envelope of the strongest tip signal along the scan
        env = sg.envelope(r["data"])
        t = r["t"]
        sel = (t > r["t_lat"] + 1.5 / s.f0) & (t < r["t_bw"] - 1.5 / s.f0)
        prof = env[:, sel].max(axis=1) if sel.any() else np.zeros(len(pos))
        r["profile"] = prof
        # per-crack monitoring: gate the indication of each crack (its depth range ± 3 mm),
        # pick the tip diffraction(s) in the trace over its centre, size along the scan
        meas = []
        tt = t - r["t_delay"]
        quiet = (tt > s.t_lateral() + 2 / s.f0) & (tt < s.t_backwall() - 2 / s.f0)
        for c in cr:
            dlo, dhi = max(c.top - 3e-3, 0.0), min(c.bottom + 3e-3, s.thickness)
            gate = (tt > max(ut.tofd_time(dlo, s), s.t_lateral() + 1.0 / s.f0)) & \
                   (tt < min(ut.tofd_time(dhi, s), s.t_backwall() - 1.0 / s.f0))
            for o in cr:
                if o is not c:
                    quiet &= ~((tt > ut.tofd_time(max(o.top - 3e-3, 0), s)) &
                               (tt < ut.tofd_time(min(o.bottom + 3e-3, s.thickness), s)))
            quiet_c = quiet & ~gate
            yc = 0.5 * (c.y0 + c.y1) if p["scan"] == "d" else c.x
            k = int(np.argmin(np.abs(pos - yc)))
            e = env[k]
            noise = float(np.sqrt(np.mean(e[quiet_c] ** 2))) if quiet_c.sum() > 10 else 1e-12
            tips = []
            if gate.any():
                idx = np.nonzero(gate)[0]
                eg = e[idx].copy()
                sep = int(1.5 / s.f0 / (t[1] - t[0]))
                n_tips = 1 if (c.top <= 1e-4 or c.bottom >= s.thickness - 1e-4) else 2
                for _ in range(n_tips):
                    j = int(np.argmax(eg))
                    if eg[j] <= 3 * noise:
                        break
                    tips.append(float(ut.tofd_depth(tt[idx[j]], s)))
                    eg[max(0, j - sep): j + sep] = 0
            tips.sort()
            if c.top <= 1e-4 and tips:
                d1, d2 = 0.0, tips[-1]
            elif c.bottom >= s.thickness - 1e-4 and tips:
                d1, d2 = tips[0], s.thickness
            elif len(tips) == 2:
                d1, d2 = tips
            else:
                d1 = d2 = np.nan
            L = np.nan
            if p["scan"] == "d" and gate.any():
                pr = env[:, gate].max(axis=1)
                w = np.abs(pos - yc) <= (c.y1 - c.y0) / 2 + 25e-3
                try:
                    L, _, _ = ut.six_db_drop(pos[w], pr[w])
                except Exception:
                    pass
            meas.append(dict(d1=d1, d2=d2, found=bool(tips),
                             snr=20 * np.log10(max(e[gate].max() if gate.any() else 0, 1e-15) / noise),
                             length=L))
        r["meas"] = meas
        return r

    def monitor(self, r):
        heads = ["#", "true tips", "measured tips", "height true / meas.", "length true / meas.",
                 "SNR dB", "status"]
        rows, flags = [], []
        for i, (c, m) in enumerate(zip(r["cracks"], r["meas"])):
            true_h = (c.bottom - c.top) * 1e3
            if np.isfinite(m["d1"]):
                h = (m["d2"] - m["d1"]) * 1e3
                tips = f"{m['d1'] * 1e3:.1f} – {m['d2'] * 1e3:.1f} mm"
                hh = f"{true_h:.1f} / {h:.1f} mm"
                ok = abs(h - true_h) <= max(1.0, 0.25 * true_h)
                status = "sized" if ok else "detected"
            else:
                tips, hh, ok = "—", f"{true_h:.1f} / —", False
                status = "detected (one tip)" if m["found"] else "missed"
            Lm = m["length"]
            ll = f"{(c.y1 - c.y0) * 1e3:.0f} / " + (f"{Lm * 1e3:.0f} mm" if np.isfinite(Lm) else "—")
            rows.append([str(i + 1), f"{c.top * 1e3:.1f} – {c.bottom * 1e3:.1f} mm", tips, hh, ll,
                         f"{m['snr']:.1f}", status])
            flags.append(ok if m["found"] else False)
        return heads, rows, flags

    def show_result(self, r):
        s = r["setup"]
        p = self.params.values()
        g = 10 ** (p["gain"] / 20)
        t = r["t"] * 1e6
        pi = self.w_a.getPlotItem(); pi.clear()
        W.curve(pi, t, g * r["data"][r["ia"]], "signal", 1.2)
        W.vline(pi, r["t_lat"] * 1e6, "ref", label="lateral")
        W.vline(pi, r["t_bw"] * 1e6, "gate", label="back wall")
        if r["sizing"]:
            d1, d2, h = r["sizing"]
            for d, lab, sd in ((d1, "upper tip", "left"), (d2, "lower tip", "right")):
                if d is not None and np.isfinite(d):
                    W.vline(pi, (ut.tofd_time(d, s) + r["t_delay"]) * 1e6, "dac", "dot",
                            label=lab, pos=0.08, side=sd)
        pi.setXRange(r["t_lat"] * 1e6 - 1.5, r["t_bw"] * 1e6 + 2.5)
        # D-scan in depth (nonlinear time → depth mapping for display)
        tt = r["t"] - r["t_delay"]
        dep = ut.tofd_depth(tt, s)
        valid = tt > s.t_lateral()
        nd = 260
        dgrid = np.linspace(0, s.thickness * 1.15, nd)
        tg = ut.tofd_time(dgrid, s)
        img = np.array([np.interp(tg, tt, row) for row in r["data"]])
        self.w_d.set(np.clip(g * img.T, -1, 1), r["scanpos"] * 1e3, dgrid * 1e3, levels=(-1, 1),
                     title="D-scan (RF, mapped to depth)" if p["scan"] == "d" else
                     "B-scan (RF, mapped to depth)")
        self.w_d.clear_overlays()
        self.w_d.add(pg.InfiniteLine(pos=s.thickness * 1e3, angle=0, pen=theme.pen("gate", 1, "dash")))
        self.w_d.add(pg.InfiniteLine(pos=r["scanpos"][r["ia"]] * 1e3, angle=90,
                                     pen=theme.pen("ref", 1, "dash")))
        # calibration
        pc = self.w_cal.getPlotItem(); pc.clear()
        if pc.legend is not None:
            pc.legend.clear()
        d = np.linspace(0, s.thickness, 300)
        W.curve(pc, (ut.tofd_time(d, s) - s.t_lateral()) * 1e6, d * 1e3, 0, name="tip depth")
        tp = 2.0 / s.f0
        res = s.mat.cL * tp * np.sqrt((s.pcs / 2) ** 2 + d ** 2) / (2 * np.maximum(d, 1e-4))
        W.curve(pc, (ut.tofd_time(d, s) - s.t_lateral()) * 1e6, np.clip(res * 1e3, 0, s.thickness * 1e3),
                1, style="dash", name="depth resolution Δd")
        pc.setTitle("Depth and depth resolution vs arrival time")

    def summary(self, r):
        s = r["setup"]
        rows = [("lateral wave", W.fmt(r["t_lat"] * 1e6, "µs")),
                ("back wall", W.fmt(r["t_bw"] * 1e6, "µs")),
                ("beam crossing depth", W.fmt(s.pcs / 2 / np.tan(np.radians(s.angle)) * 1e3, "mm")),
                ("dead zone (≈ 2 cycles)", W.fmt(float(ut.tofd_depth(s.t_lateral() + 2 / s.f0, s)) * 1e3, "mm"))]
        if r["sizing"]:
            d1, d2, h = r["sizing"]
            rows += [("A-scan: upper tip", W.fmt(d1 * 1e3, "mm")), ("A-scan: lower tip", W.fmt(d2 * 1e3, "mm")),
                     ("A-scan: height", W.fmt(h * 1e3, "mm"))]
        return rows
