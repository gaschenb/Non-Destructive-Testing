"""Active thermography: flash / step heating, 3-D diffusion with defects, TSR, PPT, lock-in."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..core import thermography as th
from ..core.materials import get
from . import theme
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

D = th.ThermalDefect
GRID = [(-13e-3, 13e-3), (13e-3, 13e-3), (-13e-3, -13e-3), (13e-3, -13e-3)]


def _def(kind, x, y, depth, diam, gap=50, fill="air", thick=0.5):
    """Defect of an example problem, display units (mm, µm)."""
    return dict(kind=kind, x=x * 1e3, y=y * 1e3, depth=depth, wx=diam, wy=diam, gap=gap,
                fill=fill, thick=thick)


def _ladder(kind, depth, diam, factors=(0.5, 1.0, 1.5, 2.0), **kw):
    """Four equal defects at depth × factors on a 2 × 2 grid."""
    return [_def(kind, x, y, depth * f, diam, **kw) for (x, y), f in zip(GRID, factors)]


def _sizes(depth, diam, factors=(0.4, 0.7, 1.0, 1.5)):
    return [_def("delam", x, y, depth, diam * f) for (x, y), f in zip(GRID, factors)]


SCENARIOS = [
    dict(name="CFRP delaminations at four depths",
         desc="4 mm quasi-isotropic CFRP, 10 mm air-gap delaminations at 0.5–2 mm, 8 kJ/m² flash. "
              "Deeper defects appear later and weaker (contrast ∝ 1/d², t ∝ d²).",
         values=dict(material="cfrp", heating="flash", view="reflection", thickness=4,
                     energy=8, pulse=5, t_end=20, nedt=20, nonuni=0, h=10, tview=1.0),
         defects=_ladder("delam", 1.0, 10, gap=50)),
    dict(name="Aspect-ratio limit",
         desc="Delaminations of 4–15 mm at 1 mm depth: lateral diffusion erases defects with "
              "diameter/depth ≲ 2 — the classic detectability rule.",
         values=dict(material="cfrp", heating="flash", view="reflection", thickness=4,
                     energy=8, pulse=5, t_end=20, nedt=20, nonuni=0, h=10, tview=2.0),
         defects=_sizes(1.0, 10)),
    dict(name="Aluminium back-drilled holes",
         desc="3 mm Al 2024 plate with flat-bottom holes leaving 0.4–1.6 mm ligaments. High "
              "diffusivity: everything happens within 100 ms; a fast camera is needed.",
         values=dict(material="aluminium-2024", heating="flash", view="reflection", thickness=3,
                     energy=8, pulse=5, t_end=0.25, nedt=20, nonuni=0,
                     h=10, tview=0.01),
         defects=_ladder("fbh", 0.8, 10)),
    dict(name="Steel wall thinning",
         desc="5 mm steel with corrosion-like back-wall thinning (fbh) to 1–4 mm.",
         values=dict(material="steel", heating="flash", view="reflection", thickness=5,
                     energy=10, pulse=5, t_end=2.0, nedt=20, nonuni=0, h=10,
                     tview=0.15),
         defects=_ladder("fbh", 2.0, 12)),
    dict(name="Water ingress in GFRP",
         desc="GFRP with 0.5 mm water inclusions: water's high ρc makes the defects appear "
              "cold — negative contrast.",
         values=dict(material="gfrp", heating="flash", view="reflection", thickness=4,
                     energy=8, pulse=5, t_end=40, nedt=20, nonuni=0, h=10,
                     tview=4.0),
         defects=_ladder("inclusion", 0.5, 12, fill="water", thick=0.5)),
    dict(name="Step heating of thick CFRP",
         desc="Halogen-lamp step heating (5 s) on 6 mm CFRP for deeper defects.",
         values=dict(material="cfrp", heating="step", view="reflection", thickness=6,
                     energy=40, pulse=5, t_end=40, nedt=20, nonuni=0, h=10,
                     tview=8.0),
         defects=_ladder("delam", 1.0, 12, gap=80)),
    dict(name="Non-uniform flash, TSR rescues it",
         desc="±30 % flash non-uniformity hides shallow defects in raw contrast; TSR's "
              "log-derivatives are insensitive to the absorbed energy.",
         values=dict(material="cfrp", heating="flash", view="reflection", thickness=4,
                     energy=8, pulse=5, t_end=20, nedt=20, nonuni=0.3, h=10,
                     tview=1.0, proc="tsr2"),
         defects=_ladder("delam", 0.6, 10, gap=50)),
    dict(name="Transmission mode",
         desc="Camera behind the plate: all depths have similar contrast, depth information "
              "comes only from timing.",
         values=dict(material="cfrp", heating="flash", view="transmission", thickness=4,
                     energy=8, pulse=5, t_end=40, nedt=20, nonuni=0, h=10,
                     tview=12.0),
         defects=_ladder("delam", 1.0, 10, gap=50)),
]


class ThermographyPage(Workspace):
    title = "Thermography"
    SCENARIOS = SCENARIOS
    HEAVY_KEYS = ("sim",)
    VIEW_KEYS = ("tview", "proc")
    auto_heavy = True
    CHOICES = [
        C("material", "Material", [("CFRP (quasi-isotropic)", "cfrp"), ("GFRP", "gfrp"),
                                   ("Aluminium 2024", "aluminium-2024"), ("Steel", "steel"),
                                   ("Stainless 304", "stainless-304"), ("Ti-6Al-4V", "titanium-6al4v"),
                                   ("PMMA", "pmma")], "cfrp"),
        C("heating", "Excitation", [("Flash", "flash"), ("Step (lamps)", "step")], "flash"),
        C("view", "Camera", [("Reflection (front)", "reflection"), ("Transmission (rear)", "transmission")],
          "reflection"),
        C("proc", "Image processing", [("Raw ΔT", "raw"), ("Absolute contrast", "contrast"),
                                       ("TSR 1st derivative", "tsr1"), ("TSR 2nd derivative", "tsr2"),
                                       ("PPT phase", "ppt")], "contrast"),
    ]
    PARAMS = [
        P("thickness", "Plate thickness (mm)", 1, 12, 4, 1e-3, decimals=2),
        P("energy", "Absorbed energy (kJ/m²)", 0.5, 80, 8, 1e3, log=True, decimals=1),
        P("pulse", "Step duration (s)", 0.1, 30, 5, 1, log=True, decimals=2),
        P("t_end", "Acquisition (s)", 0.02, 200, 20, 1, log=True, decimals=3),
        P("nedt", "Camera NETD (mK)", 0, 200, 20, 1e-3, decimals=0),
        P("nonuni", "Heating non-uniformity", 0, 0.6, 0, 1, decimals=2),
        P("h", "Convection h (W/m²K)", 0, 100, 10, 1, decimals=1),
        P("tview", "Display time (s)", 0.0005, 200, 1, 1, log=True, decimals=4),
    ]

    def build_views(self):
        self.w_img = W.ImagePanel("Surface", "x (mm)", "y (mm)", cmap="CET-L3", invert_y=False)
        self.w_curves = W.plot("Contrast at the defect centres", "t (s)", "ΔT_def − ΔT_sound (K)",
                               legend=True)
        self.w_curves.getPlotItem().setLogMode(x=True, y=False)
        self.w_tsr = W.plot("TSR: d²ln T / d(ln t)²", "t (s)", "2nd log-derivative", legend=True)
        self.w_tsr.getPlotItem().setLogMode(x=True, y=False)
        self.w_1d = W.plot("1-D model (thermal quadrupoles)", "t (s)", "ΔT (K)", legend=True)
        self.w_1d.getPlotItem().setLogMode(x=True, y=True)
        self.w_lock = W.plot("Lock-in phase", "f (Hz)", "phase (°)", legend=True)
        self.w_lock.getPlotItem().setLogMode(x=True, y=False)
        return [("Image", self.w_img), ("Contrast", self.w_curves), ("TSR", self.w_tsr),
                ("1-D model", self.w_1d), ("Lock-in", self.w_lock)]

    DEFECT_NOTE = ("Depth is measured from the heated face; for a back-drilled hole it is the "
                   "remaining ligament. The plate is 60 × 60 mm, centred on 0.")
    DEFECTS = [
        C("kind", "Type", [("Delamination (air gap)", "delam"), ("Back-drilled hole", "fbh"),
                           ("Inclusion", "inclusion")], "delam"),
        P("x", "x (mm)", -28, 28, 0, 1e-3, decimals=1),
        P("y", "y (mm)", -28, 28, 0, 1e-3, decimals=1),
        P("depth", "Depth / ligament (mm)", 0.1, 10, 1.0, 1e-3, decimals=2),
        P("wx", "Width x (mm)", 1, 40, 10, 1e-3, decimals=1),
        P("wy", "Width y (mm)", 1, 40, 10, 1e-3, decimals=1),
        P("gap", "Delamination gap (µm)", 5, 300, 50, 1e-6, log=True, decimals=0),
        C("fill", "Inclusion fill", [("Water", "water"), ("Air", "air"), ("Epoxy", "epoxy"),
                                     ("Aluminium", "aluminium-2024")], "water"),
        P("thick", "Inclusion thickness (mm)", 0.05, 3, 0.5, 1e-3, decimals=2),
    ]

    def describe_defect(self, d):
        size = (f"Ø{d['wx']:g}" if abs(d["wx"] - d["wy"]) < 1e-9 else f"{d['wx']:g} × {d['wy']:g}") + " mm"
        if d["kind"] == "fbh":
            what = f"hole {size}, ligament {d['depth']:g} mm"
        elif d["kind"] == "inclusion":
            what = f"{d['fill'].split('-')[0]} inclusion {size} at {d['depth']:g} mm"
        else:
            what = f"delamination {size} at {d['depth']:g} mm"
        return f"{what}, ({d['x']:g}, {d['y']:g})"

    def defect_hidden(self, d):
        return {"delam": ("fill", "thick"), "fbh": ("gap", "fill", "thick"),
                "inclusion": ("gap",)}[d["kind"]]

    VIEW_HELP = {
        "Image": "Surface temperature (or the selected processed quantity) from the 3-D "
                 "simulation at the display time. Hover for the value at any point; labels "
                 "mark the defects.",
        "Contrast": "Temperature difference between each defect centre and sound material "
                    "versus time (solid: 3-D, dashed: 1-D laterally infinite defect). Deeper "
                    "defects peak later and lower.",
        "TSR": "Second logarithmic derivative of the surface temperature (thermographic "
               "signal reconstruction). The peak time of the defect curve gives its depth.",
        "1-D model": "Exact 1-D thermal-quadrupole response of the sound plate and of each "
                     "defect taken as laterally infinite.",
        "Lock-in": "Phase difference to sound material under periodic heating versus frequency: "
                   "lower frequencies probe deeper (thermal diffusion length μ).",
    }

    def equations(self):
        return [
            W.eq("Heat diffusion",
                 r"$\rho c\,\dfrac{\partial T}{\partial t}=\nabla\cdot(k\nabla T),\qquad"
                 r"-k\,\dfrac{\partial T}{\partial n}=hT+q(t)$",
                 "3-D: LOD backward Euler (unconditionally stable); a delamination is a series "
                 "contact resistance R = g / k_air."),
            W.eq("1-D thermal quadrupoles",
                 r"$\bar{T}(s)=\dfrac{Z(s)\,\bar{q}(s)}{1+hZ(s)}$",
                 r"Semi-infinite:  $\Delta T=\dfrac{Q}{e\sqrt{\pi t}},\qquad e=\sqrt{k\rho c}$",
                 r"Adiabatic slab:  $\Delta T=\dfrac{Q}{\rho cL}\left[1+2\sum_{n\geq1}"
                 r"e^{-n^2\pi^2\alpha t/L^2}\right]$"),
            W.eq("Thermographic signal reconstruction",
                 r"$\ln T=\sum_n a_n(\ln t)^n,\qquad d=\sqrt{\alpha t^*/\kappa},\quad\kappa=\dfrac{1}{\pi}$",
                 "t* is the peak time of the second log-derivative contrast."),
            W.eq("Pulsed phase and lock-in",
                 r"$\varphi(f)=\arg\,\mathcal{F}[T(t)],\qquad d\approx1.82\sqrt{\dfrac{\alpha}{\pi f_b}}$",
                 r"$\mu=\sqrt{\dfrac{\alpha}{\pi f}},\qquadd_{max}\approx1.5\text{–}2\,\mu$"),
            W.eq("Defect monitor",
                 "Peak |ΔT| contrast at the defect centre and its time; SNR against the camera "
                 "NETD (1 mK floor). Detected when SNR ≥ 6 dB."),
        ]

    def hidden_keys(self, p):
        return set() if p["heating"] == "step" else {"pulse"}

    # ------------------------------------------------------------------ helpers
    def thermal_defects(self, p):
        return [D(d["kind"], d["x"], d["y"], depth=min(d["depth"], 0.98 * p["thickness"]),
                  wx=d["wx"], wy=d["wy"], gap=d["gap"],
                  fill="air" if d["kind"] == "fbh" else d["fill"], thick=d["thick"])
                for d in p.get("_defects", [])]

    def layers_for(self, p, d):
        """1-D stacks (front first) for sound material and a laterally infinite defect."""
        m = p["material"]
        L = p["thickness"]
        sound = [th.ThermalLayer.of(m, L)]
        if d.kind == "delam":
            a, b = min(d.depth, 0.98 * L), max(L - d.depth, 0.02 * L)
            return sound, ([th.ThermalLayer.of(m, a), th.ThermalLayer.of(m, b)],
                           [d.gap / get("air").k])
        if d.kind == "fbh":
            return sound, ([th.ThermalLayer.of(m, min(d.depth, L))], None)
        t1 = min(d.thick, max(L - d.depth, 1e-5))
        rest = max(L - d.depth - t1, 1e-5)
        fill = get(d.fill)
        return sound, ([th.ThermalLayer.of(m, d.depth),
                        th.ThermalLayer(t1, fill.k, fill.rho * fill.cp),
                        th.ThermalLayer.of(m, rest)], None)

    def compute(self, p, heavy, ctx):
        dfs = self.thermal_defects(p)
        mat = get(p["material"])
        kz = mat.k_z if mat.k_z is not None else mat.k
        alpha = kz / (mat.rho * mat.cp)
        t = np.geomspace(p["t_end"] / 2000, p["t_end"], 160)
        pulse = p["pulse"] if p["heating"] == "step" else None
        out = dict(t1=t, alpha=alpha, defs=dfs, curves=[], lock=[], heavy=heavy)
        f = np.geomspace(0.05 * alpha / p["thickness"] ** 2, 50 * alpha / max(min([d.depth for d in dfs] or [p["thickness"]]), 1e-4) ** 2, 120)
        sound = [th.ThermalLayer.of(p["material"], p["thickness"])]
        out["sound"] = th.flash_response_1d(sound, t, p["energy"], h_front=p["h"],
                                            h_rear=p["h"], pulse=pulse)
        out["lock_sound"] = th.lockin_1d(sound, f, h_front=p["h"], h_rear=p["h"])
        for d in dfs:
            _, (lay, res) = self.layers_for(p, d)
            out["curves"].append(th.flash_response_1d(lay, t, p["energy"], res, h_front=p["h"],
                                                       h_rear=p["h"], pulse=pulse))
            out["lock"].append(th.lockin_1d(lay, f, resistances=res, h_front=p["h"], h_rear=p["h"]))
        out["f"] = f
        # TSR depth estimate from the 1-D defect curves (flash only)
        if pulse is None:
            lt = np.log(t)

            def d2(c):
                return np.gradient(np.gradient(np.log(np.maximum(c, 1e-12)), lt), lt)
            ds = d2(out["sound"])
            tw = t < th.tsr_window(alpha, p["thickness"])
            out["tsr_1d"] = [float(th.tsr_depth(alpha, t[tw][np.argmax((d2(c) - ds)[tw])]))
                             if tw.sum() > 3 else float("nan") for c in out["curves"]]
        if not heavy:
            return out
        ctx.check()
        s = th.ThermalSetup(material=p["material"], thickness=p["thickness"], heating=p["heating"],
                            energy=p["energy"], pulse=p["pulse"], h_front=p["h"], h_rear=p["h"],
                            t_end=p["t_end"], n_frames=200, nonuniform=p["nonuni"],
                            nedt=p["nedt"], view=p["view"], nz=24 if p["thickness"] < 6e-3 else 32)
        sim = th.ThermalSolver(s, dfs).run(progress=lambda i, n: (ctx.check(), ctx.progress(i, n)))
        T, fr = sim["t"], sim["frames"]
        X, Y = np.meshgrid(sim["x"], sim["y"])
        sound_mask = np.ones(X.shape, bool)
        for d in dfs:
            sound_mask &= np.hypot((X - d.x) / (0.6 * d.wx + 3e-3), (Y - d.y) / (0.6 * d.wy + 3e-3)) > 1
        ref = th.sound_reference(fr, sound_mask)
        con = th.thermal_contrast(fr, ref)
        ctx.check()
        fpos = np.clip(fr, 1e-4, None)
        _, ev = th.tsr_fit(T, fpos, degree=7) if p["heating"] == "flash" else (None, None)
        fp, phase, amp = th.ppt(T, fr)
        idx = []
        for d in dfs:
            ix = int(np.argmin(np.abs(sim["x"] - d.x))); iy = int(np.argmin(np.abs(sim["y"] - d.y)))
            idx.append((iy, ix))
        sim.update(contrast=con, ref=ref, tsr=ev, ppt=(fp, phase), idx=idx, sound_mask=sound_mask)
        if ev is not None:
            d2 = ev(T, 2)
            sim["tsr2"] = d2
            sim["tsr1"] = ev(T, 1)
            d2s = d2[:, sound_mask].mean(axis=1)
            ok = (T > 2 * T[0]) & (T < th.tsr_window(alpha, p["thickness"]))
            ok = ok if ok.sum() > 3 else T > 0
            sim["tsr_depth"] = [float(th.tsr_depth(alpha, T[ok][np.argmax((d2[:, iy, ix] - d2s)[ok])]))
                                for iy, ix in idx]
            sim["tsr2_sound"] = d2s
        # PPT: frequency with the strongest phase contrast
        nb = max(3, min(len(fp) // 4, 40))
        sd = [np.std(phase[k][~sound_mask]) - np.std(phase[k][sound_mask]) for k in range(1, nb)]
        sim["ppt_k"] = 1 + int(np.argmax(sd))
        return dict(out, sim=sim)

    # ------------------------------------------------------------------ display
    def show_result(self, r):
        p = self.params.values()
        dfs = r["defs"]
        names = [self._label(d) for d in dfs]
        # 1-D curves
        p1 = self.w_1d.getPlotItem(); p1.clear()
        if p1.legend is not None:
            p1.legend.clear()
        W.curve(p1, r["t1"], np.maximum(r["sound"], 1e-6), 0, 2.2, name="sound")
        allc = np.concatenate([r["sound"]] + list(r["curves"]))
        W.log_ticks(p1, "left", max(allc.min(), 1e-6), allc.max())
        W.log_ticks(p1, "bottom", r["t1"][0], r["t1"][-1])
        for k, c in enumerate(r["curves"]):
            W.curve(p1, r["t1"], np.maximum(c, 1e-6), k + 1, 1.5, name=names[k])
        pl = self.w_lock.getPlotItem(); pl.clear()
        if pl.legend is not None:
            pl.legend.clear()
        ps = np.degrees(np.angle(r["lock_sound"]))
        for k, c in enumerate(r["lock"]):
            W.curve(pl, r["f"], np.degrees(np.angle(c)) - ps, k + 1, 1.6, name=names[k] + " − sound")
        W.hline(pl, 0, "fg", "dot")
        W.log_ticks(pl, "bottom", r["f"][0], r["f"][-1])
        sim = r.get("sim")
        pc = self.w_curves.getPlotItem(); pc.clear()
        if pc.legend is not None:
            pc.legend.clear()
        pt = self.w_tsr.getPlotItem(); pt.clear()
        if pt.legend is not None:
            pt.legend.clear()
        self.w_img.clear_overlays()
        if sim is None:
            for k, c in enumerate(r["curves"]):
                W.curve(pc, r["t1"], c - r["sound"], k + 1, 1.4, name=names[k] + " (1-D)", style="dash")
            return
        T = sim["t"]
        for k, (iy, ix) in enumerate(sim["idx"]):
            W.curve(pc, T, sim["contrast"][:, iy, ix], k + 1, 1.6, name=names[k])
            W.curve(pc, r["t1"], r["curves"][k] - r["sound"], k + 1, 1.0, style="dash")
        if "tsr2" in sim:
            W.curve(pt, T, sim["tsr2_sound"], 0, 2.2, name="sound")
            for k, (iy, ix) in enumerate(sim["idx"]):
                W.curve(pt, T, sim["tsr2"][:, iy, ix], k + 1, 1.6, name=names[k])
        W.log_ticks(pc, "bottom", T[0], T[-1])
        W.log_ticks(pt, "bottom", T[0], T[-1])
        tv = float(np.clip(p["tview"], T[0], T[-1]))
        it = int(np.argmin(np.abs(np.log(T) - np.log(tv))))
        x, y = sim["x"] * 1e3, sim["y"] * 1e3
        proc = p["proc"]
        cmap, bip = "CET-L3", False
        if proc == "raw":
            img, ttl = sim["frames"][it], f"Surface ΔT (K) at t = {T[it]:.4g} s"
        elif proc == "contrast":
            img, ttl, bip = sim["contrast"][it], f"Absolute contrast (K) at t = {T[it]:.4g} s", True
        elif proc in ("tsr1", "tsr2") and proc in sim:
            img = sim[proc][it]
            img = img - np.median(img[sim["sound_mask"]])
            ttl, bip = f"TSR {'1st' if proc == 'tsr1' else '2nd'} derivative − sound at t = {T[it]:.4g} s", True
        elif proc == "ppt":
            fp, ph = sim["ppt"]
            k = sim["ppt_k"]
            img = np.degrees(ph[k] - np.median(ph[k][sim["sound_mask"]]))
            ttl, bip = f"PPT phase − sound (°) at f = {fp[k]:.4g} Hz", True
        else:
            img, ttl = sim["frames"][it], f"Surface ΔT (K) at t = {T[it]:.4g} s (TSR needs flash)"
        if bip:
            m = float(np.percentile(np.abs(img), 99.5)) or 1.0
            self.w_img.set_cmap("CET-D1")
            self.w_img.set(img, x, y, levels=(-m, m), title=ttl)
        else:
            self.w_img.set_cmap(cmap)
            self.w_img.set(img, x, y, levels=(float(img.min()), float(img.max())), title=ttl)
        for k, d in enumerate(dfs):
            t = pg.TextItem(f"{k + 1}: {names[k]}", color=theme.color("fg"), anchor=(0.5, 0))
            t.setPos(d.x * 1e3, d.y * 1e3 - d.wy * 500 - 1.0)
            self.w_img.add(t)
        W.vline(pc, np.log10(T[it]), "gate", "dot", label="display time")

    def monitor(self, r):
        heads = ["#", "type", "depth", "peak |ΔT| K", "at t (s)", "TSR depth", "SNR dB", "status"]
        p = self.params.values()
        floor = max(p["nedt"], 1e-3)
        sim = r.get("sim")
        rows, flags = [], []
        kinds = dict(delam="delamination", fbh="back-drilled hole", inclusion="inclusion")
        for k, d in enumerate(r["defs"]):
            if sim is not None:
                iy, ix = sim["idx"][k]
                tp, cp = th.peak_contrast(sim["t"], np.abs(sim["contrast"][:, iy, ix]))
                src = ""
            else:
                tp, cp = th.peak_contrast(r["t1"], np.abs(r["curves"][k] - r["sound"]))
                src = " (1-D)"
            dep = (sim or {}).get("tsr_depth", r.get("tsr_1d"))
            ok = dep is not None and d.kind == "delam" and p["view"] == "reflection"
            tsr = f"{dep[k] * 1e3:.2f} mm" if ok and np.isfinite(dep[k]) else "—"
            snr = 20 * np.log10(max(cp, 1e-30) / floor)
            rows.append([str(k + 1), kinds[d.kind], f"{d.depth * 1e3:.2f} mm", f"{cp:.3g}{src}",
                         f"{tp:.3g}", tsr, f"{snr:.1f}", "detected" if snr >= 6 else "missed"])
            flags.append(snr >= 6)
        return heads, rows, flags

    def _label(self, d):
        if d.kind == "fbh":
            return f"ligament {d.depth * 1e3:.2g} mm"
        if d.kind == "inclusion":
            return f"{d.fill} @ {d.depth * 1e3:.2g} mm"
        return f"Ø{d.wx * 1e3:.3g} @ {d.depth * 1e3:.2g} mm"

    def summary(self, r):
        p = self.params.values()
        m = get(p["material"])
        e = np.sqrt((m.k_z or m.k) * m.rho * m.cp)
        L = p["thickness"]
        rows = [("diffusivity α_z", W.fmt(r["alpha"] * 1e6, "mm²/s")),
                ("effusivity e", W.fmt(e, "W s½/m²K")),
                ("rear-face half-rise t½ = 1.37L²/(π²α)", W.fmt(1.37 * L ** 2 / (np.pi ** 2 * r["alpha"]), "s")),
                ("adiabatic plateau Q/(ρcL)", W.fmt(p["energy"] / (m.rho * m.cp * L), "K"))]
        sim = r.get("sim")
        for k, d in enumerate(r["defs"]):
            lab = self._label(d)
            parts = []
            if "tsr_1d" in r:
                parts.append(f"1-D {r['tsr_1d'][k] * 1e3:.2f}")
            if sim is not None and "tsr_depth" in sim:
                parts.append(f"3-D {sim['tsr_depth'][k] * 1e3:.2f}")
            if sim is not None:
                iy, ix = sim["idx"][k]
                tp, cp = th.peak_contrast(sim["t"], np.abs(sim["contrast"][:, iy, ix]))
                rows.append((f"{lab}: peak |ΔT|", f"{cp:.3g} K at {tp:.3g} s"))
            if parts and d.kind == "delam":
                rows.append((f"{lab}: TSR depth (mm)", " / ".join(parts)))
        if sim is not None:
            fp, _ = sim["ppt"]
            rows.append(("PPT best frequency", W.fmt(fp[sim["ppt_k"]], "Hz")))
        return rows
