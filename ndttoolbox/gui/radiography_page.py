"""Digital radiography: X-ray / gamma sources, weld and casting radiographs, IQIs, spectra."""
from __future__ import annotations

import dataclasses

import numpy as np
import pyqtgraph as pg

from ..core import radiography as rt
from ..core.materials import get
from . import theme
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

F = rt.RTFlaw


def _weld_flaws():
    return [F("pore", x=-0.004, y=-0.008, size=1.0e-3), F("pore", x=-0.0015, y=-0.0095, size=0.6e-3),
            F("pore", x=0.001, y=-0.0075, size=0.4e-3),
            F("crack", x=0.002, y=0.0, size=10e-3, size2=4e-3, width=0.04e-3, angle=5),
            F("slag", x=-0.006, y=0.003, size=1.4e-3), F("tungsten", x=0.005, y=-0.012, size=0.7e-3),
            F("undercut", x=0.0078, y=0.0, size=0.8e-3, size2=0.4e-3)]


SCENARIOS = [
    dict(name="Steel butt weld, 200 kV X-ray",
         desc="12 mm plate + 2 mm cap, 0.5 mm Cu filter, CsI DDA (100 µm pixels). Porosity, a "
              "tight crack, slag, a tungsten inclusion, undercut and a W10–W19 wire IQI.",
         values=dict(source="xray", kv=200, ma=5, expo=20, focal=1.0, filt=0.5, sod=700, odd=30,
                     material="steel", thickness=12, cap=2, pixel=100, srb=130, scatter=0.6, weld=1),
         flaws=_weld_flaws, iqi="W10-W19"),
    dict(name="Ir-192 on a 25 mm weld",
         desc="40 Ci Ir-192, 1 min: much lower contrast than X-rays at the same thickness "
              "(higher mean energy) — fewer IQI wires are visible.",
         values=dict(source="Ir-192", activity=40, expo=60, focal=3.0, filt=0, sod=700, odd=40,
                     material="steel", thickness=25, cap=2.5, pixel=100, srb=130, scatter=1.0, weld=1),
         flaws=_weld_flaws, iqi="W8-W16"),
    dict(name="Aluminium casting step wedge",
         desc="Al 6061 steps 5–30 mm at 90 kV: exposure latitude, scatter build-up and pores "
              "of 0.5–2 mm in each step.",
         values=dict(source="xray", kv=90, ma=4, expo=10, focal=0.4, filt=0.0, sod=800, odd=20,
                     material="aluminium-6061", thickness=15, cap=0, pixel=100, srb=100, scatter=0.4, weld=0),
         flaws=lambda: [F("pore", x=x, y=y, size=d) for x in (-0.025, -0.015, -0.005, 0.005, 0.015, 0.025)
                        for y, d in ((-0.01, 2e-3), (-0.002, 1e-3), (0.006, 0.5e-3))],
         wedge=[5e-3, 10e-3, 15e-3, 20e-3, 25e-3, 30e-3], iqi="W12-W18"),
    dict(name="Crack orientation",
         desc="A 4 mm high, 30 µm wide crack is only visible when the beam runs along its plane. "
              "Change its tilt in the defect list.",
         values=dict(source="xray", kv=220, ma=5, expo=20, focal=1.0, filt=0.5, sod=700, odd=20,
                     material="steel", thickness=15, cap=0, pixel=75, srb=80, scatter=0.6, weld=0),
         flaws=lambda: [F("crack", x=0.0, y=0.0, size=20e-3, size2=4e-3, width=0.03e-3, angle=0)],
         iqi="W12-W18"),
    dict(name="Geometric unsharpness",
         desc="3 mm focal spot with a large object-detector distance: U_g = F·b/a blurs the "
              "pores and the thin wires; move the detector closer or use a micro-focus tube.",
         values=dict(source="xray", kv=200, ma=5, expo=20, focal=3.0, filt=0.5, sod=400, odd=150,
                     material="steel", thickness=12, cap=2, pixel=100, srb=130, scatter=0.6, weld=1),
         flaws=_weld_flaws, iqi="W10-W19"),
    dict(name="Co-60 on 80 mm steel",
         desc="Heavy section: 1.25 MeV lines, high scatter build-up, low subject contrast.",
         values=dict(source="Co-60", activity=30, expo=600, focal=4.0, filt=0, sod=1000, odd=50,
                     material="steel", thickness=80, cap=3, pixel=200, srb=250, scatter=2.0, weld=1),
         flaws=_weld_flaws, iqi="W6-W14"),
]


class RadiographyPage(Workspace):
    title = "Radiography"
    VIEW_KEYS = ("display",)
    SCENARIOS = SCENARIOS
    CHOICES = [
        C("source", "Source", [("X-ray tube (W anode)", "xray"), ("Ir-192", "Ir-192"),
                               ("Se-75", "Se-75"), ("Co-60", "Co-60")], "xray"),
        C("material", "Part material", [("Carbon steel", "steel"), ("Stainless 304", "stainless-304"),
                                        ("Aluminium 6061", "aluminium-6061"),
                                        ("Ti-6Al-4V", "titanium-6al4v"), ("Inconel 625", "inconel-625"),
                                        ("Copper", "copper")], "steel"),
        C("weld", "Geometry", [("Plate", 0), ("Butt weld with cap", 1)], 1),
        C("display", "Display", [("Detector (bright = dose)", "dda"), ("Film look (dark = dose)", "film")],
          "film"),
    ]
    PARAMS = [
        P("kv", "Tube voltage (kV)", 30, 450, 200, 1, decimals=0),
        P("ma", "Tube current (mA)", 0.1, 20, 5, 1, decimals=2),
        P("activity", "Activity (Ci)", 1, 150, 40, 1, decimals=0),
        P("expo", "Exposure time (s)", 0.1, 3600, 20, 1, log=True, decimals=1),
        P("focal", "Focal spot (mm)", 0.005, 5, 1.0, 1e-3, log=True, decimals=3),
        P("filt", "Cu filter (mm)", 0, 4, 0.5, 1, decimals=2),
        P("sod", "Source–object (mm)", 100, 2000, 700, 1e-3, decimals=0),
        P("odd", "Object–detector (mm)", 1, 500, 30, 1e-3, decimals=0),
        P("thickness", "Thickness (mm)", 1, 120, 12, 1e-3, decimals=1),
        P("cap", "Weld cap (mm)", 0, 6, 2, 1e-3, decimals=1),
        P("pixel", "Pixel pitch (µm)", 20, 400, 100, 1e-6, decimals=0),
        P("srb", "Detector SRb (µm)", 20, 500, 130, 1e-6, decimals=0),
        P("scatter", "Scatter build-up k", 0, 3, 0.6, 1, decimals=2),
    ]

    def build_views(self):
        self.w_img = W.ImagePanel("Radiograph", "x (mm)", "y (mm)", cmap="CET-L1", invert_y=False)
        self.w_prof = W.plot("Profiles", "position (mm)", "grey value", legend=True)
        self.w_spec = W.plot("Photon spectrum at the detector", "E (keV)", "photons / pixel / keV",
                             legend=True)
        self.w_spec.getPlotItem().setLogMode(x=False, y=True)
        self.w_mu = W.plot("Mass attenuation coefficient", "E (keV)", "μ/ρ (cm²/g)", legend=True)
        self.w_mu.getPlotItem().setLogMode(x=True, y=True)
        return [("Radiograph", self.w_img), ("Profiles", self.w_prof), ("Spectrum", self.w_spec),
                ("Attenuation", self.w_mu)]

    DEFECT_NOTE = "x across the weld, y along it (object plane); crack height is the extent along the beam."
    DEFECTS = [
        C("kind", "Type", [("Gas pore", "pore"), ("Slag inclusion", "slag"),
                           ("Tungsten inclusion", "tungsten"), ("Crack", "crack"),
                           ("Lack of fusion", "lof"), ("Undercut", "undercut")], "pore"),
        P("x", "x (mm)", -28, 28, 0, 1e-3, decimals=1),
        P("y", "y (mm)", -18, 18, 0, 1e-3, decimals=1),
        P("size", "Diameter / length / width (mm)", 0.1, 30, 1.0, 1e-3, decimals=2),
        P("size2", "Crack height / undercut depth (mm)", 0.05, 20, 4.0, 1e-3, decimals=2),
        P("width", "Crack opening (µm)", 5, 500, 40, 1e-6, decimals=0),
        P("angle", "Crack tilt to beam (°)", 0, 60, 5, 1, decimals=1),
    ]
    NAMES = dict(pore="pore", slag="slag", tungsten="tungsten", crack="crack",
                 lof="lack of fusion", undercut="undercut")

    def describe_defect(self, d):
        k = d["kind"]
        if k in ("pore", "slag", "tungsten"):
            what = f"Ø{d['size']:g} mm"
        elif k == "undercut":
            what = f"{d['size']:g} mm wide, {d['size2']:g} mm deep"
        else:
            what = f"{d['size']:g} × {d['size2']:g} mm, {d['width']:g} µm"
            if k == "crack":
                what += f", {d['angle']:g}°"
        return f"{self.NAMES[k]} {what} at ({d['x']:g}, {d['y']:g})"

    def defect_hidden(self, d):
        k = d["kind"]
        if k in ("pore", "slag", "tungsten"):
            return ("size2", "width", "angle")
        if k == "undercut":
            return ("width", "angle")
        return ("angle",) if k == "lof" else ()

    def scenario_defects(self, sc):
        fl = sc["flaws"]() if callable(sc.get("flaws")) else []
        return [dict(kind=f.kind, x=f.x * 1e3, y=f.y * 1e3, size=f.size * 1e3,
                     size2=f.size2 * 1e3, width=f.width * 1e6, angle=f.angle) for f in fl]

    VIEW_HELP = {
        "Radiograph": "Simulated digital radiograph with quantum noise, scatter and unsharpness. "
                      "In the film look, more dose is darker, so pores and cracks (less metal) "
                      "show dark and tungsten bright. Numbers mark the defects, W-labels the IQI "
                      "wires. Hover for the grey value.",
        "Profiles": "Grey value across the weld at y = 0 (noisy and noise-free) and along y "
                    "through the wire IQI.",
        "Spectrum": "Photon spectrum per detector pixel: from the source, after the part, and the "
                    "part actually absorbed by the scintillator. Beam hardening shifts it to "
                    "higher energy.",
        "Attenuation": "Mass attenuation coefficient μ/ρ (photoelectric + Compton + coherent) of "
                       "the part and reference materials.",
    }

    def equations(self):
        return [
            W.eq("Attenuation",
                 r"$\dfrac{\mu}{\rho}=\dfrac{N_A}{A}\sum_m w_m\left[Z\,\sigma_{KN}(E)+"
                 r"\tau_{pe}(E)+\sigma_{coh}(E)\right]$",
                 r"$\tau_{pe}=C\,Z^nE^{-m(Z)}$",
                 "Klein–Nishina Compton cross-section; photoelectric fit to XCOM within ±5 %."),
            W.eq("Image formation",
                 r"$S(x,y)=B\sum_E N(E)\,\eta(E)\,E\,"
                 r"\exp\!\left(-\sum_m\mu_m(E)\,L_m(x,y)\right)$",
                 r"$\eta(E)=1-e^{-\mu_{CsI}t},\qquad B=1+k\bar{\mu}x$",
                 r"$N(E)\propto Z\,\dfrac{E_0-E}{E}$"),
            W.eq("Unsharpness and magnification",
                 r"$U_g=\dfrac{F\,b}{a},\qquad U_T=\left(U_g^3+U_D^3\right)^{1/3},\qquad "
                 r"M=\dfrac{a+b}{a}$"),
            W.eq("Noise",
                 r"$\sigma^2=\dfrac{\sum N\eta E^2\,(1+\mathrm{SPR})}{A_S},\qquad"
                 r"\mathrm{SNR_N}=\mathrm{SNR}\,\dfrac{88.6\,\mu\mathrm{m}}{SR_b}$"),
            W.eq("Detectability (Rose)",
                 r"$d'=\dfrac{\Delta I}{\sigma}\sqrt{n_{px}}\;\geq\;5$",
                 "Used for the IQI wires and for every defect in the monitor: ΔI is the mean "
                 "noise-free contrast over the defect's projected footprint (at least the "
                 "unsharpness wide), σ the pixel noise there."),
        ]

    def hidden_keys(self, p):
        h = {"activity"} if p["source"] == "xray" else {"kv", "ma", "filt"}
        if not p["weld"]:
            h.add("cap")
        return h

    def setup(self, p):
        src = rt.Source(kind=p["source"], kv=p["kv"], ma=p["ma"], exposure_s=p["expo"],
                        activity_ci=p["activity"], focal_spot=p["focal"], filter_mm=p["filt"])
        det = rt.Detector(pixel=p["pixel"], srb=p["srb"])
        return rt.RTSetup(src, det, rt.Geometry(p["sod"], p["odd"]), scatter_k=p["scatter"])

    def obj(self, p):
        sc = self.scenario
        out = [F(d["kind"], d["x"], d["y"], d["size"], d["size2"], d["width"], d["angle"])
               for d in p.get("_defects", [])]
        return rt.RTObject(material=p["material"], thickness=p["thickness"], weld=bool(p["weld"]),
                           cap=p["cap"] if p["weld"] else 0.0, flaws=out, iqi=sc.get("iqi"),
                           step_wedge=sc.get("wedge"))

    def compute(self, p, heavy, ctx):
        s = self.setup(p)
        o = self.obj(p)
        res = rt.radiograph(s, o, n_e=70)
        ctx.check()
        wires, best = rt.wire_visibility(res, o)
        ref = rt.radiograph(s, dataclasses.replace(o, flaws=[]), n_e=70, noise=False,
                            gain=res["gain"]) if o.flaws else None
        meas = self.measure(res, ref, o) if o.flaws else []
        ny, nx = res["image"].shape
        reg = (slice(int(0.05 * ny), int(0.3 * ny)), slice(int(0.75 * nx), int(0.95 * nx)))
        snrn, snr = rt.snr_normalized(res["image"], reg, s.detector.srb)
        E, N = s.source.spectrum(150)
        mats = [p["material"], "aluminium-6061", "copper", "lead", "water", "csi"]
        Eg = np.geomspace(10, 2000, 200)
        mus = {m: rt.mu_rho(m, Eg) for m in dict.fromkeys(mats)}
        # transmitted spectrum through the nominal thickness
        L = o.thickness + (o.cap if o.weld else 0)
        T = np.exp(-rt.mu(p["material"], E) * L)
        return dict(res=res, setup=s, obj=o, wires=wires, best=best, snrn=snrn, snr=snr,
                    spec=(E, N, T), mus=(Eg, mus), region=reg, meas=meas)

    @staticmethod
    def _footprint(f, X, Y, grow=0.0):
        """Projected footprint of a defect on the detector grid, widened by `grow`."""
        if f.kind in ("pore", "slag", "tungsten"):
            return (X - f.x) ** 2 + (Y - f.y) ** 2 <= (f.size / 2 + grow) ** 2
        if f.kind in ("crack", "lof"):
            ang = np.radians(f.angle if f.kind == "crack" else 30.0)
            w = f.size2 * np.sin(ang) + f.width * np.cos(ang)
            return (np.abs(X - f.x) <= w / 2 + 0.15e-3 + grow) & (np.abs(Y - f.y) <= f.size / 2 + grow)
        return (np.abs(X - f.x) <= f.size / 2 + grow) & (np.abs(Y - f.y) < 15e-3 + grow)

    def measure(self, res, ref, o):
        """Per defect: noise-free signal (radiograph minus the defect-free one) over the pixels
        where it exceeds a quarter of its peak; d′ = |ΔI|/σ·√n_px."""
        x, y = res["x"], res["y"]
        X, Y = np.meshgrid(x, y)
        sig, noise = res["clean"] - ref["clean"], res["image"] - res["clean"]
        u = max(res["unsharpness"] / res["magnification"], res["p_obj"])
        out = []
        for f in o.flaws:
            own = self._footprint(f, X, Y, 1.5 * u + 0.3e-3)
            if not own.any():
                out.append(None); continue
            s_ = np.where(own, sig, 0.0)
            peak = np.max(np.abs(s_))
            core = own & (np.abs(s_) >= 0.25 * peak) if peak > 0 else own
            dI = float(np.mean(sig[core]))
            bg = float(np.mean(ref["clean"][core]))
            sd = float(np.std(noise[own])) or 1e-30
            d = abs(dI) / sd * np.sqrt(core.sum())
            out.append(dict(contrast=dI / max(bg, 1e-30) * 100, cnr=abs(dI) / sd, dprime=d,
                            npx=int(core.sum()), detected=d >= 5.0))
        return out

    def monitor(self, r):
        heads = ["#", "type", "x, y", "size", "contrast %", "CNR", "d′", "status"]
        rows, flags = [], []
        for i, (f, m) in enumerate(zip(r["obj"].flaws, r["meas"])):
            size = (f"Ø{f.size * 1e3:.2g} mm" if f.kind in ("pore", "slag", "tungsten") else
                    f"{f.size * 1e3:.3g} × {f.size2 * 1e3:.3g} mm")
            if m is None:
                rows.append([str(i + 1), self.NAMES[f.kind], f"{f.x * 1e3:.1f}, {f.y * 1e3:.1f} mm",
                             size, "—", "—", "—", "outside view"]); flags.append(False); continue
            rows.append([str(i + 1), self.NAMES[f.kind], f"{f.x * 1e3:.1f}, {f.y * 1e3:.1f} mm",
                         size, f"{m['contrast']:+.2f}", f"{m['cnr']:.2f}", f"{m['dprime']:.1f}",
                         "detected" if m["detected"] else "missed"])
            flags.append(m["detected"])
        return heads, rows, flags

    def show_result(self, r):
        res = r["res"]
        p = self.params.values()
        img = res["image"]
        x, y = res["x"] * 1e3, res["y"] * 1e3
        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
        if p["display"] == "film":
            self.w_img.set(-img, x, y, levels=(-hi, -lo), title="Radiograph (film look)")
        else:
            self.w_img.set(img, x, y, levels=(lo, hi), title="Radiograph (detector grey values)")
        self.w_img.clear_overlays()
        wires = r["obj"].iqi_wires()
        keep = {wires[0][0], wires[-1][0], r["best"]} if wires else set()
        for num, d, xw, yw, half in wires:
            if num not in keep:
                continue
            seen = num == r["best"]
            t = pg.TextItem(f"W{num} ✓" if seen else f"W{num}",
                            color=theme.color("hit" if seen else 3), anchor=(1, 0.5))
            t.setPos((xw - half) * 1e3 - 0.5, yw * 1e3)
            self.w_img.add(t)
        for i, f in enumerate(r["obj"].flaws):
            t = pg.TextItem(str(i + 1), color=theme.color(5), anchor=(0.5, 1.0))
            t.setPos(f.x * 1e3, (f.y + (f.size / 2 if f.kind in ("pore", "slag", "tungsten")
                                        else min(f.size / 2, 4e-3))) * 1e3 + 0.6)
            self.w_img.add(t)
        pr = self.w_prof.getPlotItem(); pr.clear()
        if pr.legend is not None:
            pr.legend.clear()
        mid = img.shape[0] // 2
        W.curve(pr, x, img[mid], 0, 1.0, name="across the weld (y = 0)")
        W.curve(pr, x, res["clean"][mid], 1, 1.4, name="noise-free")
        wires = r["obj"].iqi_wires()
        if wires:
            ix = slice(np.searchsorted(x, -3), np.searchsorted(x, 3))
            W.curve(pr, y, img[:, ix].mean(axis=1), 2, 1.2, name="along y over the IQI (mean of 6 mm)")
        ps = self.w_spec.getPlotItem(); ps.clear()
        if ps.legend is not None:
            ps.legend.clear()
        E, N, T = r["spec"]
        geo = r["setup"].geometry
        det = r["setup"].detector
        Np = N / geo.sdd ** 2 * det.pixel ** 2
        dE = np.gradient(E) if len(E) > 1 else np.ones_like(E)
        if r["setup"].source.kind == "xray":
            ys = (Np / dE, Np * T / dE, Np * T * det.absorption(E) / dE)
            top = float(np.max(ys[0]))
            floor = max(float(np.max(ys[2])) * 1e-4, top * 1e-9, 1e-12)
            for k, (yy, nm) in enumerate(zip(ys, ("incident", "behind the part",
                                                  "absorbed in CsI"))):
                W.curve(ps, E, np.maximum(yy, floor), k, 1.6, name=nm)
            ps.setYRange(np.log10(floor), np.log10(top) + 0.3, padding=0)
        else:
            for k, (yy, nm) in enumerate(((Np, "incident lines"), (Np * T, "behind the part"))):
                sc = pg.BarGraphItem(x=E, height=np.log10(np.maximum(yy, 1e-6)), width=6,
                                     brush=theme.color(k), name=nm)
                ps.addItem(sc)
        pm = self.w_mu.getPlotItem(); pm.clear()
        if pm.legend is not None:
            pm.legend.clear()
        Eg, mus = r["mus"]
        for k, (m, v) in enumerate(mus.items()):
            W.curve(pm, Eg, v, k, 1.6, name=m)
        W.log_ticks(pm, "bottom", Eg[0], Eg[-1])

    def summary(self, r):
        res, s, o = r["res"], r["setup"], r["obj"]
        geo = s.geometry
        L = o.thickness + (o.cap if o.weld else 0)
        rows = [("penetrated thickness", W.fmt(L * 1e3, "mm")),
                ("mean energy at detector", W.fmt(res["E_mean_transmitted"], "keV")),
                ("μ at that energy", W.fmt(rt.mu(o.material, res["E_mean_transmitted"]) / 100, "1/cm")),
                ("HVL at that energy", W.fmt(rt.half_value_layer(o.material, res["E_mean_transmitted"]) * 1e3, "mm")),
                ("scatter-to-primary", W.fmt(float(np.median(res["spr"])))),
                ("magnification", W.fmt(res["magnification"])),
                ("U_g / U_D / U_T", f"{res['ug'] * 1e3:.3f} / {res['ud'] * 1e3:.3f} / {res['unsharpness'] * 1e3:.3f} mm"),
                ("SNR / SNR_N", f"{r['snr']:.0f} / {r['snrn']:.0f}"),
                ("visible IQI wire", f"W{r['best']}" if r["best"] else "none"),
                ]
        if r["meas"]:
            rows.append(("defects detected (d′ ≥ 5)",
                         f"{sum(bool(m and m['detected']) for m in r['meas'])} / {len(r['meas'])}"))
        if r["best"]:
            rows.append(("contrast sensitivity", W.fmt(rt.WIRE_IQI[r["best"] - 1] / (L * 1e3) * 100, "%")))
        for w in r["wires"][-4:]:
            rows.append((f"  {w['wire']} ({w['diameter_mm']:.3g} mm) d′", W.fmt(w["dprime"], "", 3)))
        return rows
