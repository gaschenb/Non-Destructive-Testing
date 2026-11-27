"""Guided (Lamb / SH) waves in plates: dispersion, mode shapes, propagation, long-range screening."""
from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np

from ..core import guided_waves as gw
from ..core.materials import get
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

def _d(x, loss, width):
    """Defect of an example problem: distance (m), wall loss (%), axial length (mm)."""
    return dict(x=x, loss=loss, width=width)


SCENARIOS = [
    dict(name="Aluminium plate dispersion",
         desc="1 mm Al 6061: S0, A0 and the higher modes up to 5 MHz·mm with their cut-offs; the "
              "operating point sits in the low-dispersion S0 region.",
         defects=[_d(0.4, 30, 3)],
         values=dict(material="aluminium-6061", thickness=1, fmax=5, mode="S0", f0=300, cycles=5,
                     xmax=0.5, length=1.0, att=2, noise=1, lam=0)),
    dict(name="A0 dispersion over distance",
         desc="2 mm Al, 100 kHz, 3 cycles: the flexural A0 burst spreads and develops a chirp "
              "(high frequencies run ahead) — the reason long-range tests use many cycles.",
         defects=[_d(0.6, 30, 3)],
         values=dict(material="aluminium-6061", thickness=2, fmax=2, mode="A0", f0=100, cycles=3,
                     xmax=1.0, length=1.5, att=2, noise=1, lam=0)),
    dict(name="S0 above the knee",
         desc="S0 at f·d ≈ 2.4 MHz·mm in steel: group velocity collapses towards the Rayleigh "
              "regime and pulses disperse strongly.",
         defects=[_d(0.3, 30, 2)],
         values=dict(material="steel", thickness=2, fmax=3, mode="S0", f0=1200, cycles=5,
                     xmax=0.4, length=0.6, att=5, noise=1, lam=0)),
    dict(name="Long-range screening (SH0)",
         desc="10 mm steel wall, 40 kHz SH0 (the plate analogue of the T(0,1) pipe mode): "
              "non-dispersive, so 10-cycle bursts stay compact over 10 m. Corrosion patches at "
              "4 m and 6.5 m.",
         defects=[_d(4, 25, 40), _d(6.5, 12, 60)],
         values=dict(material="steel", thickness=10, fmax=0.6, mode="SH0", f0=40, cycles=10,
                     xmax=10, length=12, att=0.15, noise=1, lam=0)),
    dict(name="Wall-loss screening with A0",
         desc="6 mm steel, 50 kHz A0: sensitive to wall thinning; part of each echo returns as "
              "S0 (mode conversion) and arrives earlier.",
         defects=[_d(2.5, 40, 30), _d(4.2, 20, 25)],
         values=dict(material="steel", thickness=6, fmax=1.0, mode="A0", f0=50, cycles=8,
                     xmax=5, length=6, att=0.5, noise=1, lam=0)),
    dict(name="Comb transducer mode selection",
         desc="A comb of pitch λ excites modes where c_p = λf (straight line through the origin). "
              "Choose λ to excite a single mode.",
         defects=[_d(0.4, 30, 3)],
         values=dict(material="aluminium-6061", thickness=1.5, fmax=6, mode="S0", f0=400, cycles=5,
                     xmax=0.5, length=1.0, att=2, noise=1, lam=12)),
    dict(name="Zero-group-velocity S1 resonance",
         desc="Steel: the S1 mode has a minimum of f·d where c_g = 0 (backward wave below it). "
              "Laser-ultrasonic ZGV resonances measure thickness locally.",
         defects=[_d(0.1, 30, 1)],
         values=dict(material="steel", thickness=1, fmax=4, mode="S1", f0=2800, cycles=20,
                     xmax=0.05, length=0.2, att=2, noise=1, lam=0)),
]

MODES = ["S0", "A0", "SH0", "S1", "A1", "S2", "A2"]


class GuidedPage(Workspace):
    title = "Guided waves"
    SCENARIOS = SCENARIOS
    VIEW_KEYS = ("axis",)
    CHOICES = [
        C("material", "Plate material", [("Aluminium 6061", "aluminium-6061"), ("Aluminium 2024", "aluminium-2024"),
                                         ("Carbon steel", "steel"), ("Stainless 304", "stainless-304"),
                                         ("Ti-6Al-4V", "titanium-6al4v"), ("Inconel 625", "inconel-625"),
                                         ("Copper", "copper"), ("PMMA", "pmma")], "aluminium-6061"),
        C("mode", "Mode", [(m, m) for m in MODES], "S0"),
        C("axis", "Dispersion axis", [("f·d (MHz·mm)", "fd"), ("f (kHz)", "f")], "fd"),
    ]
    PARAMS = [
        P("thickness", "Thickness d (mm)", 0.2, 40, 1, 1e-3, log=True, decimals=2),
        P("fmax", "Max f·d (MHz·mm)", 0.2, 12, 5, 1, decimals=2),
        P("f0", "Centre frequency (kHz)", 5, 5000, 300, 1e3, log=True, decimals=1),
        P("cycles", "Cycles (Hann)", 1, 30, 5, 1, integer=True),
        P("xmax", "Max propagation distance (m)", 0.02, 20, 0.5, 1, log=True, decimals=3),
        P("length", "Plate length (m)", 0.1, 30, 1, 1, log=True, decimals=2),
        P("att", "Attenuation (dB/m)", 0, 20, 2, 1, decimals=2),
        P("noise", "Coherent noise (%)", 0, 20, 1, 0.01, decimals=1),
        P("lam", "Comb wavelength λ (mm, 0 = off)", 0, 60, 0, 1e-3, decimals=1),
    ]

    def __init__(self, *a, **k):
        self._cache = {}
        self._lock = threading.Lock()
        super().__init__(*a, **k)

    def build_views(self):
        self.w_cp = W.plot("Phase velocity", "f·d (MHz·mm)", "c_p (m/s)", legend=True)
        self.w_cg = W.plot("Group velocity", "f·d (MHz·mm)", "c_g (m/s)", legend=True)
        self.w_shape = W.plot("Mode shape at the centre frequency", "displacement (normalised)",
                              "z (mm)", legend=True)
        self.w_prop = W.plot("Propagated burst", "t (µs)", "distance (m) + signal", legend=False)
        self.w_scan = W.plot("Pulse-echo screening", "distance (m)", "amplitude", legend=True)
        return [("Phase velocity", self.w_cp), ("Group velocity", self.w_cg),
                ("Mode shape", self.w_shape), ("Propagation", self.w_prop),
                ("Screening", self.w_scan)]

    DEFECT_NOTE = "Wall-loss patches along the strip, at a distance from the transducer."
    DEFECTS = [
        P("x", "Distance (m)", 0.01, 25, 0.4, 1, log=True, decimals=3),
        P("loss", "Wall loss (%)", 1, 90, 30, 0.01, decimals=0),
        P("width", "Axial length (mm)", 0.5, 200, 3, 1e-3, log=True, decimals=1),
    ]

    def describe_defect(self, d):
        return f"{d['loss']:g} % wall loss, {d['width']:g} mm long at {d['x']:g} m"

    VIEW_HELP = {
        "Phase velocity": "Phase velocity of every propagating mode versus f·d (or f). The "
                          "selected mode is drawn thick; the dashed line is the centre "
                          "frequency. Hover for values.",
        "Group velocity": "Energy (group) velocity of each mode: the speed at which a burst "
                          "travels. Negative values are backward waves below the ZGV point.",
        "Mode shape": "Through-thickness displacement of the selected mode at the centre "
                      "frequency: in-plane u_x and out-of-plane u_z (SH: u_y).",
        "Propagation": "The burst at five distances; dashed markers show the arrival predicted "
                       "by the group velocity. Dispersion spreads and chirps the burst.",
        "Screening": "Pulse-echo trace converted to distance with the group velocity. Dashed "
                     "lines are the true defect positions, numbered as in the defect list.",
    }

    def equations(self):
        return [
            W.eq("Rayleigh–Lamb dispersion",
                 r"$p^2=\dfrac{\omega^2}{c_L^2}-k^2,\qquad q^2=\dfrac{\omega^2}{c_T^2}-k^2,"
                 r"\qquad h=\dfrac{d}{2}$",
                 r"Symmetric:  $\dfrac{\tan qh}{\tan ph}=-\dfrac{4k^2pq}{(q^2-k^2)^2}$",
                 r"Antisymmetric:  $\dfrac{\tan qh}{\tan ph}=-\dfrac{(q^2-k^2)^2}{4k^2pq}$",
                 r"SH$_n$:  $c_p=\dfrac{c_T}{\sqrt{1-\left(n\,c_T/2fd\right)^2}}$"),
            W.eq("Group velocity",
                 r"$c_g=\dfrac{d\omega}{dk}=\dfrac{c_p^2}{c_p-f\,dc_p/df}$"),
            W.eq("Propagation",
                 r"$u(x,t)=\dfrac{1}{\sqrt{x}}\,\mathcal{F}^{-1}\!\left[U(\omega)\,"
                 r"e^{-ik(\omega)x-\alpha x}\right]$"),
            W.eq("Defect echo and comb excitation",
                 r"$|R|\approx\dfrac{\Delta d}{d}\left(\left|\sin\dfrac{\pi w}{\lambda}\right|"
                 r"+0.05\right),\qquad c_p=\lambda f$"),
            W.eq("Defect monitor",
                 "Echo peak in a gate around each defect's expected arrival, located with the "
                 "group velocity; level relative to the plate-end echo; SNR against the trace "
                 "rms outside all echoes. Detected when SNR ≥ 6 dB."),
        ]

    def dispersion(self, p):
        key = (p["material"], round(p["thickness"], 9), round(p["fmax"], 6))
        with self._lock:
            if key not in self._cache:
                fmax = p["fmax"] * 1e3 / p["thickness"]          # MHz·mm → Hz
                d = gw.lamb_dispersion(p["material"], p["thickness"], f_max=fmax, n_f=240,
                                       max_modes=4, n_k=320)
                sh = gw.sh_dispersion(p["material"], p["thickness"], d.f, n_modes=3)
                modes = dict(d.modes); group = dict(d.group)
                for k, (cp, cg) in sh.items():
                    modes[k], group[k] = cp, cg
                if len(self._cache) > 12:
                    self._cache.clear()
                self._cache[key] = replace(d, modes=modes, group=group)
            return self._cache[key]

    def compute(self, p, heavy, ctx):
        d = self.dispersion(p)
        mode = p["mode"] if p["mode"] in d.modes else "S0"
        f0, cyc = p["f0"], int(p["cycles"])
        cp0 = float(gw.phase_velocity_at(d, mode, f0))
        g = d.group[mode]
        good = np.isfinite(g)
        cg0 = float(np.interp(f0, d.f[good], g[good], left=np.nan, right=np.nan)) if good.any() else np.nan
        out = dict(disp=d, mode=mode, cp0=cp0, cg0=cg0)
        if np.isfinite(cp0) and mode.startswith(("S", "A")) and not mode.startswith("SH"):
            out["shape"] = gw.mode_shape(p["material"], p["thickness"], f0, cp0, mode[0])
        ctx.check()
        if np.isfinite(cp0):
            xs = np.linspace(p["xmax"] / 5, p["xmax"], 5)
            t, sig = gw.propagate(d, mode, f0, cyc, xs, attenuation_db_m=p["att"])
            out["prop"] = (t, sig, xs)
            dfs = [gw.PlateDefect(q["x"], q["loss"], q["width"]) for q in p.get("_defects", [])]
            dfs = [q for q in dfs if q.x < p["length"]]
            mc = 0.0 if mode.startswith("SH") else 0.15
            t2, s2, _ = gw.pulse_echo_plate(d, mode, f0, cyc, dfs, length=p["length"],
                                            attenuation_db_m=p["att"], noise=p["noise"],
                                            mode_conversion=mc)
            out["scan"] = (t2, s2, dfs)
            if np.isfinite(cg0):
                out["meas"] = self.measure(t2, s2, dfs, cg0, cp0, p)
        if p["lam"] > 0:
            hits = {}
            for k in d.modes:
                try:
                    fh = gw.excitability_wavelength(d, k, p["lam"])
                except Exception:
                    fh = np.nan
                if np.isfinite(fh):
                    hits[k] = fh
            out["comb"] = hits
        out["zgv"] = gw.zgv_frequency(d, "S1")
        return out

    def measure(self, t, s, dfs, cg, cp, p):
        from ..core.signals import envelope
        env = envelope(s)
        dist = (t - p["cycles"] / (2 * p["f0"])) * cg / 2      # burst centre → distance
        burst = p["cycles"] / p["f0"] * cg / 2
        hw = [0.75 * burst + 2 * q.width + 0.02 * q.x for q in dfs]
        quiet = (dist > 1.5 * burst) & (np.abs(dist - p["length"]) > 1.5 * burst)
        for q, h in zip(dfs, hw):
            quiet &= np.abs(dist - q.x) > 2 * h
        rms = float(np.sqrt(np.mean(env[quiet] ** 2))) if quiet.sum() > 20 else float(np.median(env))
        endg = np.abs(dist - p["length"]) < 1.5 * burst
        ref = float(env[endg].max()) if endg.any() else float(env.max())
        lam = cp / p["f0"]
        out = []
        for q, h in zip(dfs, hw):
            g = np.abs(dist - q.x) < h
            if not g.any():
                out.append(None); continue
            k = np.flatnonzero(g)[np.argmax(env[g])]
            a = float(env[k])
            R = q.depth_ratio * abs(np.sin(np.pi * q.width / lam)) + 0.05 * q.depth_ratio
            snr = 20 * np.log10(max(a, 1e-30) / max(rms, 1e-30))
            out.append(dict(x=float(dist[k]), db=20 * np.log10(max(a, 1e-30) / max(ref, 1e-30)),
                            R=20 * np.log10(max(R, 1e-9)), snr=snr, detected=snr >= 6.0))
        return out

    def monitor(self, r):
        heads = ["#", "true distance", "wall loss", "located at", "|R| dB", "dB re end echo",
                 "SNR dB", "status"]
        rows, flags = [], []
        dfs = r["scan"][2] if "scan" in r else []
        for i, (q, m) in enumerate(zip(dfs, r.get("meas", [None] * len(dfs)))):
            base = [str(i + 1), f"{q.x:.3g} m", f"{q.depth_ratio * 100:.0f} %"]
            if m is None:
                rows.append(base + ["—", "—", "—", "—", "missed"]); flags.append(False); continue
            rows.append(base + [f"{m['x']:.3g} m", f"{m['R']:.1f}", f"{m['db']:.1f}",
                                f"{m['snr']:.1f}", "detected" if m["detected"] else "missed"])
            flags.append(m["detected"])
        return heads, rows, flags

    def show_result(self, r):
        p = self.params.values()
        d = r["disp"]
        fd_axis = p["axis"] == "fd"
        X = d.f / 1e6 * d.thickness * 1e3 if fd_axis else d.f / 1e3
        xl = "f·d (MHz·mm)" if fd_axis else "f (kHz)"
        x0 = p["f0"] / 1e6 * d.thickness * 1e3 if fd_axis else p["f0"] / 1e3
        m = get(p["material"])
        conv = (lambda f: f / 1e6 * d.thickness * 1e3) if fd_axis else (lambda f: f / 1e3)
        curves = {}
        for k, b in (d.branches or {}).items():
            curves[k] = (conv(b["f"]), b["cp"], b["cg"])
        for k in d.modes:
            if k.startswith("SH"):
                curves[k] = (X, d.modes[k], d.group[k])
        for w, which, ttl, yl in ((self.w_cp, 1, "Phase velocity", "c_p (m/s)"),
                                  (self.w_cg, 2, "Group velocity", "c_g (m/s)")):
            pi = w.getPlotItem(); pi.clear()
            if pi.legend is not None:
                pi.legend.clear()
            W.style(pi, ttl, xl, yl)
            names = sorted(curves, key=lambda k: (k.startswith("SH"), int(k[-1]), k))
            for k in names:
                c = 0 if k.startswith("S") and not k.startswith("SH") else 1 if k.startswith("A") else 2
                sty = None if k[-1] == "0" else "dash" if k[-1] == "1" else "dot"
                wd = 3.0 if k == r["mode"] else 1.5
                W.curve(pi, curves[k][0], curves[k][which], c, wd, name=k, style=sty)
            W.vline(pi, x0, "gate", "dash", label=f"f₀ = {p['f0'] / 1e3:g} kHz")
            for lab, val in (("c_L", m.cL), ("c_T", m.cT)):
                W.hline(pi, val, "data", "dot", label=lab)
            pi.setYRange(0, 3 * m.cL) if w is self.w_cp else pi.setYRange(-0.25 * m.cT, 1.1 * m.cL)
            if w is self.w_cg:
                W.hline(pi, 0, "fg", "dot")
            if w is self.w_cp and "comb" in r:
                lam = p["lam"]
                f_line = np.linspace(0, d.f[-1], 50)
                xx = f_line / 1e6 * d.thickness * 1e3 if fd_axis else f_line / 1e3
                W.curve(pi, xx, lam * f_line, "dac", 2.0, name=f"comb λ = {lam * 1e3:g} mm")
        ps = self.w_shape.getPlotItem(); ps.clear()
        if ps.legend is not None:
            ps.legend.clear()
        if "shape" in r:
            z, ux, uz = r["shape"]
            W.curve(ps, ux, z * 1e3, 0, 2.2, name="u_x (in-plane)")
            W.curve(ps, uz, z * 1e3, 1, 2.2, name="u_z (out-of-plane)")
            W.vline(ps, 0, "data", "dot")
            ps.setXRange(-1.1, 1.1)
        elif r["mode"].startswith("SH"):
            n = int(r["mode"][-1])
            z = np.linspace(-d.thickness / 2, d.thickness / 2, 81)
            uy = np.cos(n * np.pi * (z / d.thickness + 0.5))
            W.curve(ps, uy, z * 1e3, 2, 2.2, name="u_y (shear horizontal)")
            ps.setXRange(-1.1, 1.1)
        pp = self.w_prop.getPlotItem(); pp.clear()
        if "prop" in r:
            t, sig, xs = r["prop"]
            sc = np.max(np.abs(sig)) or 1
            step = xs[1] - xs[0] if len(xs) > 1 else 1
            for i, x in enumerate(xs):
                W.curve(pp, t * 1e6, x + 0.45 * step * sig[i] / sc, 0, 1.2)
                if np.isfinite(r["cg0"]):
                    t_arr = x / r["cg0"] + p["cycles"] / (2 * p["f0"])   # burst centre
                    W.curve(pp, [t_arr * 1e6] * 2, [x - 0.45 * step, x + 0.45 * step],
                            "gate", 1.0, style="dash")
            cg = r["cg0"] if np.isfinite(r["cg0"]) and r["cg0"] > 50 else (
                np.nanmin(d.group[r["mode"]][d.group[r["mode"]] > 0])
                if np.any(d.group[r["mode"]] > 0) else 50)
            tm = xs[-1] / max(cg, 50) * 1e6
            pp.setXRange(0, min(t[-1] * 1e6, 1.3 * tm + 2 * p["cycles"] / p["f0"] * 1e6))
        pc = self.w_scan.getPlotItem(); pc.clear()
        if pc.legend is not None:
            pc.legend.clear()
        if "scan" in r and np.isfinite(r["cg0"]):
            t2, s2, dfs = r["scan"]
            dist = (t2 - p["cycles"] / (2 * p["f0"])) * r["cg0"] / 2
            from ..core.signals import envelope
            W.curve(pc, dist, s2, "data", 0.8, name="RF")
            W.curve(pc, dist, envelope(s2), 0, 1.8, name="envelope")
            for i, q in enumerate(dfs):
                W.vline(pc, q.x, "gate", "dash", label=f"#{i + 1}")
            W.vline(pc, p["length"], "ref", "dash", label="plate end", pos=0.6)
            pc.setXRange(0, p["length"] * 1.15)

    def summary(self, r):
        p = self.params.values()
        d = r["disp"]
        m = get(p["material"])
        rows = [("c_L / c_T", f"{m.cL:.0f} / {m.cT:.0f} m/s"),
                ("f·d at f₀", W.fmt(p["f0"] * d.thickness / 1e3, "MHz·mm")),
                (f"{r['mode']} phase velocity", W.fmt(r["cp0"], "m/s")),
                (f"{r['mode']} group velocity", W.fmt(r["cg0"], "m/s")),
                ("wavelength", W.fmt(r["cp0"] / p["f0"] * 1e3, "mm")),
                ("burst length", W.fmt(p["cycles"] * r["cp0"] / p["f0"] * 1e3, "mm")),
                ("modes present at f₀", ", ".join(k for k in d.modes
                                                  if np.isfinite(gw.phase_velocity_at(d, k, p["f0"]))))]
        cut = gw.cutoff_frequencies(p["material"], p["thickness"], n=2)
        rows.append(("first cut-offs (shear / long.)",
                     f"{cut['shear'][0] * d.thickness / 1e3:.3g} / {cut['longitudinal'][0] * d.thickness / 1e3:.3g} MHz·mm"))
        if np.isfinite(r.get("zgv", np.nan)):
            fz = r["zgv"] * d.thickness / 1e3
            rows.append(("S1 zero-group-velocity f·d", W.fmt(fz, "MHz·mm")))
            rows.append(("ZGV thickness at f₀ (same f·d)", W.fmt(fz / (p["f0"] / 1e6), "mm")))
        if "comb" in r:
            for k, fh in sorted(r["comb"].items(), key=lambda kv: kv[1])[:5]:
                rows.append((f"comb excites {k} at", W.fmt(fh / 1e3, "kHz")))
        meas = [m for m in r.get("meas", []) if m is not None]
        if r.get("meas"):
            rows.append(("defects detected (SNR ≥ 6 dB)",
                         f"{sum(m['detected'] for m in meas)} / {len(r['meas'])}"))
        return rows
