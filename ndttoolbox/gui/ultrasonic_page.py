"""Bulk-wave ultrasonic testing: A-scan, B-scan (+SAFT), C-scan, transducer beam, DGS and
interface coefficients."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg

from ..core import signals as sg
from ..core import ultrasonics as ut
from ..core.materials import get
from . import theme
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

MATS = [("Carbon steel", "steel"), ("Stainless 304", "stainless-304"),
        ("Austenitic weld metal", "austenitic-weld"), ("Aluminium 2024", "aluminium-2024"),
        ("Ti-6Al-4V", "titanium-6al4v"), ("Inconel 625", "inconel-625"), ("CFRP laminate", "cfrp"),
        ("Copper", "copper")]

F = ut.Flaw


def _fl(kind, x, z, d, **kw):
    return F(kind, x=x * 1e-3, z=z * 1e-3, size=d * 1e-3, **kw)


SCENARIOS = [
    dict(name="Steel plate with flat-bottom holes",
         desc="50 mm carbon-steel block, Ø2/3/4 mm flat-bottom holes at 15, 25 and 35 mm. "
              "Contact 5 MHz Ø10 mm probe. Compare echo heights with the DGS diagram.",
         values=dict(material="steel", mode="contact", thickness=50, f0=5, bw=60, diam=10,
                     gain=14, xprobe=0, span=60),
         flaws=[_fl("fbh", 0, 25, 3), _fl("fbh", -14, 15, 2), _fl("fbh", 14, 35, 4)]),
    dict(name="Side-drilled hole reference block (DAC)",
         desc="Ø3 mm side-drilled holes at 10–50 mm depth, staggered in x. The DAC curve is "
              "built from the same holes; the A-scan probe sits over the selected one.",
         values=dict(material="steel", mode="contact", thickness=60, f0=4, bw=55, diam=12,
                     gain=19, xprobe=0, span=80, dac=1),
         flaws=[_fl("sdh", 0, 30, 3), _fl("sdh", -24, 10, 3), _fl("sdh", -12, 20, 3),
                _fl("sdh", 12, 40, 3), _fl("sdh", 24, 50, 3)]),
    dict(name="Immersion C-scan of a titanium forging",
         desc="40 mm Ti-6Al-4V billet in immersion (30 mm water path, 10 MHz). Ø2 mm "
              "flat-bottom-hole-like reflector, a tungsten inclusion, a pore and a tilted "
              "planar flaw. Run the C-scan for amplitude and depth maps.",
         values=dict(material="titanium-6al4v", mode="immersion", thickness=40, f0=10, bw=60,
                     diam=12, wp=30, gain=44, xprobe=4,
                     span=40, csize=30),
         flaws=[_fl("fbh", 4, 18, 2), _fl("sphere", -8, 10, 1.5, fill="tungsten"),
                _fl("disc", -5, 28, 3, tilt=10), _fl("sphere", 8, 32, 1.2)], auto_run=True),
    dict(name="Lamination in steel plate",
         desc="25 mm plate with a 16 mm lamination at mid-thickness: strong flaw echo and loss "
              "of back-wall echo. The 6 dB-drop length on the C-scan sizes it.",
         values=dict(material="steel", mode="contact", thickness=25, f0=5, bw=60, diam=10,
                     gain=16, xprobe=0, span=50, csize=40),
         flaws=[_fl("disc", 0, 12, 16)]),
    dict(name="Coarse-grained austenitic weld",
         desc="Austenitic weld metal (≈250 µm grains): structural noise grows with f⁴ "
              "scattering. Drop the frequency to 2 MHz to recover the Ø3 mm side-drilled hole.",
         values=dict(material="austenitic-weld", mode="contact", thickness=40, f0=5, bw=60,
                     diam=12, gain=30, xprobe=0, span=50,
                     grain=1.0),
         flaws=[_fl("sdh", 0, 25, 3)]),
    dict(name="CFRP impact delaminations",
         desc="6 mm CFRP laminate in immersion with stacked delaminations from an impact "
              "(growing with depth). Run the C-scan: the depth map shows the cone.",
         values=dict(material="cfrp", mode="immersion", thickness=6, f0=5, bw=70, diam=10,
                     wp=15, gain=40, xprobe=0, span=40,
                     csize=40),
         flaws=[_fl("disc", 0, 2.0, 8), _fl("disc", 2, 3.2, 12, aspect=0.8),
                _fl("disc", -2, 4.4, 16, aspect=0.7)]),
    dict(name="Tilted planar flaw",
         desc="8 mm lack-of-fusion-like disc at 30 mm. Increase the tilt: the specular lobe "
              "leaves the probe and the echo falls by tens of dB (Kirchhoff directivity).",
         values=dict(material="steel", mode="contact", thickness=50, f0=5, bw=60, diam=10,
                     gain=34, xprobe=0, span=60),
         flaws=[_fl("disc", 0, 30, 8, tilt=15)]),
]


class UltrasonicPage(Workspace):
    title = "Ultrasonic testing"
    HEAVY_KEYS = ("cscan", "beam")
    run_label = "Run C-scan + beam"
    SCENARIOS = SCENARIOS
    CHOICES = [
        C("material", "Material", MATS, "steel"),
        C("mode", "Coupling", [("Contact", "contact"), ("Immersion", "immersion")], "contact"),
        C("dac", "DAC curve", [("Off", 0), ("From Ø3 mm SDH", 1)], 0),
        C("saft", "B-scan", [("Envelope", 0), ("SAFT", 1)], 0),
    ]
    PARAMS = [
        P("thickness", "Thickness (mm)", 2, 200, 50, 1e-3, decimals=1),
        P("f0", "Frequency (MHz)", 0.5, 20, 5, 1e6, log=True, decimals=2),
        P("bw", "Bandwidth (%)", 20, 120, 60, 1e-2, decimals=0),
        P("diam", "Crystal Ø (mm)", 3, 30, 10, 1e-3, decimals=1),
        P("wp", "Water path (mm)", 3, 100, 20, 1e-3, decimals=1),
        P("gain", "Gain (dB)", -10, 80, 20, 1, decimals=1),
        P("noise", "Electronic noise (dB)", -120, -40, -90, 1, decimals=0),
        P("grain", "Grain noise ×", 0, 5, 1, 1, decimals=2),
        P("xprobe", "Probe x (mm)", -60, 60, 0, 1e-3, decimals=1),
        P("span", "B-scan length (mm)", 10, 150, 60, 1e-3, decimals=0),
        P("csize", "C-scan size (mm)", 10, 80, 40, 1e-3, decimals=0, heavy=True),
    ]

    DEFECT_NOTE = "Select a defect to edit it; changes update every view."
    DEFECTS = [
        C("kind", "Type", [("Flat-bottom hole", "fbh"), ("Side-drilled hole", "sdh"),
                           ("Planar crack / lack of fusion", "disc"), ("Pore (spherical void)", "pore"),
                           ("Inclusion (sphere)", "incl")], "fbh"),
        C("fill", "Inclusion material", [("Tungsten", "tungsten"), ("Copper", "copper"),
                                         ("Lead", "lead"), ("Epoxy / slag-like", "epoxy")], "tungsten"),
        P("x", "x (mm)", -75, 75, 0, 1e-3, decimals=1),
        P("y", "y (mm)", -40, 40, 0, 1e-3, decimals=1),
        P("z", "Depth (mm)", 0.3, 200, 25, 1e-3, decimals=1),
        P("size", "Diameter (mm)", 0.2, 40, 3, 1e-3, decimals=2),
        P("tilt", "Tilt (°)", -70, 70, 0, 1, decimals=1),
        P("aspect", "Length / diameter", 0.2, 5, 1, 1, decimals=2),
        P("length", "Hole length (mm)", 2, 100, 30, 1e-3, decimals=0),
    ]
    _KIND = {"fbh": "FBH", "sdh": "SDH", "disc": "crack", "pore": "pore", "incl": "inclusion"}

    def describe_defect(self, d):
        extra = f", tilt {d['tilt']:g}°" if d["kind"] == "disc" and d["tilt"] else ""
        return (f"{self._KIND[d['kind']]} Ø{d['size']:g} mm at x {d['x']:g}, y {d['y']:g}, "
                f"depth {d['z']:g} mm{extra}")

    def defect_hidden(self, d):
        k = d["kind"]
        h = set()
        if k != "disc":
            h |= {"tilt", "aspect"}
        if k != "sdh":
            h.add("length")
        if k != "incl":
            h.add("fill")
        return h

    def scenario_defects(self, sc):
        out = []
        for f in sc.get("flaws", []):
            kind = f.kind if f.kind in ("fbh", "sdh", "disc") else ("incl" if f.fill else "pore")
            out.append(dict(kind=kind, fill=f.fill or "tungsten", x=f.x * 1e3, y=f.y * 1e3,
                            z=f.z * 1e3, size=f.size * 1e3, tilt=f.tilt, aspect=f.aspect,
                            length=f.length * 1e3))
        return out

    # ------------------------------------------------------------------ views
    def build_views(self):
        self.w_a = W.plot("A-scan", "depth (mm)", "amplitude (% FSH)", legend=True)
        self.w_b = W.ImagePanel("B-scan (dB)", "x (mm)", "depth (mm)", cmap="inferno")
        self.w_c = pg.GraphicsLayoutWidget()
        self.c_amp = W.ImagePanel("C-scan amplitude (dB)", "x (mm)", "y (mm)", invert_y=False)
        self.c_tof = W.ImagePanel("C-scan depth (mm)", "x (mm)", "y (mm)", cmap="viridis",
                                  invert_y=False)
        from PyQt5 import QtWidgets
        cw = QtWidgets.QWidget(); h = QtWidgets.QVBoxLayout(cw); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.c_amp); h.addWidget(self.c_tof)
        self.w_c = cw
        bw = QtWidgets.QWidget(); bv = QtWidgets.QVBoxLayout(bw); bv.setContentsMargins(0, 0, 0, 0)
        self.w_beam = W.ImagePanel("Transducer field |p/p₀| (CW)", "x (mm)", "z (mm)",
                                   cmap="viridis")
        self.w_axis = W.plot("On-axis pressure", "z (mm)", "|p/p₀|")
        self.w_axis.setMaximumHeight(220)
        bv.addWidget(self.w_beam, 3); bv.addWidget(self.w_axis, 1)
        self.w_dgs = W.plot("DGS diagram (centre frequency)", "distance (mm)", "dB re back wall at N",
                            legend=True)
        self.w_dgs.getPlotItem().setLogMode(x=True, y=False)
        self.w_if = W.plot("Energy transmission vs incidence angle", "incidence angle (°)",
                           "energy fraction", legend=True)
        return [("A-scan", self.w_a), ("B-scan", self.w_b), ("C-scan", self.w_c),
                ("Beam", bw), ("DGS", self.w_dgs), ("Interface", self.w_if)]

    VIEW_HELP = {
        "A-scan": "Received RF signal (grey) and its envelope (orange) at the probe position, "
                  "plotted against depth = c·t/2. From left: front-surface / main bang, defect "
                  "echoes, back-wall echo and its repeats. Red dashed lines mark the gate; purple "
                  "curves are the DAC (0, −6, −14 dB) when enabled. Hover for depth and amplitude.",
        "B-scan": "Cross-section along x at y = 0: echo envelope in dB (0 dB = strongest) "
                  "versus probe position and depth. Point reflectors draw hyperbolas; planar "
                  "flaws draw flat lines. Circles number the true defect positions; the dashed "
                  "line is the A-scan probe position.",
        "C-scan": "Plan view after pressing Run: left, peak gated amplitude (dB) at every probe "
                  "position; right, depth of that peak, shown where amplitude > −14 dB. Circles "
                  "number the true defects. The 6 dB-drop length sizes large flaws.",
        "Beam": "Continuous-wave pressure field of the transducer in the part (run to compute) "
                "and the on-axis pressure. The last maximum is the near-field length N, the "
                "natural focus where sensitivity is highest.",
        "DGS": "Distance–gain–size diagram: back-wall and flat-bottom-hole echo levels versus "
               "distance, relative to a back wall at N. The red dot is defect 1's measured echo — "
               "the FBH curve through it gives its equivalent reflector size (ERS).",
        "Interface": "Energy reflected and transmitted (longitudinal / shear) versus incidence "
                     "angle at the coupling interface; vertical lines are the critical angles.",
    }

    def equations(self):
        return [
            W.eq("Pulse-echo patch model",
                 r"$S(\omega)=P(\omega)\sum_q \mathcal{R}_q\,\dfrac{|\hat n_q\cdot\hat r_q|\,\Delta A_q}{\pi a^2}\,\hat p_q^{\,2}\,E_q(\omega)$",
                 r"$E_q(\omega)=e^{-2\alpha(\omega)R_q}\,e^{-2i\omega R_q/c(\omega)}$",
                 r"$\hat p=\left(1-e^{-ik(\sqrt{R^2+a^2}-R)}\right)\dfrac{2J_1(ka\sin\theta)}{ka\sin\theta}$",
                 "Kirchhoff scattering from surface patches, reciprocity for transmit/receive."),
            W.eq("Back-wall echo (Seki diffraction)",
                 r"$D(s)=1-e^{-ika^2/s}\left[J_0\!\left(\dfrac{ka^2}{s}\right)+iJ_1\!\left(\dfrac{ka^2}{s}\right)\right],\quad s=2T$"),
            W.eq("Near field and beam spread",
                 r"$N=\dfrac{D^2-\lambda^2}{4\lambda},\qquad \sin\gamma_{-6\,\mathrm{dB}}=0.51\,\dfrac{\lambda}{D}$"),
            W.eq("Causal attenuation (Szabo)",
                 r"$\alpha(f)=\alpha_1\,(f/1\,\mathrm{MHz})^{n}$",
                 r"$\dfrac{1}{c(\omega)}=\dfrac{1}{c_0}+\alpha_0\tan\!\left(\dfrac{n\pi}{2}\right)\left(\omega^{n-1}-\omega_0^{n-1}\right)$"),
            W.eq("DGS / equivalent reflector size",
                 r"$\dfrac{p_{\mathrm{FBH}}}{p_0}=p_{ax}(z)^2\left(\dfrac{b}{a}\right)^2\;\rightarrow\;"
                 r"\dfrac{\pi^2a^2b^2}{\lambda^2z^2}\ \ (z\gg N)$"),
            W.eq("Grain noise",
                 r"$\sigma_{\mathrm{grain}}\propto\left(\dfrac{d_g}{\lambda}\right)^{3/2},\qquad "
                 r"\mathrm{SNR}=20\log_{10}\dfrac{A_{\mathrm{defect}}}{\mathrm{rms}_{\mathrm{noise}}}$",
                 "A defect counts as detected when its SNR ≥ 6 dB."),
        ]

    # ------------------------------------------------------------------ compute
    def setup(self, p):
        f0 = p["f0"]
        fs = 100e6 if f0 <= 8e6 else 250e6
        noise = 10 ** (p["noise"] / 20)
        return ut.UTSetup(material=p["material"], thickness=p["thickness"],
                          probe=ut.Probe(f0, p["bw"], p["diam"]), mode=p["mode"],
                          water_path=p["wp"], fs=fs, noise=noise, grain_scale=p["grain"],
                          grain_noise=p["grain"] > 0)

    def flaws(self, p):
        out = []
        for d in p.get("_defects", []):
            kind = {"pore": "sphere", "incl": "sphere"}.get(d["kind"], d["kind"])
            f = F(kind, x=d["x"], y=d["y"], z=d["z"], size=d["size"],
                  tilt=d["tilt"] if kind == "disc" else 0.0, aspect=d["aspect"],
                  length=d["length"], fill=d["fill"] if d["kind"] == "incl" else None,
                  label=str(len(out) + 1))
            if 0 < f.z < p["thickness"]:
                out.append(f)
        return out

    def measure_defects(self, s, fl):
        """Probe placed over each defect: echo amplitude, depth, SNR and ERS."""
        if not fl:
            return []
        t, tr = ut.pulse_echo(s, fl, [[f.x, f.y] for f in fl])
        t0, ref = ut.pulse_echo(s, [], [[0.0, 0.0]])
        z = ut.depth_axis(s, t)
        lam = s.lam()
        env_ref = sg.envelope(ref[0])
        bwin = np.abs(z - s.thickness) < 2 * lam
        bw = float(env_ref[bwin].max()) if bwin.any() else np.nan
        res = []
        for f, a in zip(fl, tr):
            env = sg.envelope(a)
            ext = f.size / 2 * abs(np.sin(np.radians(f.tilt))) + (f.size / 2 if f.kind == "sdh" else 0)
            win = np.abs(z - f.z) <= ext + 3 * lam
            quiet = (z > 0.05 * s.thickness) & (z < s.thickness - 3 * lam) & (np.abs(z - f.z) > ext + 6 * lam)
            for g in fl:
                quiet &= np.abs(z - g.z) > g.size / 2 + 6 * lam
            amp = float(env[win].max()) if win.any() else 0.0
            zpk = float(z[win][np.argmax(env[win])]) if win.any() else np.nan
            noise = float(np.sqrt(np.mean(env[quiet] ** 2))) if quiet.sum() > 10 else \
                float(np.median(env[(z > 0) & (z < s.thickness)]) + 1e-12)
            snr = 20 * np.log10(max(amp, 1e-15) / max(noise, 1e-15))
            ratio = amp / bw if np.isfinite(bw) and bw > 0 else np.nan
            try:
                ers = float(ut.ers_from_echo(s, max(zpk, 1e-4), ratio)) if np.isfinite(ratio) else np.nan
            except Exception:
                ers = np.nan
            res.append(dict(amp=amp, z=zpk, snr=snr, db_bw=20 * np.log10(max(ratio, 1e-15)),
                            ers=ers, detected=snr >= 6.0))
        return res

    def compute(self, p, heavy, ctx):
        s = self.setup(p)
        fl = self.flaws(p)
        out = {"setup": s, "flaws": fl}
        if not heavy:
            out["meas"] = self.measure_defects(s, fl)
            t, a = ut.ascan(s, fl, x=p["xprobe"])
            out.update(t=t, a=a)
            ctx.check()
            xs = np.linspace(-p["span"] / 2, p["span"] / 2, 121)
            tb, B = ut.bscan(s, fl, xs)
            out.update(bx=xs, bt=tb, B=B)
            if p["saft"]:
                z_img = np.linspace(0.5e-3, s.thickness * 1.05, 220)
                tdel = s.t0()
                if s.mode == "immersion":
                    # treat the part only (water path removed), approximate refraction-free
                    pass
                lam = s.mat.cL / s.probe.f0
                ap = np.degrees(ut.beam_spread(s.probe.diameter, s.probe.f0, s.mat.cL, 12))
                out["saft"] = (z_img, ut.saft(tb, B, xs, s.mat.cL, tdel, xs, z_img,
                                                 aperture_deg=min(ap, 40)))
            ctx.check()
            if p["dac"]:
                depths = np.linspace(0.1, 0.9, 7) * s.thickness
                out["dac"] = ut.dac_curve(s, depths, 3e-3)
            z = np.geomspace(max(0.05 * ut.near_field_length(s.probe.diameter, s.probe.f0, s.mat.cL),
                                 0.5e-3), 2.5 * max(s.thickness, 1e-3) + 0.1, 300)
            out["dgs"] = (z, ut.dgs_curves(s, z, (1e-3, 2e-3, 3e-3, 5e-3, 8e-3)))
            ang = np.linspace(0, 89.5, 360)
            if s.mode == "immersion":
                out["iface"] = (ang, ut.fluid_solid(np.radians(ang), "water", s.material),
                                ut.critical_angles("water", s.material))
            else:
                out["iface"] = (ang, ut.solid_solid(np.radians(ang), "rexolite", s.material),
                                ut.critical_angles("rexolite", s.material))
            k = 2 * np.pi * s.probe.f0 / s.mat.cL
            za = np.linspace(1e-4, 3 * ut.near_field_length(s.probe.diameter, s.probe.f0, s.mat.cL)
                             + 1e-3, 400)
            out["axis"] = (za, ut.on_axis_pressure(za, s.probe.diameter / 2, k))
            return out
        # heavy: C-scan and beam field
        L = p["csize"]
        n = 44
        xs = np.linspace(-L / 2, L / 2, n) + (fl[0].x if fl else 0)
        ys = np.linspace(-L / 2, L / 2, n)
        s2 = replace(s, fs=min(s.fs, 50e6 if s.probe.f0 <= 6e6 else 100e6))
        ctx.progress(1, 4)
        out["cscan"] = ut.cscan(s2, fl, xs, ys, max_patches=300)
        out["cscan"]["xs"], out["cscan"]["ys"] = xs, ys
        out["cscan"].pop("traces"); out["cscan"].pop("env")
        ctx.progress(3, 4)
        N = ut.near_field_length(s.probe.diameter, s.probe.f0, s.mat.cL)
        zmax = max(2.2 * N, 1.2 * s.thickness, 5e-3)
        xb = np.linspace(-1.6 * s.probe.diameter, 1.6 * s.probe.diameter, 121)
        zb = np.linspace(zmax / 200, zmax, 160)
        out["beam"] = (xb, zb, ut.piston_field_cw(xb, zb, s.probe.diameter / 2, s.probe.f0,
                                                   s.mat.cL, n_r=20, n_phi=40))
        ctx.progress(4, 4)
        return out

    # ------------------------------------------------------------------ display
    def show_result(self, r):
        s = r["setup"]
        p = self.params.values()
        g = 10 ** (p["gain"] / 20)
        t = r["t"]
        z = ut.depth_axis(s, t) * 1e3
        pi = self.w_a.getPlotItem(); pi.clear()
        if pi.legend is not None:
            pi.legend.clear()
        a = np.clip(100 * g * r["a"], -110, 110)
        W.curve(pi, z, a, "signal", 1.0, name="RF")
        W.curve(pi, z, np.clip(100 * g * sg.envelope(r["a"]), 0, 110), "envelope", 1.4,
                name="envelope")
        g0, g1 = ut.default_gate(s)
        W.vline(pi, ut.depth_axis(s, np.array([g0]))[0] * 1e3, "gate", label="gate start", pos=0.97)
        W.vline(pi, ut.depth_axis(s, np.array([g1]))[0] * 1e3, "gate", label="gate end", pos=0.97)
        if "dac" in r:
            d, amps = r["dac"]
            for lvl, sty in ((0, None), (-6, "dash"), (-14, "dot")):
                W.curve(pi, d * 1e3, 100 * g * amps * 10 ** (lvl / 20), "dac", 1.4, style=sty,
                        name=f"DAC {lvl} dB" if lvl else "DAC")
        pi.setYRange(-110, 110)
        pi.setXRange(max(z.min(), -0.12 * s.thickness * 1e3), z.max())
        # B-scan
        bz = ut.depth_axis(s, r["bt"])
        sel = (bz > -0.15 * s.thickness) & (bz < 1.25 * s.thickness)
        env = sg.envelope(r["B"])[:, sel]
        if "saft" in r:
            z_img, img = r["saft"]
            self.w_b.set(sg.db(img, floor=-50).T, r["bx"] * 1e3, z_img * 1e3,
                         levels=(-50, 0), title="B-scan, SAFT (dB)")
        else:
            inside = bz[sel] > 0.03 * s.thickness            # 0 dB = strongest echo inside the part
            ref = env[:, inside].max() if inside.any() else env.max()
            self.w_b.set(sg.db(env, ref=ref, floor=-50).T, r["bx"] * 1e3, bz[sel] * 1e3,
                         levels=(-50, 0), title="B-scan envelope (dB re strongest echo in the part)")
        self.w_b.clear_overlays()
        line = pg.InfiniteLine(pos=p["xprobe"] * 1e3, angle=90, pen=theme.pen("gate", 1, "dash"))
        self.w_b.add(line)
        self._mark(self.w_b, [(i + 1, f.x, f.z) for i, f in enumerate(r["flaws"]) if abs(f.y) < 5e-3])
        # C-scan
        if "cscan" in r:
            c = r["cscan"]
            self.c_amp.set(sg.db(c["amp"], floor=-40), c["xs"] * 1e3, c["ys"] * 1e3,
                           levels=(-40, 0))
            dmap = np.where(sg.db(c["amp"]) > -14, c["depth"] * 1e3, np.nan)
            self.c_tof.set(dmap, c["xs"] * 1e3, c["ys"] * 1e3, levels=(0, s.thickness * 1e3))
            for panel in (self.c_amp, self.c_tof):
                self._mark(panel, [(i + 1, f.x, f.y) for i, f in enumerate(r["flaws"])])
        # beam
        if "beam" in r:
            xb, zb, B = r["beam"]
            self.w_beam.set(B, xb * 1e3, zb * 1e3)
        pa = self.w_axis.getPlotItem(); pa.clear()
        za, pax = r["axis"]
        W.curve(pa, za * 1e3, pax, 0)
        N = ut.near_field_length(s.probe.diameter, s.probe.f0, s.mat.cL)
        W.vline(pa, N * 1e3, "gate", label="N")
        # DGS
        pd = self.w_dgs.getPlotItem(); pd.clear()
        if pd.legend is not None:
            pd.legend.clear()
        zd, curves = r["dgs"]
        for i, (k, v) in enumerate((k, v) for k, v in curves.items() if k != "N"):
            W.curve(pd, zd * 1e3, v, i, 1.6, name=k)
        pd.setYRange(-90, 10)
        ticks = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
                 if zd[0] * 1e3 <= v <= zd[-1] * 1e3]
        pd.getAxis("bottom").setTicks([[(np.log10(v), f"{v:g}") for v in ticks], []])
        if r.get("flaw_db") is not None:
            pass
        m = self._flaw_echo(r)
        if m is not None:
            zf, dbv = m
            sc = pg.ScatterPlotItem([np.log10(zf * 1e3)], [dbv], size=10,
                                    brush=pg.mkBrush(theme.color("gate")), pen=None)
            pd.addItem(sc)
        # interface
        pf = self.w_if.getPlotItem(); pf.clear()
        if pf.legend is not None:
            pf.legend.clear()
        ang, co, crit = r["iface"]
        if "EL" in co:
            W.curve(pf, ang, co["ER"], 3, name="reflected")
            W.curve(pf, ang, co["EL"], 0, name="L transmitted")
            W.curve(pf, ang, co["ES"], 1, name="S transmitted")
            pf.setTitle("water → part (energy coefficients)")
        else:
            W.curve(pf, ang, np.real(co["ERp"] + co["ERs"]), 3, name="reflected (P+S)")
            W.curve(pf, ang, np.real(co["ETp"]), 0, name="L transmitted")
            W.curve(pf, ang, np.real(co["ETs"]), 1, name="S transmitted")
            pf.setTitle("Rexolite wedge → part (Zoeppritz, energy)")
        for j, (k, v) in enumerate(crit.items()):
            if np.isfinite(v):
                l = W.vline(pf, v, "data", label=f"{k} critical {v:.1f}°")
                l.label.setPosition(0.55 - 0.1 * j)

    def _mark(self, panel, pts):
        """Numbered markers at the true defect positions (m)."""
        panel.clear_overlays() if panel is not self.w_b else None
        for num, x, y in pts:
            sc = pg.ScatterPlotItem([x * 1e3], [y * 1e3], symbol="o", size=14,
                                    pen=pg.mkPen(theme.color("ref"), width=1.5), brush=None)
            panel.add(sc)
            t = pg.TextItem(str(num), color=theme.color("ref"), anchor=(-0.3, 1.1))
            t.setPos(x * 1e3, y * 1e3)
            panel.add(t)

    def _flaw_echo(self, r, i=0):
        """Depth and DGS-scale level (dB re back wall at N) of defect i."""
        meas = r.get("meas") or []
        if i >= len(meas) or not np.isfinite(meas[i]["z"]):
            return None
        s = r["setup"]
        k = 2 * np.pi * s.probe.f0 / s.mat.cL
        N = ut.near_field_length(s.probe.diameter, s.probe.f0, s.mat.cL)
        ref = np.abs(ut.seki(2 * N, s.probe.diameter / 2, k))
        return meas[i]["z"], float(sg.db(meas[i]["amp"], ref=ref, floor=-200))

    def monitor(self, r):
        fl, meas = r["flaws"], r.get("meas") or []
        heads = ["#", "type", "true depth", "echo depth", "dB re BW", "SNR dB", "ERS Ø", "status"]
        rows, flags = [], []
        c = r.get("cscan")
        if c is not None:
            heads.insert(-1, "6 dB length")
        for i, (f, m) in enumerate(zip(fl, meas)):
            kind = {"sphere": "inclusion" if f.fill else "pore"}.get(f.kind, self._KIND.get(f.kind, f.kind))
            row = [str(i + 1), kind, f"{f.z * 1e3:.1f} mm",
                   f"{m['z'] * 1e3:.1f} mm" if np.isfinite(m["z"]) else "—",
                   f"{m['db_bw']:.1f}", f"{m['snr']:.1f}",
                   f"{m['ers'] * 1e3:.2f} mm" if np.isfinite(m["ers"]) else "—"]
            if c is not None:
                iy = int(np.argmin(np.abs(c["ys"] - f.y)))
                sel = np.abs(c["xs"] - f.x) <= f.size + 10e-3
                try:
                    L6, _, _ = ut.six_db_drop(c["xs"][sel], c["amp"][iy][sel])
                    row.append(f"{L6 * 1e3:.1f} mm")
                except Exception:
                    row.append("—")
            row.append("detected" if m["detected"] else "missed")
            rows.append(row); flags.append(m["detected"])
        return heads, rows, flags

    def summary(self, r):
        s = r["setup"]
        pr = s.probe
        m = s.mat
        N = ut.near_field_length(pr.diameter, pr.f0, m.cL)
        rows = [("wavelength λ", W.fmt(m.cL / pr.f0 * 1e3, "mm")),
                ("near field N", W.fmt(N * 1e3, "mm")),
                ("−6 dB beam half-angle", W.fmt(np.degrees(ut.beam_spread(pr.diameter, pr.f0, m.cL)), "°")),
                ("−6 dB beam Ø at N", W.fmt(1.02 * N * m.cL / (pr.f0 * pr.diameter) * 1e3, "mm")),
                ("attenuation at f₀", W.fmt(m.alpha(pr.f0) / 115.13, "dB/mm")),
                ("c_L / c_T", f"{m.cL:.0f} / {m.cT:.0f} m/s"),
                ("Z_L", W.fmt(m.ZL / 1e6, "MRayl"))]
        z = ut.depth_axis(s, r["t"])
        env = sg.envelope(r["a"])
        bw = np.abs(z - s.thickness) < 2 * s.lam()
        if bw.any():
            rows.append(("back-wall echo", W.fmt(20 * np.log10(env[bw].max() + 1e-12), "dB re p₀")))
        meas = r.get("meas") or []
        if meas:
            nd = sum(m["detected"] for m in meas)
            rows.append(("defects detected (SNR ≥ 6 dB)", f"{nd} / {len(meas)}"))
        if "cscan" in r:
            c = r["cscan"]
            contrast = c["amp"].max() / max(np.median(c["amp"]), 1e-30)
            rows.append(("C-scan peak / median", W.fmt(20 * np.log10(contrast), "dB")))
        crit = r["iface"][2]
        rows.append(("1st / 2nd critical angle", f"{crit['first']:.1f}° / {crit['second']:.1f}°"))
        return rows
