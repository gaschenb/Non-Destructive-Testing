"""Probability of detection: â-vs-a (censored regression), hit/miss GLM, model-assisted POD."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..core import pod
from ..core import ultrasonics as ut
from . import theme
from . import widgets as W
from .widgets import C, P
from .workspace import Workspace

SCENARIOS = [
    dict(name="Signal response (â vs a)",
         desc="60 cracks, log-linear response with scatter τ; censored regression, Wald/delta-method "
              "95 % lower bound on POD and a₉₀/₉₅.",
         values=dict(source="ahat", n=60, b0=4.0, b1=1.1, tau=0.35, thr=60, floor=0, sat=0,
                     amin=0.3, amax=6, seed=1, target=1.1)),
    dict(name="Censored data (floor and saturation)",
         desc="Recording floor at 25 and saturation at 400: censored points are handled by the "
              "Tobit likelihood instead of being dropped.",
         values=dict(source="ahat", n=60, b0=4.0, b1=1.1, tau=0.35, thr=60, floor=25, sat=400,
                     amin=0.3, amax=6, seed=1, target=1.1)),
    dict(name="Small sample, wide bounds",
         desc="Only 20 specimens: a₉₀ barely changes but a₉₀/₉₅ grows — bounds, not means, "
              "drive qualification.",
         values=dict(source="ahat", n=20, b0=4.0, b1=1.1, tau=0.35, thr=60, floor=0, sat=0,
                     amin=0.3, amax=6, seed=3, target=1.1)),
    dict(name="Hit/miss (logit)",
         desc="120 hit/miss calls (e.g. FPI or visual). Logistic GLM in ln a with likelihood-ratio "
              "confidence bounds (MIL-HDBK-1823A).",
         values=dict(source="hitmiss", n=120, a50=1.2, slope=4.0, link="logit", fcall=0.0,
                     mfloor=0.0, amin=0.2, amax=5, seed=0, target=2.0)),
    dict(name="Hit/miss with false calls",
         desc="5 % false-call rate and a 3 % miss floor: the 2-parameter model is biased — "
              "compare the fitted curve with the truth.",
         values=dict(source="hitmiss", n=150, a50=1.0, slope=3.5, link="logit", fcall=0.05,
                     mfloor=0.03, amin=0.2, amax=5, seed=2, target=2.0)),
    dict(name="Model-assisted POD — UT cracks",
         desc="Kirchhoff simulation of 5 MHz contact UT on 40 mm steel: disc cracks with random "
              "tilt (σ = 12°), depth and position; noise from flaw-free simulations.",
         values=dict(source="mapod_ut", n=4, tilt=12, thr=0.0, amin=0.08, amax=2, seed=0,
                     target=0.2, nsizes=12)),
    dict(name="Model-assisted POD — eddy-current cracks",
         desc="Pancake probe at 100 kHz on aluminium: semi-circular surface cracks with random "
              "lift-off (σ = 50 µm) and orientation (heavy: runs in the background).",
         values=dict(source="mapod_ec", n=3, lsd=50, thr=0.0, amin=0.3, amax=4, seed=0,
                     target=0.35, nsizes=10)),
]


class PODPage(Workspace):
    title = "Probability of detection"
    SCENARIOS = SCENARIOS
    HEAVY_KEYS = ("full",)
    auto_heavy = True
    CHOICES = [
        C("source", "Data", [("Synthetic â vs a", "ahat"), ("Synthetic hit/miss", "hitmiss"),
                             ("Model-assisted: UT", "mapod_ut"), ("Model-assisted: eddy current", "mapod_ec")],
          "ahat"),
        C("link", "Hit/miss link", [("Logit", "logit"), ("Probit", "probit"),
                                    ("Complementary log-log", "cloglog"), ("Log-log", "loglog")], "logit"),
    ]
    PARAMS = [
        P("n", "Specimens (per size for model-assisted)", 2, 400, 60, 1, integer=True),
        P("nsizes", "Model-assisted sizes", 4, 30, 12, 1, integer=True),
        P("amin", "Smallest size (mm)", 0.05, 5, 0.3, 1e-3, log=True, decimals=2),
        P("amax", "Largest size (mm)", 0.5, 30, 6, 1e-3, log=True, decimals=2),
        P("b0", "β₀ (ln â at a = 1 mm)", 0, 8, 4, 1, decimals=2),
        P("b1", "β₁ (slope)", 0.2, 3, 1.1, 1, decimals=2),
        P("tau", "τ (scatter, ln units)", 0.05, 1.5, 0.35, 1, decimals=2),
        P("thr", "Decision threshold â_dec (0 = auto)", 0, 1000, 60, 1, decimals=4),
        P("floor", "Recording floor (0 = none)", 0, 200, 0, 1, decimals=1),
        P("sat", "Saturation (0 = none)", 0, 5000, 0, 1, decimals=0),
        P("a50", "True a₅₀ (mm)", 0.1, 5, 1.2, 1e-3, decimals=2),
        P("slope", "True slope (per ln a)", 0.5, 15, 4, 1, decimals=2),
        P("fcall", "False-call rate", 0, 0.3, 0, 1, decimals=3),
        P("mfloor", "Miss floor", 0, 0.3, 0, 1, decimals=3),
        P("tilt", "Crack tilt σ (°)", 0, 40, 12, 1, decimals=1),
        P("lsd", "Lift-off σ (µm)", 0, 200, 50, 1e-6, decimals=0),
        P("target", "Target size for ROC (mm)", 0.1, 10, 2, 1e-3, decimals=2),
        P("seed", "Random seed", 0, 999, 0, 1, integer=True),
    ]

    def build_views(self):
        self.w_ahat = W.plot("Signal response", "a (mm)", "â", legend=True)
        self.w_ahat.getPlotItem().setLogMode(x=True, y=True)
        self.w_pod = W.plot("Probability of detection", "a (mm)", "POD", legend=True)
        self.w_pod.getPlotItem().setLogMode(x=True, y=False)
        self.w_roc = W.plot("Noise and signal at the target size", "ln â", "density", legend=True)
        self.w_roc2 = W.plot("ROC at the target size", "P(false call)", "POD(target)", legend=True)
        W.legend_corner(self.w_roc2.getPlotItem(), "bottom-right")
        self.w_cr = W.plot("Parameter confidence region", "β₀ / μ", "β₁ / σ", legend=True)
        return [("â vs a", self.w_ahat), ("POD", self.w_pod), ("Noise vs signal", self.w_roc),
                ("ROC", self.w_roc2), ("Confidence region", self.w_cr)]

    VIEW_HELP = {
        "â vs a": "Simulated inspection data: signal response â against defect size a on log "
                  "axes, with the fitted line, the decision threshold â_dec and censoring "
                  "limits. Hover for values.",
        "POD": "Probability of detection versus defect size with its 95 % lower confidence "
               "bound; a₉₀ and a₉₀/₉₅ are marked.",
        "Noise vs signal": "Distribution of the noise and of the signal from defects at the "
                           "target size; the threshold divides misses from hits and false calls.",
        "ROC": "Receiver operating characteristic at the target size: POD against false-call "
               "probability as the threshold moves.",
        "Confidence region": "Likelihood-ratio confidence region of the fitted parameters; the "
                             "lower POD bound is taken over this region.",
    }

    def equations(self):
        return [
            W.eq("Signal response (â vs a)",
                 r"$\ln\hat{a}=\beta_0+\beta_1\ln a+\varepsilon,\qquad\varepsilon\sim N(0,\tau^2)$",
                 "Censored (Tobit) likelihood for responses below the floor or above saturation.",
                 r"$\mathrm{POD}(a)=\Phi\!\left(\dfrac{\ln a-\mu}{\sigma}\right),\qquad"
                 r"\mu=\dfrac{\ln\hat{a}_{dec}-\beta_0}{\beta_1},\qquad\sigma=\dfrac{\tau}{\beta_1}$"),
            W.eq("Lower confidence bound",
                 r"$x_{p,95}=\mu+z_p\sigma+1.645\sqrt{V_{\mu\mu}+z_p^2V_{\sigma\sigma}+2z_pV_{\mu\sigma}}$"),
            W.eq("Hit / miss",
                 r"$g(\mathrm{POD})=\beta_0+\beta_1\ln a$",
                 r"$g\in\{\mathrm{logit},\mathrm{probit},\mathrm{cloglog},\mathrm{loglog}\}$",
                 r"$2\left(\hat{\ell}-\ell(\beta)\right)\leq\chi^2_{1,0.90}=2.706$"),
            W.eq("False calls",
                 r"$P_{FA}=P\left(\mathrm{noise}>\hat{a}_{dec}\right)$",
                 "The ROC is traced by varying the decision threshold."),
        ]

    def hidden_keys(self, p):
        src = p["source"]
        groups = dict(ahat={"b0", "b1", "tau", "floor", "sat"}, hitmiss={"a50", "slope", "fcall", "mfloor", "link"},
                      mapod_ut={"tilt", "nsizes"}, mapod_ec={"lsd", "nsizes"})
        shown = set(groups[src]) | {"n", "amin", "amax", "seed"}
        if src != "hitmiss":
            shown |= {"thr", "target"}
        if src.startswith("mapod"):
            shown |= {"sat"}
        every = set().union(*groups.values()) | {"thr", "target"}
        return every - shown

    def compute(self, p, heavy, ctx):
        src = p["source"]
        if src == "mapod_ec" and not heavy:
            return dict(pending=True)
        seed = int(p["seed"])
        a_rng = (p["amin"], max(p["amax"], 1.2 * p["amin"]))
        out = dict(source=src)
        if src == "hitmiss":
            a, h = pod.synthetic_hitmiss(int(p["n"]), a_rng, p["a50"], p["slope"], seed, p["link"],
                                         p["fcall"], p["mfloor"])
            res = pod.hit_miss(a, h, p["link"])
            out.update(a=a, hits=h, res=res, kind="hm")
            return dict(full=out)
        noise = None
        floor = p["floor"] or None
        sat = p["sat"] or None
        if src == "ahat":
            a, ah = pod.synthetic_ahat(int(p["n"]), a_rng, p["b0"], p["b1"], p["tau"], floor, sat, seed)
            rng = np.random.default_rng(seed + 7)
            noise = np.exp(np.log(15.0) + 0.45 * rng.standard_normal(200))
            thr = p["thr"] or float(np.exp(np.mean(np.log(noise)) + 3 * np.std(np.log(noise))))
        else:
            sizes = np.geomspace(a_rng[0], a_rng[1], int(p["nsizes"]))
            if src == "mapod_ut":
                s = ut.UTSetup(thickness=40e-3, multiples=1)
                a, ah, noise = pod.mapod_ultrasonic(sizes, s, n_per=int(p["n"]), seed=seed,
                                                    tilt_sd=p["tilt"])
            else:
                a, ah, noise = pod.mapod_eddy(sizes, n_per=int(p["n"]), seed=seed,
                                              liftoff_sd=p["lsd"])
            ctx.check()
            # decision threshold: user value, else noise mean + 3σ (log domain)
            ln = np.log(noise)
            thr = p["thr"] or float(np.exp(ln.mean() + 3 * ln.std(ddof=1)))
            floor = None
        res = pod.ahat_vs_a(a, ah, thr, floor=floor, saturation=sat)
        nm, ns = pod.noise_fit(noise)
        out.update(a=a, ahat=ah, res=res, kind="ahat", thr=thr, noise=noise, nm=nm, ns=ns,
                   floor=floor, sat=sat)
        if src == "ahat":
            out["truth"] = dict(b0=p["b0"] - p["b1"] * np.log(1e-3), b1=p["b1"], tau=p["tau"])
        return dict(full=out)

    def _clear(self, w):
        pi = w.getPlotItem(); pi.clear()
        if pi.legend is not None:
            pi.legend.clear()
        return pi

    def show_result(self, r):
        p = self.params.values()
        pis = [self._clear(w) for w in (self.w_ahat, self.w_pod, self.w_roc, self.w_roc2, self.w_cr)]
        if "full" not in r:
            if r.get("pending"):
                W.text(pis[1], 0, 0.5, "running the eddy-current simulations…")
            return
        o = r["full"]
        res = o["res"]
        a_mm = o["a"] * 1e3
        aa = np.geomspace(min(o["a"].min(), 0.5 * res.a50), max(o["a"].max(), 2 * res.a90_95), 300)
        pa, pp, pr, pc, pcr = pis
        lx = lambda v: np.log10(v)
        if o["kind"] == "ahat":
            ah = o["ahat"]
            low = np.zeros(len(ah), bool) if not o["floor"] else ah <= o["floor"]
            high = np.zeros(len(ah), bool) if not o["sat"] else ah >= o["sat"]
            ok = ~(low | high)
            pa.plot(a_mm[ok], ah[ok], pen=None, symbol="o", symbolSize=6,
                    symbolBrush=theme.color("data"), symbolPen=None, name="data")
            if (low | high).any():
                pa.plot(a_mm[low | high], ah[low | high], pen=None, symbol="t", symbolSize=8,
                        symbolBrush=theme.color("miss"), symbolPen=None, name="censored")
            fit = np.exp(res.beta0 + res.beta1 * np.log(aa))
            W.curve(pa, aa * 1e3, fit, "fit", 2.2, name="fit ln â = β₀ + β₁ ln a")
            for s in (-1.645, 1.645):
                W.curve(pa, aa * 1e3, fit * np.exp(s * res.tau), "fit", 1.0, style="dash")
            W.hline(pa, lx(o["thr"]), "gate", "dash", label=f"â_dec = {o['thr']:.3g}")
            for lim, nm in ((o["floor"], "floor"), (o["sat"], "saturation")):
                if lim:
                    W.hline(pa, lx(lim), "data", "dot", label=nm)
            yy = np.concatenate([ah, o["noise"] if o.get("noise") is not None else ah])
            W.log_ticks(pa, "left", max(yy.min(), 1e-12), yy.max())
            if o.get("noise") is not None:
                nz = o["noise"]
                xs = np.full(len(nz), aa[0] * 1e3 * 0.8)
                pa.plot(xs, nz, pen=None, symbol="x", symbolSize=5, symbolBrush=theme.color("ref"),
                        symbolPen=theme.color("ref"), name="noise")
        else:
            jitter = np.random.default_rng(0).uniform(-0.03, 0.03, len(a_mm))
            hit = o["hits"] > 0
            pp.plot(a_mm[hit], 1 + jitter[hit], pen=None, symbol="o", symbolSize=5,
                    symbolBrush=theme.color("hit"), symbolPen=None, name="hit")
            pp.plot(a_mm[~hit], 0 + jitter[~hit], pen=None, symbol="o", symbolSize=5,
                    symbolBrush=theme.color("miss"), symbolPen=None, name="miss")
            W.text(pa, 0, 0, "Hit/miss data has no signal amplitude — see the POD tab.")
        W.curve(pp, aa * 1e3, res.pod(aa), "fit", 2.4, name="POD")
        W.curve(pp, aa * 1e3, res.pod_lower(aa), "bound", 1.8, name="95 % lower bound", style="dash")
        if o["kind"] == "hm":
            W.curve(pp, aa * 1e3, res.pod_upper(aa), "bound", 1.0, name="upper bound", style="dot")
            F = pod.LINKS[p["link"]][0]
            truth = p["fcall"] + (1 - p["fcall"] - p["mfloor"]) * F(p["slope"] * (np.log(aa) - np.log(p["a50"])))
            W.curve(pp, aa * 1e3, truth, "data", 1.4, name="true POD", style="dash")
        elif "truth" in o:
            t = o["truth"]
            mu = (np.log(o["thr"]) - t["b0"]) / t["b1"]
            from scipy.stats import norm
            W.curve(pp, aa * 1e3, norm.cdf((np.log(aa) - mu) / (t["tau"] / t["b1"])), "data", 1.4,
                    name="true POD", style="dash")
        W.hline(pp, 0.9, "data", "dot")
        for v, lab, ps, sd in ((res.a90, "a₉₀", 0.55, "left"), (res.a90_95, "a₉₀/₉₅", 0.4, "right")):
            if np.isfinite(v) and v > 0:
                W.vline(pp, lx(v * 1e3), "gate" if lab != "a₉₀" else "fit", "dash",
                        label=f"{lab} = {v * 1e3:.3g} mm", pos=ps, side=sd)
        pp.setYRange(-0.08, 1.08)
        W.log_ticks(pp, "bottom", aa[0] * 1e3, aa[-1] * 1e3)
        W.log_ticks(pa, "bottom", aa[0] * 1e3 * 0.8, aa[-1] * 1e3)
        # noise vs signal and ROC
        if o["kind"] == "ahat":
            at = p["target"]
            mu_s = res.beta0 + res.beta1 * np.log(at)
            x = np.linspace(min(o["nm"] - 4 * o["ns"], mu_s - 4 * res.tau),
                            max(o["nm"] + 4 * o["ns"], mu_s + 4 * res.tau), 400)
            from scipy.stats import norm
            W.curve(pr, x, norm.pdf(x, o["nm"], o["ns"]), "ref", 2, name="noise")
            W.curve(pr, x, norm.pdf(x, mu_s, res.tau), "fit", 2, name=f"signal, a = {at * 1e3:.3g} mm")
            hist, edges = np.histogram(np.log(o["noise"]), bins=20, density=True)
            pr.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=hist,
                                       brush=pg.mkBrush(theme.color("ref") + "55"), pen=None))
            W.vline(pr, np.log(o["thr"]), "gate", "dash", label="â_dec")
            ths = np.exp(np.linspace(x[0], x[-1], 300))
            pfa, pd = pod.roc_curve(res, at, o["nm"], o["ns"], ths)
            W.curve(pc, pfa, pd, "fit", 2.2, name=f"a = {at * 1e3:.3g} mm")
            pfa0 = pod.false_call_probability(o["thr"], o["nm"], o["ns"])
            pc.plot([pfa0], [float(res.pod(at))], pen=None, symbol="o", symbolSize=10,
                    symbolBrush=theme.color("gate"), name="operating point")
            W.curve(pc, [0, 1], [0, 1], "data", 1, style="dot")
            # (μ, σ) confidence ellipse
            V = res.cov_ms
            ev, evec = np.linalg.eigh(V)
            th = np.linspace(0, 2 * np.pi, 200)
            circ = np.vstack([np.cos(th), np.sin(th)]) * np.sqrt(np.maximum(ev, 0))[:, None] * np.sqrt(5.991)
            el = evec @ circ
            W.curve(pcr, res.mu + el[0], res.sigma + el[1], "bound", 2, name="95 % region (μ, σ)")
            pcr.plot([res.mu], [res.sigma], pen=None, symbol="+", symbolSize=12,
                     symbolBrush=theme.color("fit"), symbolPen=theme.color("fit"), name="estimate")
            W.style(pcr, "Parameter confidence region", "μ = ln a₅₀", "σ")
        else:
            B = res.boundary
            W.curve(pcr, B[:, 0], B[:, 1], "bound", 2, name="LR region (χ²₁ = 2.706)")
            pcr.plot([res.beta[0]], [res.beta[1]], pen=None, symbol="+", symbolSize=12,
                     symbolBrush=theme.color("fit"), symbolPen=theme.color("fit"), name="MLE")
            W.style(pcr, "Parameter confidence region", "β₀", "β₁")
            W.text(pr, 0, 0, "Noise statistics need signal-response data.")

    def summary(self, r):
        if "full" not in r:
            return [("status", "simulating…")]
        o = r["full"]
        res = o["res"]
        rows = [("a₅₀", W.fmt(res.a50 * 1e3, "mm")), ("a₉₀", W.fmt(res.a90 * 1e3, "mm")),
                ("a₉₀/₉₅", W.fmt(res.a90_95 * 1e3, "mm"))]
        if o["kind"] == "ahat":
            rows += [("β₀ / β₁ / τ", f"{res.beta0:.3g} / {res.beta1:.3g} / {res.tau:.3g}"),
                     ("μ / σ (ln m)", f"{res.mu:.3f} / {res.sigma:.3f}"),
                     ("decision threshold", W.fmt(o["thr"])),
                     ("false-call probability", W.fmt(pod.false_call_probability(o["thr"], o["nm"], o["ns"]))),
                     ("censored low / high", f"{res.n_censored_low} / {res.n_censored_high}"),
                     ("observations", str(len(o["a"])))]
        else:
            rows += [("β₀ / β₁", f"{res.beta[0]:.3g} / {res.beta[1]:.3g}"),
                     ("link", res.link), ("hits / total", f"{int(o['hits'].sum())} / {len(o['hits'])}"),
                     ("log-likelihood", W.fmt(res.loglik))]
        return rows
