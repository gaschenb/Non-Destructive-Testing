"""Command-line interface: quick engineering calculations and data export.

    ndttoolbox gui [--dark]
    ndttoolbox materials
    ndttoolbox beam      --material steel --freq 5e6 --diameter 10e-3
    ndttoolbox interface --from water --to steel
    ndttoolbox ascan     --material steel --thickness 50e-3 --fbh 2e-3 --depth 30e-3 --out a.csv
    ndttoolbox ers       --material steel --freq 4e6 --diameter 24e-3 --thickness 100e-3 --depth 60e-3 --ratio-db -26
    ndttoolbox tofd      --thickness 25e-3 --pcs 60e-3 --times 9.5e-6 10.4e-6
    ndttoolbox eddy      --sigma-iacs 30 --freq 100e3
    ndttoolbox xray      --material steel --kev 100 200 662 1250
    ndttoolbox lamb      --material aluminium-6061 --thickness 1e-3 --fmax 5e6 --out lamb.csv
    ndttoolbox thermal   --material cfrp --tstar 0.8
    ndttoolbox pod       data.csv --threshold 60      (columns: a, ahat)   |   --hitmiss (a, hit)
All quantities are SI.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np


def _table(rows):
    w = max(len(str(r[0])) for r in rows)
    for k, v in rows:
        print(f"  {str(k):<{w}}  {v}")


def _write_csv(path, header, cols):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for row in zip(*cols):
            wr.writerow([f"{v:.8g}" if isinstance(v, (float, np.floating)) else v for v in row])
    print(f"wrote {path}")


def cmd_materials(a):
    from .core.materials import MATERIALS
    rows = [("name", "cL (m/s)   cT (m/s)   rho (kg/m3)   sigma (%IACS)   k (W/mK)")]
    for k, m in MATERIALS.items():
        iacs = f"{m.iacs:8.1f}" if m.sigma else "       —"
        rows.append((k, f"{m.cL:7.0f}   {m.cT:7.0f}   {m.rho:9.0f}   {iacs}       {m.k:7.3g}"))
    _table(rows)


def cmd_beam(a):
    from .core import ultrasonics as ut
    from .core.materials import get
    c = get(a.material).cL
    N = ut.near_field_length(a.diameter, a.freq, c)
    _table([("wavelength", f"{c / a.freq * 1e3:.3f} mm"),
            ("near-field length N", f"{N * 1e3:.2f} mm"),
            ("-6 dB half-angle", f"{np.degrees(ut.beam_spread(a.diameter, a.freq, c, 6)):.2f} deg"),
            ("-20 dB half-angle", f"{np.degrees(ut.beam_spread(a.diameter, a.freq, c, 20)):.2f} deg"),
            ("-6 dB beam width at 2N", f"{4 * N * np.tan(ut.beam_spread(a.diameter, a.freq, c, 6)) * 1e3:.2f} mm")])


def cmd_interface(a):
    from .core import ultrasonics as ut
    from .core.materials import get
    m1, m2 = get(a.from_), get(a.to)
    r = ut.normal_incidence(m1.rho * m1.cL, m2.rho * m2.cL)
    rows = [("Z1 / Z2", f"{m1.rho * m1.cL / 1e6:.3f} / {m2.rho * m2.cL / 1e6:.3f} MRayl"),
            ("R (pressure)", f"{r['R']:.4f}  ({20 * np.log10(abs(r['R']) + 1e-30):.1f} dB)"),
            ("T (pressure)", f"{r['T']:.4f}"),
            ("energy R / T", f"{r['ER']:.4f} / {r['ET']:.4f}")]
    for k, v in ut.critical_angles(a.from_, a.to).items():
        rows.append((f"critical angle {k}", f"{v:.2f} deg" if np.isfinite(v) else "none"))
    _table(rows)


def cmd_ascan(a):
    from .core import ultrasonics as ut
    from .core.signals import envelope
    s = ut.UTSetup(material=a.material, thickness=a.thickness,
                   probe=ut.Probe(f0=a.freq, diameter=a.diameter), mode=a.mode,
                   water_path=a.water_path)
    flaws = [ut.Flaw("fbh", x=0.0, z=a.depth, size=a.fbh)] if a.fbh > 0 else []
    t, tr = ut.ascan(s, flaws)
    env = envelope(tr)
    z = ut.depth_axis(s, t)
    if a.fbh > 0:
        g = np.abs(z - a.depth) < 3e-3
        bw = np.abs(z - a.thickness) < 3e-3
        _table([("flaw echo", f"{env[g].max():.4g} p/p0"),
                ("back-wall echo", f"{env[bw].max():.4g} p/p0"),
                ("flaw / back wall", f"{20 * np.log10(env[g].max() / env[bw].max()):.1f} dB")])
    if a.out:
        _write_csv(a.out, ["t_s", "depth_m", "rf", "envelope"], [t, z, tr, env])


def cmd_ers(a):
    from .core import ultrasonics as ut
    s = ut.UTSetup(material=a.material, thickness=a.thickness,
                   probe=ut.Probe(f0=a.freq, diameter=a.diameter))
    d = ut.ers_from_echo(s, a.depth, 10 ** (a.ratio_db / 20))
    _table([("equivalent FBH diameter", f"{d * 1e3:.2f} mm")])


def cmd_tofd(a):
    from .core import ultrasonics as ut
    s = ut.TOFDSetup(material=a.material, thickness=a.thickness, pcs=a.pcs)
    rows = [("lateral wave", f"{ut.tofd_time(0.0, s) * 1e6:.3f} us"),
            ("back wall", f"{ut.tofd_time(a.thickness, s) * 1e6:.3f} us")]
    t_lw = ut.tofd_time(0.0, s)
    for t in a.times:
        rows.append((f"t = {t * 1e6:.3f} us", f"depth {ut.tofd_depth(t, s) * 1e3:.2f} mm"))
    if len(a.times) == 2:
        d1, d2 = sorted(ut.tofd_depth(np.array(a.times), s))
        rows.append(("through-wall height", f"{(d2 - d1) * 1e3:.2f} mm"))
    del t_lw
    _table(rows)


def cmd_eddy(a):
    from .core import eddy_current as ec
    sigma = a.sigma if a.sigma else a.sigma_iacs / 100 * 5.8e7
    d = ec.skin_depth(a.freq, sigma, a.mur)
    _table([("conductivity", f"{sigma / 1e6:.3f} MS/m ({sigma / 5.8e7 * 100:.1f} %IACS)"),
            ("standard depth of penetration", f"{d * 1e3:.3f} mm"),
            ("effective depth (3 delta)", f"{3 * d * 1e3:.3f} mm"),
            ("phase lag at 1 delta", "57.3 deg")])


def cmd_xray(a):
    from .core import radiography as rt
    rows = []
    for e in a.kev:
        m = float(rt.mu(a.material, e))
        rows.append((f"{e:g} keV", f"mu = {m / 100:.4g} 1/cm   HVL = {np.log(2) / m * 1e3:.3f} mm   "
                                   f"TVL = {np.log(10) / m * 1e3:.3f} mm"))
    _table(rows)


def cmd_lamb(a):
    from .core import guided_waves as gw
    d = gw.lamb_dispersion(a.material, a.thickness, f_max=a.fmax, n_f=a.n)
    names = list(d.modes)
    if a.out:
        cols = [d.f] + [d.modes[k] for k in names] + [d.group[k] for k in names]
        _write_csv(a.out, ["f_Hz"] + [f"cp_{k}" for k in names] + [f"cg_{k}" for k in names], cols)
    rows = []
    for k in names:
        v = d.modes[k]
        ok = np.isfinite(v)
        rows.append((k, f"from {d.f[ok][0] * a.thickness * 1e3 / 1e6:.3f} MHz·mm" if ok.any() else "—"))
    z = gw.zgv_frequency(d, "S1")
    if np.isfinite(z):
        rows.append(("S1 ZGV", f"{z * a.thickness * 1e3 / 1e6:.3f} MHz·mm"))
    _table(rows)


def cmd_thermal(a):
    from .core import thermography as th
    from .core.materials import get
    m = get(a.material)
    alpha = (m.k_z if m.k_z is not None else m.k) / (m.rho * m.cp)
    rows = [("through-thickness diffusivity", f"{alpha * 1e6:.4g} mm²/s")]
    for t in a.tstar:
        rows.append((f"TSR t* = {t:g} s", f"depth {th.tsr_depth(alpha, t) * 1e3:.3f} mm"))
    for f in a.fblind:
        rows.append((f"PPT f_b = {f:g} Hz", f"depth {th.ppt_depth(alpha, f) * 1e3:.3f} mm"))
    _table(rows)


def cmd_pod(a):
    from .core import pod
    data = np.genfromtxt(a.csv, delimiter=",", names=True)
    cols = data.dtype.names
    x = np.asarray(data[cols[0]], float) * a.scale
    y = np.asarray(data[cols[1]], float)
    if a.hitmiss:
        r = pod.hit_miss(x, y, a.link)
        extra = [("beta0 / beta1", f"{r.beta[0]:.4g} / {r.beta[1]:.4g}")]
    else:
        r = pod.ahat_vs_a(x, y, a.threshold, floor=a.floor, saturation=a.saturation)
        extra = [("beta0 / beta1 / tau", f"{r.beta0:.4g} / {r.beta1:.4g} / {r.tau:.4g}"),
                 ("censored low / high", f"{r.n_censored_low} / {r.n_censored_high}")]
    _table([("observations", str(len(x))), ("a50", f"{r.a50:.4g} m"), ("a90", f"{r.a90:.4g} m"),
            ("a90/95", f"{r.a90_95:.4g} m")] + extra)


def build_parser():
    p = argparse.ArgumentParser(prog="ndttoolbox", description="Non-destructive testing toolbox")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="start the graphical application")
    g.add_argument("--dark", action="store_true")
    sub.add_parser("materials", help="list the material database")

    b = sub.add_parser("beam", help="near field and beam spread of a circular probe")
    b.add_argument("--material", default="steel"); b.add_argument("--freq", type=float, default=5e6)
    b.add_argument("--diameter", type=float, default=10e-3)

    i = sub.add_parser("interface", help="reflection/transmission and critical angles")
    i.add_argument("--from", dest="from_", default="water"); i.add_argument("--to", default="steel")

    s = sub.add_parser("ascan", help="simulate a pulse-echo A-scan (CSV export)")
    s.add_argument("--material", default="steel"); s.add_argument("--thickness", type=float, default=50e-3)
    s.add_argument("--freq", type=float, default=5e6); s.add_argument("--diameter", type=float, default=10e-3)
    s.add_argument("--mode", choices=["contact", "immersion"], default="contact")
    s.add_argument("--water-path", type=float, default=20e-3)
    s.add_argument("--fbh", type=float, default=2e-3); s.add_argument("--depth", type=float, default=30e-3)
    s.add_argument("--out")

    e = sub.add_parser("ers", help="equivalent reflector size (DGS) from an echo ratio")
    e.add_argument("--material", default="steel"); e.add_argument("--freq", type=float, default=4e6)
    e.add_argument("--diameter", type=float, default=24e-3)
    e.add_argument("--thickness", type=float, required=True); e.add_argument("--depth", type=float, required=True)
    e.add_argument("--ratio-db", type=float, required=True, help="flaw echo relative to back wall (dB)")

    t = sub.add_parser("tofd", help="TOFD depth from arrival times")
    t.add_argument("--material", default="steel"); t.add_argument("--thickness", type=float, default=25e-3)
    t.add_argument("--pcs", type=float, default=60e-3); t.add_argument("--times", type=float, nargs="*", default=[])

    c = sub.add_parser("eddy", help="standard depth of penetration")
    c.add_argument("--sigma", type=float, default=0.0); c.add_argument("--sigma-iacs", type=float, default=30.0)
    c.add_argument("--mur", type=float, default=1.0); c.add_argument("--freq", type=float, default=100e3)

    x = sub.add_parser("xray", help="attenuation coefficient and half-value layer")
    x.add_argument("--material", default="steel")
    x.add_argument("--kev", type=float, nargs="+", default=[100, 200, 317, 662, 1250])

    l = sub.add_parser("lamb", help="Lamb-wave dispersion curves (CSV export)")
    l.add_argument("--material", default="aluminium-6061"); l.add_argument("--thickness", type=float, default=1e-3)
    l.add_argument("--fmax", type=float, default=5e6); l.add_argument("-n", type=int, default=200)
    l.add_argument("--out")

    h = sub.add_parser("thermal", help="thermographic depth from TSR / PPT times")
    h.add_argument("--material", default="cfrp")
    h.add_argument("--tstar", type=float, nargs="*", default=[])
    h.add_argument("--fblind", type=float, nargs="*", default=[])

    q = sub.add_parser("pod", help="POD analysis of a CSV file (a, ahat) or (a, hit)")
    q.add_argument("csv"); q.add_argument("--threshold", type=float, default=None)
    q.add_argument("--floor", type=float); q.add_argument("--saturation", type=float)
    q.add_argument("--hitmiss", action="store_true")
    q.add_argument("--link", default="logit", choices=["logit", "probit", "cloglog", "loglog"])
    q.add_argument("--scale", type=float, default=1.0, help="multiply sizes by this (e.g. 1e-3 for mm)")
    return p


COMMANDS = dict(materials=cmd_materials, beam=cmd_beam, interface=cmd_interface, ascan=cmd_ascan,
                ers=cmd_ers, tofd=cmd_tofd, eddy=cmd_eddy, xray=cmd_xray, lamb=cmd_lamb,
                thermal=cmd_thermal, pod=cmd_pod)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:                    # never fail on consoles that cannot print · ² µ
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        return 0
    if a.cmd == "gui":
        from .gui.app import main as gui_main
        return gui_main(["ndttoolbox"] + (["--dark"] if a.dark else []))
    if a.cmd == "pod" and not a.hitmiss and a.threshold is None:
        p.error("pod: --threshold is required for signal-response data")
    COMMANDS[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
