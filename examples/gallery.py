"""Render the documentation figures (application screenshots) into docs/figures.

    python examples/gallery.py            # all figures
    python examples/gallery.py --list     # what would be rendered
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "figures")

# (file, page, example problem, view, theme)
FIGURES = [
    ("ut_bscan.png", "Ultrasonic testing", "Steel plate with flat-bottom holes", "B-scan", "light"),
    ("ut_cscan.png", "Ultrasonic testing", "Immersion C-scan of a titanium forging", "C-scan", "light"),
    ("ut_dgs.png", "Ultrasonic testing", "Steel plate with flat-bottom holes", "DGS", "light"),
    ("tofd_dscan.png", "TOFD", "Embedded fatigue crack", "D-scan", "light"),
    ("pa_tfm.png", "Phased array", "Side-drilled holes, contact TFM", "Image", "dark"),
    ("pa_sscan.png", "Phased array", "Wedge sectorial scan of a weld", "Image", "light"),
    ("ec_impedance.png", "Eddy current", "Surface crack, pencil probe", "Impedance plane", "light"),
    ("ec_pulsed.png", "Eddy current", "Pulsed eddy current: wall loss under insulation", "Pulsed EC", "light"),
    ("rt_weld.png", "Radiography", "Steel butt weld, 200 kV X-ray", "Radiograph", "light"),
    ("rt_spectrum.png", "Radiography", "Steel butt weld, 200 kV X-ray", "Spectrum", "light"),
    ("ct_pipe.png", "Computed tomography", "Steel pipe with corrosion pits", "Reconstruction", "light"),
    ("ct_casting.png", "Computed tomography", "Aluminium casting with porosity", "Reconstruction", "dark"),
    ("ir_contrast.png", "Thermography", "CFRP delaminations at four depths", "Image", "light"),
    ("ir_tsr.png", "Thermography", "Non-uniform flash, TSR rescues it", "Image", "light"),
    ("gw_dispersion.png", "Guided waves", "Aluminium plate dispersion", "Phase velocity", "light"),
    ("gw_zgv.png", "Guided waves", "Zero-group-velocity S1 resonance", "Group velocity", "light"),
    ("gw_screening.png", "Guided waves", "Long-range screening (SH0)", "Screening", "light"),
    ("pod_ahat.png", "Probability of detection", "Censored data (floor and saturation)", "â vs a", "light"),
    ("pod_curve.png", "Probability of detection", "Hit/miss with false calls", "POD", "light"),
]


def main(argv):
    if "--list" in argv:
        for f in FIGURES:
            print(*f, sep="  |  ")
        return 0
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["gallery"])
    from ndttoolbox.gui.app import MainWindow
    os.makedirs(OUT, exist_ok=True)
    # the colour theme is application-wide: render one theme at a time
    for theme in ("light", "dark"):
        jobs = [f for f in FIGURES if f[4] == theme]
        if not jobs:
            continue
        w = MainWindow(theme)
        w.resize(1600, 940)
        w.show()
        for fname, page_name, scenario, view, _ in jobs:
            t0 = time.time()
            idx = [p.title for p in w.pages].index(page_name)
            w.nav.setCurrentRow(idx)
            page = w.pages[idx]
            page.cb_scn.setCurrentIndex(page.cb_scn.findText(scenario))
            page.settle()
            page.run_all()
            page.settle()
            tab = [page.tabs.tabText(i) for i in range(page.tabs.count())].index(view)
            page.tabs.setCurrentIndex(tab)
            page.settle()
            for _ in range(10):
                app.processEvents()
            w.grab().save(os.path.join(OUT, fname))
            print(f"{fname:22s} {time.time() - t0:5.1f} s")
        w.close()
        app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
