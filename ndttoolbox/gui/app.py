"""NDT Toolbox desktop app (PyQt5 + pyqtgraph).

    ndttoolbox-gui      or      python -m ndttoolbox.gui

Sidebar workspaces: ultrasonic testing, TOFD, phased arrays, eddy currents, radiography,
computed tomography, thermography, guided waves and probability of detection. Every workspace
has built-in example problems, live sliders (a fast pass re-runs on every change in a worker
thread; stale results are discarded) and a heavier "Run" pass for C-scans, images and bounds.
"""
from __future__ import annotations

import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from . import theme

APP_NAME = "NDT Toolbox"

PAGES = [
    ("Ultrasonic testing", "ultrasonic_page", "UltrasonicPage"),
    ("TOFD", "tofd_page", "TOFDPage"),
    ("Phased array", "array_page", "ArrayPage"),
    ("Eddy current", "eddy_page", "EddyPage"),
    ("Radiography", "radiography_page", "RadiographyPage"),
    ("Computed tomography", "ct_page", "CTPage"),
    ("Thermography", "thermography_page", "ThermographyPage"),
    ("Guided waves", "guided_page", "GuidedPage"),
    ("Probability of detection", "pod_page", "PODPage"),
]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, theme_name="light", pages=None):
        super().__init__()
        app = QtWidgets.QApplication.instance()
        theme.apply(app, theme_name)
        import importlib
        self.setWindowTitle(APP_NAME)
        self.resize(1680, 960)
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        side = QtWidgets.QWidget(); side.setFixedWidth(190)
        sv = QtWidgets.QVBoxLayout(side); sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(0)
        b = QtWidgets.QLabel(APP_NAME); b.setObjectName("brand"); sv.addWidget(b)
        s = QtWidgets.QLabel("non-destructive testing"); s.setObjectName("brandSub"); sv.addWidget(s)
        self.nav = QtWidgets.QListWidget(); self.nav.setObjectName("nav")
        self.nav.setFocusPolicy(Qt.NoFocus)
        sv.addWidget(self.nav, 1)
        h.addWidget(side)
        self.stack = QtWidgets.QStackedWidget()
        h.addWidget(self.stack, 1)
        self.pages = []
        for name, mod, cls in PAGES:
            if pages is not None and name not in pages:
                continue
            m = importlib.import_module(f".{mod}", __package__)
            p = getattr(m, cls)()
            p.status.connect(self.show_status)
            for r in (p.runner, p.heavy_runner):
                r.busy.connect(self._busy)
            self.stack.addWidget(p)
            self.pages.append(p)
            self.nav.addItem(name)
        self.nav.currentRowChanged.connect(self._page_changed)

        tb = QtWidgets.QToolBar(); tb.setMovable(False); tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        tb.addAction("▶  Run", self.run).setShortcut("Ctrl+R")
        tb.addAction("■  Cancel", self.cancel).setShortcut("Esc")
        self.a_live = tb.addAction("Live updates"); self.a_live.setCheckable(True)
        self.a_live.setChecked(True); self.a_live.setShortcut("Ctrl+L")
        self.a_live.toggled.connect(lambda on: [p.set_live(on) for p in self.pages])
        tb.addSeparator()
        tb.addAction("Export plot…", self.export_png).setShortcut("Ctrl+E")
        tb.addAction("Export results…", self.export_csv).setShortcut("Ctrl+Shift+E")
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.a_theme = tb.addAction("Dark theme", self.toggle_theme); self.a_theme.setShortcut("Ctrl+T")
        tb.addAction("Help", self.about).setShortcut("F1")
        self.busy = QtWidgets.QProgressBar(); self.busy.setRange(0, 0); self.busy.setFixedWidth(120)
        self.busy.setVisible(False)
        self.statusBar().addPermanentWidget(self.busy)
        if theme_name == "dark":
            self.a_theme.setText("Light theme")
        self.nav.setCurrentRow(0)

    def _busy(self, _):
        self.busy.setVisible(any(r.running() for p in self.pages
                                 for r in (p.runner, p.heavy_runner)))

    def page(self):
        return self.stack.currentWidget()

    def _page_changed(self, i):
        self.stack.setCurrentIndex(i)
        self.page().refresh()

    def show_status(self, msg):
        self.statusBar().showMessage(msg)

    def run(self):
        self.page().run_all()

    def cancel(self):
        self.page().cancel()

    def export_png(self):
        name = self.page().tabs.tabText(self.page().tabs.currentIndex()).lower().replace(" ", "_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", f"{name}.png",
                                                        "PNG image (*.png)")
        if path:
            self.page().export_png(path); self.show_status(f"saved {path}")

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "results.csv",
                                                        "CSV (*.csv)")
        if path:
            self.page().export_data(path); self.show_status(f"saved {path}")

    def toggle_theme(self):
        new = "dark" if theme.current["name"] == "light" else "light"
        theme.apply(QtWidgets.QApplication.instance(), new)
        self.a_theme.setText("Light theme" if new == "dark" else "Dark theme")
        for p in self.pages:
            p.invalidate()

    def about(self):
        QtWidgets.QMessageBox.information(self, APP_NAME, (
            f"{APP_NAME}\n\n"
            "Ctrl+R  run the full computation of the workspace (C-scans, images, bounds)\n"
            "Esc     cancel\n"
            "Ctrl+L  toggle live updates\n"
            "Ctrl+E  export the current plot, Ctrl+Shift+E the results table\n"
            "Ctrl+T  light / dark theme\n\n"
            "Sliders re-compute the fast views live in a background thread. Plots: drag to "
            "pan, wheel to zoom, right-click for export and axis options."))

    def closeEvent(self, e):
        for p in self.pages:
            for r in (p.runner, p.heavy_runner):
                r.cancel()
                for t in list(r.threads):
                    t.wait(5000)
        super().closeEvent(e)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    f = app.font(); f.setPointSize(9)
    if sys.platform.startswith("win"):
        f.setFamily("Segoe UI")
    app.setFont(f)
    w = MainWindow("dark" if "--dark" in argv or os.environ.get("NDTTOOLBOX_THEME") == "dark"
                   else "light")
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
