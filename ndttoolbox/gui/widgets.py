"""Shared GUI building blocks: background runner, parameter sliders, cards, plots, images."""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import theme


# =========================================================================== background jobs
class Cancelled(Exception):
    pass


class JobContext:
    def __init__(self, runner, gen):
        self.gen, self.cancelled, self._r = gen, False, runner

    def check(self):
        if self.cancelled:
            raise Cancelled

    def progress(self, i, n):
        self.check()
        self._r.progressed.emit(self.gen, int(i), int(n))


class _JobThread(QtCore.QThread):
    def __init__(self, fn, ctx, runner):
        super().__init__()
        self.fn, self.ctx, self.runner = fn, ctx, runner

    def run(self):
        try:
            out = self.fn(self.ctx)
            if not self.ctx.cancelled:
                self._emit("done", self.ctx.gen, out)
        except Cancelled:
            pass
        except Exception as e:
            if not self.ctx.cancelled:
                self._emit("failed", self.ctx.gen,
                           f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _emit(self, name, *args):
        try:
            getattr(self.runner, name).emit(*args)
        except RuntimeError:        # the window was closed while the job was running
            pass


class Runner(QtCore.QObject):
    """One job at a time in a QThread. A new submission cancels the previous one
    (cooperatively, at its next check/progress) and results carry a generation number, so
    stale results are never shown."""
    done = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)
    progressed = QtCore.pyqtSignal(int, int, int)
    busy = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gen = 0
        self.ctx = None
        self.threads = set()

    def submit(self, fn):
        if self.ctx is not None:
            self.ctx.cancelled = True
        self.gen += 1
        self.ctx = JobContext(self, self.gen)
        t = _JobThread(fn, self.ctx, self)
        self.threads.add(t)
        t.finished.connect(self._finished)
        t.start()
        self.busy.emit(True)
        return self.gen

    def cancel(self):
        if self.ctx is not None:
            self.ctx.cancelled = True

    def _finished(self):
        t = self.sender()
        self.threads.discard(t)
        if t is not None:
            t.deleteLater()
        if not self.running():
            self.busy.emit(False)

    def running(self):
        return any(t.isRunning() for t in self.threads)

    def is_current(self, gen):
        return gen == self.gen

    def wait_idle(self, ms=600_000):
        app = QtWidgets.QApplication.instance()
        timer = QtCore.QElapsedTimer(); timer.start()
        while (self.running() or self.threads) and timer.elapsed() < ms:
            app.processEvents()
            QtCore.QThread.msleep(5)
        app.processEvents()


# =========================================================================== small widgets
def card(title=None, subtitle=None):
    f = QtWidgets.QFrame(); f.setObjectName("card")
    v = QtWidgets.QVBoxLayout(f); v.setContentsMargins(12, 10, 12, 12); v.setSpacing(6)
    if title:
        lab = QtWidgets.QLabel(title); lab.setObjectName("cardTitle"); v.addWidget(lab)
    if subtitle:
        s = QtWidgets.QLabel(subtitle); s.setObjectName("muted"); s.setWordWrap(True)
        v.addWidget(s)
    return f, v


def muted(text):
    lab = QtWidgets.QLabel(text); lab.setObjectName("muted"); lab.setWordWrap(True)
    return lab


# =========================================================================== typeset equations
_MATH_CACHE = {}


def render_math(text, color, size=11.0, dpr=1.0):
    """Typeset a line of text/TeX (``$...$`` segments, matplotlib mathtext with Computer
    Modern fonts) to a QPixmap. Returns None if matplotlib is unavailable."""
    key = (text, color, round(size, 2), round(dpr, 2))
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    try:
        import io
        from matplotlib import mathtext, rc_context
        from matplotlib.font_manager import FontProperties
    except Exception:
        return None
    buf = io.BytesIO()
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        with rc_context({"mathtext.fontset": "cm"}):
            prop = FontProperties(size=size)
            w, h, d, _, _ = mathtext.MathTextParser("path").parse(text, dpi=72, prop=prop)
            fig = Figure(figsize=(w / 72 + 0.04, h / 72 + 0.04))
            FigureCanvasAgg(fig)
            fig.text(0.02 / (w / 72 + 0.04), (d + 1.4) / (h + 2.9), text, fontproperties=prop,
                     color=color)
            fig.savefig(buf, dpi=96 * dpr, format="png", transparent=True)
    except Exception:
        return None
    img = QtGui.QImage.fromData(buf.getvalue())
    pm = QtGui.QPixmap.fromImage(img)
    pm.setDevicePixelRatio(dpr)
    if len(_MATH_CACHE) > 2000:
        _MATH_CACHE.clear()
    _MATH_CACHE[key] = pm
    return pm


def eq(title, *lines):
    """One section of a model panel: a heading and lines; a line containing ``$`` is
    typeset as mathematics, other lines are explanatory text."""
    return (title, list(lines))


class _MathLabel(QtWidgets.QLabel):
    """A typeset formula, scaled down (never up) to the available width."""

    def __init__(self, pm, parent=None):
        super().__init__(parent)
        self._pm = pm
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self._apply(10_000)

    def _apply(self, width):
        pm = self._pm
        src = pm.devicePixelRatio()
        logical_w = pm.width() / src
        target = min(logical_w, width) if width > 40 else logical_w
        dpr = max(self.devicePixelRatioF(), 1.0)
        sc = pm.scaledToWidth(max(1, int(round(target * dpr))), Qt.SmoothTransformation)
        sc.setDevicePixelRatio(dpr)
        self.setPixmap(sc)
        self.setFixedHeight(int(round(sc.height() / dpr)) + 2)

    def resizeEvent(self, e):
        self._apply(e.size().width())
        super().resizeEvent(e)


class EquationView(QtWidgets.QFrame):
    """Model panel: headed sections of typeset equations (matplotlib mathtext, rendered in
    the application) with short explanations. Accepts a list of ``eq(...)`` sections; a
    plain string is shown as rich text."""

    def __init__(self, spec=None):
        super().__init__()
        self.setObjectName("equation")
        self.v = QtWidgets.QVBoxLayout(self)
        self.v.setContentsMargins(10, 8, 10, 10)
        self.v.setSpacing(3)
        self._spec = None
        self._pending = spec or []

    def _clear(self):
        while self.v.count():
            it = self.v.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()

    def set_html(self, spec):
        """Show `spec`; typesetting is deferred until the panel is first shown."""
        self._pending = spec
        if self.isVisible():
            self._render()

    def showEvent(self, e):
        super().showEvent(e)
        self._render()

    def _render(self):
        spec = self._pending
        key = (repr(spec), theme.current.get("name"))
        if key == self._spec:
            return
        self._spec = key
        self._clear()
        if isinstance(spec, str):
            lab = QtWidgets.QLabel(spec); lab.setWordWrap(True); lab.setTextFormat(Qt.RichText)
            self.v.addWidget(lab)
            return
        col = theme.current["eq"]
        dpr = 2.0
        app = QtWidgets.QApplication.instance()
        if app is not None and app.primaryScreen() is not None:
            dpr = max(2.0, app.primaryScreen().devicePixelRatio() * 1.5)
        for i, (title, lines) in enumerate(spec):
            if title:
                h = QtWidgets.QLabel(title); h.setObjectName("eqTitle")
                if i:
                    self.v.addSpacing(6)
                self.v.addWidget(h)
            for line in lines:
                if "$" in line:
                    pm = render_math(line, col, 11.5, dpr)
                    if pm is not None:
                        self.v.addWidget(_MathLabel(pm))
                        continue
                    lab = QtWidgets.QLabel(line.replace("$", ""))
                    lab.setWordWrap(True)
                else:
                    lab = QtWidgets.QLabel(line); lab.setObjectName("eqNote"); lab.setWordWrap(True)
                self.v.addWidget(lab)


@dataclass
class P:
    """Numeric parameter: shown in display units (value × scale = SI)."""
    key: str
    label: str
    lo: float
    hi: float
    default: float
    scale: float = 1.0
    log: bool | None = None
    integer: bool = False
    decimals: int = 3
    tip: str = ""
    heavy: bool = False          # changing it only triggers the heavy run, not live updates


@dataclass
class C:
    key: str
    label: str
    options: list               # [(text, value)]
    default: object = None
    tip: str = ""


class ParamRow(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    N = 1000

    def __init__(self, p: P, parent=None):
        super().__init__(parent)
        self.p = p
        self.log = p.log if p.log is not None else (p.lo > 0 and p.hi / p.lo > 60)
        self.label = QtWidgets.QLabel(p.label)
        self.label.setWordWrap(True)
        if p.tip:
            self.label.setToolTip(p.tip)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.N); self.slider.setMinimumWidth(80)
        if p.integer:
            self.spin = QtWidgets.QSpinBox(); self.spin.setRange(int(p.lo), int(p.hi))
        else:
            self.spin = QtWidgets.QDoubleSpinBox(); self.spin.setRange(p.lo, p.hi)
            self.spin.setDecimals(p.decimals)
            span = p.hi - p.lo
            self.spin.setSingleStep(10 ** math.floor(math.log10(span / 50)) if span > 0 else 0.1)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(84)
        self.spin.setValue(p.default)
        self._sync()
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_pos(self, v):
        p = self.p
        if self.log:
            u = (math.log(max(v, p.lo)) - math.log(p.lo)) / (math.log(p.hi) - math.log(p.lo))
        else:
            u = (v - p.lo) / (p.hi - p.lo) if p.hi > p.lo else 0
        return int(round(min(max(u, 0.0), 1.0) * self.N))

    def _from_pos(self, pos):
        p = self.p
        u = pos / self.N
        if self.log:
            v = math.exp(math.log(p.lo) + u * (math.log(p.hi) - math.log(p.lo)))
        else:
            v = p.lo + u * (p.hi - p.lo)
        if p.integer:
            return int(round(v))
        return float(f"{v:.4g}")

    def _sync(self):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(self.spin.value()))
        self.slider.blockSignals(False)

    def _from_slider(self, pos):
        self.spin.blockSignals(True); self.spin.setValue(self._from_pos(pos))
        self.spin.blockSignals(False)
        self.changed.emit()

    def _from_spin(self, _):
        self._sync(); self.changed.emit()

    def value(self):
        return self.spin.value()

    def set_value(self, v):
        self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)
        self._sync()


class ParamPanel(QtWidgets.QWidget):
    """Grid of parameter rows and choice boxes; emits changed(key)."""
    changed = QtCore.pyqtSignal(str)

    def __init__(self, params, choices=(), parent=None):
        super().__init__(parent)
        g = QtWidgets.QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0); g.setHorizontalSpacing(8); g.setVerticalSpacing(5)
        self.rows, self.boxes, self.specs, self.box_labels = {}, {}, {}, {}
        self.choice_defaults = {c.key: c.default for c in choices if c.default is not None}
        r = 0
        for c in choices:
            lab = QtWidgets.QLabel(c.label)
            lab.setWordWrap(True)
            cb = QtWidgets.QComboBox()
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
            for text, val in c.options:
                cb.addItem(text, val)
            if c.default is not None:
                cb.setCurrentIndex(max(0, cb.findData(c.default)))
            if c.tip:
                cb.setToolTip(c.tip)
            cb.currentIndexChanged.connect(lambda _, k=c.key: self.changed.emit(k))
            g.addWidget(lab, r, 0); g.addWidget(cb, r, 1, 1, 2)
            self.boxes[c.key] = cb
            self.box_labels[c.key] = lab
            r += 1
        for p in params:
            row = ParamRow(p, self)
            row.changed.connect(lambda k=p.key: self.changed.emit(k))
            g.addWidget(row.label, r, 0); g.addWidget(row.slider, r, 1); g.addWidget(row.spin, r, 2)
            self.rows[p.key] = row
            self.specs[p.key] = p
            r += 1
        g.setColumnStretch(1, 1)

    def values(self):
        out = {k: row.value() * row.p.scale for k, row in self.rows.items()}
        out.update({k: cb.currentData() for k, cb in self.boxes.items()})
        return out

    def display_values(self):
        out = {k: row.value() for k, row in self.rows.items()}
        out.update({k: cb.currentData() for k, cb in self.boxes.items()})
        return out

    def set_values(self, d):
        for k, v in d.items():
            if k in self.rows:
                self.rows[k].set_value(v)
            elif k in self.boxes:
                cb = self.boxes[k]
                cb.blockSignals(True); cb.setCurrentIndex(max(0, cb.findData(v)))
                cb.blockSignals(False)

    def set_hidden(self, keys):
        """Hide the controls that do not apply to the current configuration."""
        keys = set(keys or ())
        for k, row in self.rows.items():
            for w in (row.label, row.slider, row.spin):
                w.setVisible(k not in keys)
        for k, cb in self.boxes.items():
            cb.setVisible(k not in keys); self.box_labels[k].setVisible(k not in keys)

    def defaults(self):
        out = {k: r.p.default for k, r in self.rows.items()}
        out.update(self.choice_defaults)
        return out


class ResultTable(QtWidgets.QTableWidget):
    def __init__(self):
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels(["quantity", "value"])
        hh = self.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.ElideMiddle)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setMinimumHeight(140)

    def set_rows(self, rows):
        self.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            a, b = QtWidgets.QTableWidgetItem(str(k)), QtWidgets.QTableWidgetItem(str(v))
            a.setToolTip(f"{k}: {v}"); b.setToolTip(f"{k}: {v}")
            self.setItem(i, 0, a)
            self.setItem(i, 1, b)

    def rows(self):
        return [(self.item(i, 0).text(), self.item(i, 1).text()) for i in range(self.rowCount())]


# =========================================================================== plots
# =========================================================================== hover read-out
def _axis_name(pi, ax):
    lab = pi.getAxis(ax).labelText or ""
    return lab.strip() or ("x" if ax == "bottom" else "y")


def _fmtv(v):
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.3e}"
    return f"{v:.4g}"


class HoverReadout(QtCore.QObject):
    """Cross-hair and a small read-out next to the cursor: the axis values and, for images,
    the pixel value; for line plots, each named curve's value at the cursor."""

    def __init__(self, pi, panel=None):
        super().__init__(pi)
        self.pi, self.panel = pi, panel
        self.text = pg.TextItem(anchor=(0, 0))
        self.text.setZValue(10_000)
        self.vl = pg.InfiniteLine(angle=90, movable=False)
        self.hl = pg.InfiniteLine(angle=0, movable=False)
        for it in (self.vl, self.hl):
            it.setZValue(9_000)
        self._hide()
        self._connected = False
        QtCore.QTimer.singleShot(0, self._connect)

    def _connect(self):
        sc = self.pi.scene()
        if sc is None or self._connected:
            return
        sc.sigMouseMoved.connect(self._moved)
        self._connected = True

    def _ensure(self):
        t = theme.current
        self.text.fill = pg.mkBrush(QtGui.QColor(t["panel"]).lighter(100).name() + "e6")
        self.text.border = pg.mkPen(t["border"])
        self.text.setColor(t["text"])
        pen = pg.mkPen(t["muted"], width=1, style=Qt.DotLine)
        self.vl.setPen(pen); self.hl.setPen(pen)
        for it in (self.text, self.vl, self.hl):
            if it.scene() is None:
                self.pi.addItem(it, ignoreBounds=True)

    def _hide(self):
        for it in (self.text, self.vl, self.hl):
            it.setVisible(False)

    def _moved(self, pos):
        vb = self.pi.vb
        if not vb.sceneBoundingRect().contains(pos):
            self._hide()
            return
        self._ensure()
        pt = vb.mapSceneToView(pos)
        vx, vy = pt.x(), pt.y()
        lx = self.pi.ctrl.logXCheck.isChecked()
        ly = self.pi.ctrl.logYCheck.isChecked()
        x = 10 ** vx if lx else vx
        y = 10 ** vy if ly else vy
        lines = [f"{_axis_name(self.pi, 'bottom')}: {_fmtv(x)}"]
        if self.panel is not None and self.panel.data is not None:
            lines.append(f"{_axis_name(self.pi, 'left')}: {_fmtv(y)}")
            ip = self.panel.img.mapFromScene(pos)
            c, r = int(np.floor(ip.x())), int(np.floor(ip.y()))
            d = self.panel.data
            if 0 <= r < d.shape[0] and 0 <= c < d.shape[1]:
                lines.append(f"{self.panel.value_name}: {_fmtv(float(d[r, c]))}")
            self.hl.setVisible(True); self.hl.setPos(vy)
        else:
            n = 0
            unnamed = False
            for item in self.pi.listDataItems():
                if not item.isVisible():
                    continue
                name = item.name() if hasattr(item, "name") else None
                if not name:
                    if unnamed:
                        continue
                    unnamed = True
                    name = _axis_name(self.pi, "left")
                xd, yd = item.getData()
                if xd is None or len(xd) == 0:
                    continue
                k = int(np.nanargmin(np.abs(np.asarray(xd, float) - vx)))
                yv = float(yd[k])
                lines.append(f"{name}: {_fmtv(10 ** yv if ly else yv)}")
                n += 1
                if n >= 6:
                    break
            if n == 0:
                lines.append(f"{_axis_name(self.pi, 'left')}: {_fmtv(y)}")
                self.hl.setVisible(True); self.hl.setPos(vy)
            else:
                self.hl.setVisible(False)
        self.vl.setVisible(True); self.vl.setPos(vx)
        self.text.setText("\n".join(lines))
        # keep the read-out inside the plot: flip to the left/top near the edges
        r = vb.sceneBoundingRect()
        ax = 1 if pos.x() > r.center().x() + 0.2 * r.width() else 0
        ay = 1 if pos.y() > r.center().y() + 0.2 * r.height() else 0
        self.text.setAnchor((ax, ay))
        off = QtCore.QPointF(-12 if ax else 12, -12 if ay else 12)
        p2 = vb.mapSceneToView(pos + off)
        self.text.setPos(p2.x(), p2.y())
        self.text.setVisible(True)


def plot(title=None, x=None, y=None, invert_y=False, legend=False, aspect=False):
    w = pg.PlotWidget()
    pi = w.getPlotItem()
    style(pi, title, x, y)
    if invert_y:
        pi.invertY(True)
    if aspect:
        pi.setAspectLocked(True)
    if legend:
        pi.addLegend(offset=(-8, 8), labelTextSize="8pt")
        legend_style(pi)
    w.hover = HoverReadout(pi)
    return w


def style(pi, title=None, x=None, y=None):
    t = theme.current
    if title:
        pi.setTitle(f"<span style='font-size:9pt;color:{t['plot_fg']}'>{title}</span>")
    if x:
        pi.setLabel("bottom", x)
    if y:
        pi.setLabel("left", y)
    pi.showGrid(x=True, y=True, alpha=t["grid"] / 255)
    for ax in ("left", "bottom"):
        pi.getAxis(ax).enableAutoSIPrefix(False)
        pi.getAxis(ax).setPen(pg.mkPen(t["plot_fg"]))
        pi.getAxis(ax).setTextPen(pg.mkPen(t["plot_fg"]))


class ImagePanel(pg.GraphicsLayoutWidget):
    """Image with physical axes and a colour bar."""

    def __init__(self, title=None, x="x (mm)", y="z (mm)", cmap="inferno", invert_y=True,
                 bipolar=False):
        super().__init__()
        self.pi = self.addPlot(row=0, col=0)
        style(self.pi, title, x, y)
        self.img = pg.ImageItem()
        self.pi.addItem(self.img)
        if invert_y:
            self.pi.invertY(True)
        self.pi.setAspectLocked(True)
        self.cmap_name = "CET-D1" if bipolar else cmap
        self.bipolar = bipolar
        self.bar = pg.ColorBarItem(colorMap=theme.cmap(self.cmap_name), width=12,
                                   interactive=False)
        self.bar.setImageItem(self.img, insert_in=self.pi)
        self.overlays = []
        self.data = None
        self.value_name = (title or "value")
        self.hover = HoverReadout(self.pi, self)

    def set(self, data, x, y, levels=None, title=None):
        data = np.asarray(data, float)
        self.data = data
        if title:
            self.value_name = title
        if levels is None:
            if self.bipolar:
                m = np.nanmax(np.abs(data)) or 1.0
                levels = (-m, m)
            else:
                lo, hi = np.nanmin(data), np.nanmax(data)
                levels = (lo, hi if hi > lo else lo + 1)
        self.img.setImage(data, autoLevels=False, levels=levels)
        dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
        dy = (y[-1] - y[0]) / max(len(y) - 1, 1)
        self.img.setRect(QtCore.QRectF(x[0] - dx / 2, y[0] - dy / 2, x[-1] - x[0] + dx,
                                       y[-1] - y[0] + dy))
        self.bar.setLevels(levels)
        if title:
            style(self.pi, title)

    def clear_overlays(self):
        for it in self.overlays:
            self.pi.removeItem(it)
        self.overlays = []

    def add(self, item):
        self.pi.addItem(item); self.overlays.append(item)
        return item

    def set_cmap(self, name):
        self.cmap_name = name
        self.bar.setColorMap(theme.cmap(name))


def legend_corner(pi, corner="top-right"):
    """Move a plot's legend to another corner of the view."""
    lg = getattr(pi, "legend", None)
    if lg is None:
        return
    ax = 1 if "right" in corner else 0
    ay = 1 if "bottom" in corner else 0
    lg.anchor((ax, ay), (ax, ay), offset=(-8 if ax else 8, -8 if ay else 8))


def legend_style(pi):
    """Opaque, bordered legend in the current theme, drawn above the data."""
    lg = getattr(pi, "legend", None)
    if lg is None:
        return
    t = theme.current
    lg.setBrush(pg.mkBrush(QtGui.QColor(t["plot_bg"]).name() + "e0"))
    lg.setPen(pg.mkPen(t["border"]))
    lg.setLabelTextColor(t["plot_fg"])
    lg.setZValue(20_000)


def curve(pi, x, y, c=0, width=1.6, name=None, style=None, symbol=None):
    kw = {}
    if symbol:
        kw = dict(symbol=symbol, symbolSize=6, symbolBrush=theme.color(c), symbolPen=None)
    if name:
        legend_style(pi)
    return pi.plot(x, y, pen=None if style == "none" else theme.pen(c, width, style), name=name,
                   **kw)


class _SideLabel(pg.InfLineLabel):
    """Line label that stays inside the plot: a vertical line's label sits on the side of
    the line with more room; a horizontal line's label sits above the line at the left."""

    def updatePosition(self):
        super().updatePosition()
        vb = self.line.getViewBox()
        if vb is None:
            return
        r = vb.viewRect()
        if self.line.angle % 180 == 0:
            self.setAnchor((0, 1))
            return
        side = getattr(self, "side", "auto")
        if side == "auto":
            frac = (self.line.value() - r.left()) / r.width() if r.width() else 0.5
            side = "left" if frac > 0.55 else "right"
        self.setAnchor((1.08, 0.5) if side == "left" else (-0.08, 0.5))


def _line(pi, pos, angle, c, style, label, where, side="auto"):
    l = pg.InfiniteLine(pos=pos, angle=angle, pen=theme.pen(c, 1.2, style))
    if label:
        l.label = _SideLabel(l, text=label, position=where, color=theme.color(c))
        l.label.side = side
        l.label.updatePosition()
    pi.addItem(l)
    return l


def vline(pi, x, c="gate", style="dash", label=None, pos=0.92, side="auto"):
    """Vertical marker; the label goes on the side with more room unless `side` is
    'left' or 'right'."""
    return _line(pi, x, 90, c, style, label, pos, side)


def hline(pi, y, c="gate", style="dash", label=None):
    return _line(pi, y, 0, c, style, label, 0.02)


def text(pi, x, y, s, c=None, anchor=(0, 1)):
    t = pg.TextItem(s, color=c or theme.current["plot_fg"], anchor=anchor)
    t.setPos(x, y)
    pi.addItem(t)
    return t


def grab_png(widget, path):
    pm = widget.grab()
    pm.save(path, "PNG")


def fmt(v, unit="", digits=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v} {unit}".strip()
    return f"{v:.{digits}g} {unit}".strip()


def log_ticks(pi, axis, lo, hi):
    """Clean 1-2-5 tick labels on a log-mode axis spanning [lo, hi] (data units)."""
    vals = [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
            for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    major = [(np.log10(v), f"{v:g}") for v in vals if abs(np.log10(v) % 1) < 1e-9]
    minor = [(np.log10(v), f"{v:g}") for v in vals if abs(np.log10(v) % 1) >= 1e-9]
    if len(major) >= 4:
        minor = [(t, "") for t, _ in minor]
    pi.getAxis(axis).setTicks([major, minor])


# =========================================================================== user defects
class DefectEditor(QtWidgets.QWidget):
    """List of user-editable defects. The selected defect is edited with the same sliders
    and boxes as the parameters (fields = P/C specs). Values are stored in display units;
    `get()` returns SI dictionaries. `describe(d)` gives the list text, `hidden(d)` the
    fields that do not apply to a defect of that type, `template(d)` a new defect."""
    changed = QtCore.pyqtSignal()
    selected = QtCore.pyqtSignal(int)

    def __init__(self, fields, describe, hidden=None, max_items=12, parent=None):
        super().__init__(parent)
        self.fields = fields
        self.describe = describe
        self.hidden = hidden or (lambda d: ())
        self.max_items = max_items
        self.items = []
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(6)
        self.list = QtWidgets.QListWidget(); self.list.setObjectName("defectList")
        self.list.setMaximumHeight(118)
        self.list.currentRowChanged.connect(self._select)
        v.addWidget(self.list)
        row = QtWidgets.QHBoxLayout(); row.setSpacing(6)
        self.b_add = QtWidgets.QPushButton("Add"); self.b_dup = QtWidgets.QPushButton("Duplicate")
        self.b_del = QtWidgets.QPushButton("Remove")
        self.b_add.setToolTip("Add a new defect (a copy of the defaults)")
        self.b_dup.setToolTip("Copy the selected defect")
        self.b_del.setToolTip("Delete the selected defect")
        for b in (self.b_add, self.b_dup, self.b_del):
            row.addWidget(b)
        v.addLayout(row)
        self.b_add.clicked.connect(self.add)
        self.b_dup.clicked.connect(self.duplicate)
        self.b_del.clicked.connect(self.remove)
        self.form = ParamPanel([f for f in fields if isinstance(f, P)],
                               [f for f in fields if isinstance(f, C)])
        self.form.changed.connect(self._edited)
        v.addWidget(self.form)
        self.empty = muted("No defects — the part is sound. Press Add to place one.")
        v.addWidget(self.empty)
        self._sync_buttons()

    def defaults(self):
        return self.form.defaults()

    def set(self, defects):
        """Replace all defects (display-unit dictionaries; missing keys take defaults)."""
        self.items = [dict(self.defaults(), **d) for d in defects][: self.max_items]
        self._rebuild(0 if self.items else -1)

    def get(self):
        out = []
        for d in self.items:
            o = {}
            for f in self.fields:
                o[f.key] = d[f.key] * f.scale if isinstance(f, P) else d[f.key]
            out.append(o)
        return out

    def current(self):
        return self.list.currentRow()

    def add(self):
        if len(self.items) >= self.max_items:
            return
        self.items.append(self.defaults())
        self._rebuild(len(self.items) - 1)
        self.changed.emit()

    def duplicate(self):
        i = self.current()
        if i < 0 or len(self.items) >= self.max_items:
            return
        self.items.insert(i + 1, dict(self.items[i]))
        self._rebuild(i + 1)
        self.changed.emit()

    def remove(self):
        i = self.current()
        if i < 0:
            return
        self.items.pop(i)
        self._rebuild(min(i, len(self.items) - 1))
        self.changed.emit()

    def _label(self, i, d):
        return f"{i + 1}.  {self.describe(d)}"

    def _rebuild(self, sel):
        self.list.blockSignals(True)
        self.list.clear()
        for i, d in enumerate(self.items):
            self.list.addItem(self._label(i, d))
        self.list.blockSignals(False)
        if sel >= 0:
            self.list.setCurrentRow(sel)
            self._select(sel)
        self._sync_buttons()

    def _sync_buttons(self):
        has = bool(self.items)
        self.form.setVisible(has)
        self.empty.setVisible(not has)
        self.b_dup.setEnabled(has and len(self.items) < self.max_items)
        self.b_del.setEnabled(has)
        self.b_add.setEnabled(len(self.items) < self.max_items)

    def _select(self, i):
        if 0 <= i < len(self.items):
            self.form.blockSignals(True)
            self.form.set_values(self.items[i])
            self.form.blockSignals(False)
            self.form.set_hidden(self.hidden(self.items[i]))
            self.selected.emit(i)

    def _edited(self, key):
        i = self.current()
        if i < 0:
            return
        d = self.items[i]
        d.update(self.form.display_values())
        self.list.item(i).setText(self._label(i, d))
        self.form.set_hidden(self.hidden(d))
        self.changed.emit()


class MonitorTable(QtWidgets.QTableWidget):
    """Per-defect monitoring table (one row per defect, page-defined columns)."""

    def __init__(self):
        super().__init__(0, 1)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(120)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

    def set_data(self, headers, rows, flags=None):
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setRowCount(len(rows))
        bad = QtGui.QColor(theme.color("miss"))
        good = QtGui.QColor(theme.color("hit"))
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                it = QtWidgets.QTableWidgetItem(str(v))
                it.setToolTip(f"{headers[j]}: {v}")
                if flags is not None and j == len(row) - 1 and flags[i] is not None:
                    it.setForeground(good if flags[i] else bad)
                self.setItem(i, j, it)
        self.resizeColumnsToContents()
        n = min(max(len(rows), 1), 8)
        rh = self.verticalHeader().defaultSectionSize()
        hh = self.horizontalHeader().sizeHint().height()
        sb = self.horizontalScrollBar().sizeHint().height() \
            if self.horizontalHeader().length() > self.viewport().width() else 0
        self.setFixedHeight(hh + n * rh + sb + 2 * self.frameWidth() + 2)

    def rows(self):
        return [[self.item(i, j).text() for j in range(self.columnCount())] for i in range(self.rowCount())]


# ---------------------------------------------------------------------- zoom limits
def plot_items(widget):
    """All plot areas (PlotItems, colour bars excluded) inside a view widget."""
    views = [widget] if isinstance(widget, pg.GraphicsView) else []
    views += widget.findChildren(pg.GraphicsView)
    out = []
    for v in views:
        sc = v.scene()
        if sc is None:
            continue
        for it in sc.items():
            if isinstance(it, pg.PlotItem) and not isinstance(it, pg.ColorBarItem) \
                    and it not in out:
                out.append(it)
    return out


def release_zoom_limits(pi):
    lim = pi.vb.state["limits"]
    lim["xLimits"] = [-1e307, 1e307]
    lim["yLimits"] = [-1e307, 1e307]
    lim["xRange"] = [None, None]
    lim["yRange"] = [None, None]


def _bounds(vb):
    """Extent of the data in view coordinates, plus the range the page chose."""
    b = vb.childrenBounds()
    (x0, x1), (y0, y1) = vb.viewRange()
    xs = [v for v in ((b[0] if b[0] is not None else (None, None))) if v is not None and np.isfinite(v)]
    ys = [v for v in ((b[1] if b[1] is not None else (None, None))) if v is not None and np.isfinite(v)]
    return xs, ys, (x0, x1), (y0, y1)


def apply_zoom_limits(pi, pad=0.02):
    """Never zoom out past the full extent of the data (with a small margin) or past the
    range the page displays; zooming in is unrestricted. Aspect-locked views (images) are
    widened on one axis to the widget's shape so the whole image still fits."""
    vb = pi.vb
    xs, ys, (vx0, vx1), (vy0, vy1) = _bounds(vb)
    if len(xs) < 2 or len(ys) < 2:
        return
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or abs(x0) or 1.0
    dy = (y1 - y0) or abs(y0) or 1.0
    x0, x1 = x0 - pad * dx, x1 + pad * dx
    y0, y1 = y0 - pad * dy, y1 + pad * dy
    ratio = vb.state.get("aspectLocked")
    # the page may set a wider fixed window on an axis (auto-range off): allow that too
    ar = vb.state.get("autoRange", [True, True])
    if not ratio and not ar[0]:
        x0, x1 = min(x0, vx0), max(x1, vx1)
    if not ratio and not ar[1]:
        y0, y1 = min(y0, vy0), max(y1, vy1)
    if ratio:
        w, h = max(vb.width(), 1.0), max(vb.height(), 1.0)
        # locked: (Δy/h) = ratio·(Δx/w)  → widen the axis that is too short
        need_dy = ratio * (x1 - x0) * h / w
        if need_dy > (y1 - y0):
            c = 0.5 * (y0 + y1); y0, y1 = c - need_dy / 2, c + need_dy / 2
        else:
            need_dx = (y1 - y0) * w / (h * ratio)
            c = 0.5 * (x0 + x1); x0, x1 = c - need_dx / 2, c + need_dx / 2
        if not getattr(vb, "_zoom_resize_hooked", False):
            vb._zoom_resize_hooked = True
            vb.sigResized.connect(lambda *_: (release_zoom_limits(pi), apply_zoom_limits(pi)))
    vb.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1,
                 maxXRange=x1 - x0, maxYRange=y1 - y0)
