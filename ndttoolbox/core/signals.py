"""Pulses, causal attenuation, envelopes and frequency-domain helpers shared by the
ultrasonic, phased-array and guided-wave models."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy.signal import hilbert

from .materials import Material


def gaussian_pulse(t, f0, bw=0.6, phase=0.0):
    """Gaussian-modulated sinusoid with centre frequency f0 [Hz] and fractional −6 dB
    bandwidth bw (full width / f0)."""
    sf = bw * f0 / (2 * np.sqrt(2 * np.log(2)))
    s = 1.0 / (2 * np.pi * sf)
    return np.exp(-t ** 2 / (2 * s ** 2)) * np.cos(2 * np.pi * f0 * t + phase)


def gaussian_spectrum(f, f0, bw=0.6):
    """Analytic (one-sided) spectrum of :func:`gaussian_pulse`, normalised to 1 at f0."""
    sf = bw * f0 / (2 * np.sqrt(2 * np.log(2)))
    f = np.asarray(f, float)
    return np.exp(-(f - f0) ** 2 / (2 * sf ** 2)) + np.exp(-(f + f0) ** 2 / (2 * sf ** 2))


def tone_burst(t, f0, cycles=5):
    """Hann-windowed tone burst starting at t = 0 (guided-wave excitation)."""
    T = cycles / f0
    w = np.where((t >= 0) & (t <= T), 0.5 * (1 - np.cos(2 * np.pi * t / T)), 0.0)
    return w * np.sin(2 * np.pi * f0 * t)


def band(f0, bw, fs, n, floor=1e-4):
    """Frequency grid of an rfft of length n and the indices where the pulse spectrum exceeds
    `floor` (all frequency-domain models only evaluate those bins)."""
    f = np.fft.rfftfreq(n, 1 / fs)
    idx = np.nonzero(gaussian_spectrum(f, f0, bw) > floor)[0]
    return f, idx


def causal_phase_slowness(mat: Material, f, f_ref):
    """Excess slowness 1/c(f) − 1/c(f_ref) implied by the power-law attenuation through the
    Kramers–Kronig relations (Szabo 1994):
        n ≠ 1:  Δ(1/c) = α₀ tan(nπ/2) (|ω|^{n−1} − |ω_ref|^{n−1})
        n = 1:  Δ(1/c) = −(2/π) α₀ ln|ω/ω_ref|
    with α(ω) = α₀|ω|ⁿ. Returns s/m."""
    n = mat.atten_n
    w = 2 * np.pi * np.maximum(np.abs(np.asarray(f, float)), 1.0)
    wr = 2 * np.pi * f_ref
    a1 = mat.alpha(1e6)                       # Np/m at 1 MHz
    a0 = a1 / (2 * np.pi * 1e6) ** n          # Np/m per (rad/s)^n
    if a1 == 0 or abs(n - 2.0) < 1e-9:
        return np.zeros_like(w)
    if abs(n - 1.0) < 1e-6:
        return -(2 / np.pi) * a0 * np.log(w / wr)
    return a0 * np.tan(n * np.pi / 2) * (w ** (n - 1) - wr ** (n - 1))


def propagator(mat: Material, f, dist, f_ref, c=None, dispersive=True):
    """exp(−α(f) d − i ω d / c(f)) for a one-way path of length `dist` (broadcast)."""
    c = mat.cL if c is None else c
    w = 2 * np.pi * f
    slow = 1.0 / c + (causal_phase_slowness(mat, f, f_ref) if dispersive else 0.0)
    return np.exp(-mat.alpha(f) * dist - 1j * w * dist * slow)


def irfft_band(spec_band, idx, n, axis=-1):
    """Inverse rfft of a spectrum that is non-zero only on bins `idx` (last axis)."""
    shape = list(spec_band.shape)
    shape[axis] = n // 2 + 1
    full = np.zeros(shape, complex)
    sl = [slice(None)] * len(shape); sl[axis] = idx
    full[tuple(sl)] = spec_band
    return sfft.irfft(full, n, axis=axis, workers=-1)


def analytic_band(spec_band, idx, n, axis=-1):
    """Analytic signal (positive frequencies doubled) from band-limited rfft bins."""
    shape = list(spec_band.shape)
    shape[axis] = n
    full = np.zeros(shape, complex)
    sl = [slice(None)] * len(shape); sl[axis] = idx
    full[tuple(sl)] = 2 * spec_band
    return sfft.ifft(full, n, axis=axis, workers=-1)


def envelope(x, axis=-1):
    return np.abs(analytic(x, axis))


def analytic(x, axis=-1):
    """Analytic signal via scipy.fft (multi-threaded)."""
    x = np.asarray(x, float)
    n = x.shape[axis]
    X = sfft.fft(x, axis=axis, workers=-1)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    shp = [1] * x.ndim; shp[axis] = n
    return sfft.ifft(X * h.reshape(shp), axis=axis, workers=-1)


def db(x, ref=None, floor=-120.0):
    x = np.abs(np.asarray(x, float))
    ref = np.max(x) if ref is None else ref
    with np.errstate(divide="ignore"):
        return np.maximum(20 * np.log10(np.maximum(x, 1e-300) / max(ref, 1e-300)), floor)


def snr_db(signal_peak, noise):
    return 20 * np.log10(np.abs(signal_peak) / max(np.std(noise), 1e-300))


def wiener_deconvolve(trace, wavelet, snr=100.0):
    """Wiener deconvolution of `trace` by `wavelet` (same sampling), returning the reflectivity
    estimate. Regularisation 1/snr relative to the peak wavelet power."""
    n = len(trace)
    W = np.fft.rfft(wavelet, n)
    X = np.fft.rfft(trace, n)
    reg = np.max(np.abs(W)) ** 2 / snr
    return np.fft.irfft(X * np.conj(W) / (np.abs(W) ** 2 + reg), n)
