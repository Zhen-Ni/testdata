#!/usr/bin/env python3

"""Find fundamental frequency of harmonics."""


from __future__ import annotations
import functools

import numpy as np

from ..core import as_linrange, Spectrum

__all__ = ('find_fundamental', 'find_fundamental_hps',
           'find_fundamental_brute', 'find_fundamental_cepstrum')


def find_fundamental_hps(spec: Spectrum,
                         order: int,
                         start: float | None = None,
                         stop: float | None = None,
                         threshold: float | None = None) -> float:
    """Use HPS algorithm to estimate fundamental frequency.

    The Harmonic Product Spectrum (HPS) algorithm finds the
    fundamental frequency by multiplying the frequencies by the ratio
    of harmonic number. The strongest harmonic peaks should line up
    and the fundamental frequency can be obtained.

    Note that sometimes the estimated frequency is sometimes too high
    (octave error)[1]. To correct, after the strongest peak is chosen,
    peaks with frequency (f / n) are compared with the strongest peak,
    where f is the frequency of the strongest peak and n = 1, 2, ...,
    order. If the ratios of the amplitude of these peaks and the
    strongest peak are larger than a given threshold, the peak is also
    selected. The gcd of the frequencies of these selected peaks is
    chosen as the fundamental frequency.

    Parameters
    ----------
    spec: Spectrum
        The spectrum to process.
    order: int
        The number of harmonics being considered.
    start, stop: float or None
        The bounds for estimating the fundamental frequency.
    threshold: float or None, optional
        The threshold used to correct octave error, between 0 and 1.
        If None is given, octave error is not corrected. defaults to
        None.

    Returns
    -------
    fundamental_frequency: float
        The fundamental tonal frequency.

    Reference
    ---------
    [1] http://musicweb.ucsd.edu/~trsmyth/analysis/Harmonic_Product_Spectrum.html
    """
    freq_start = spec.f[0]
    if start is not None:
        freq_start = max(freq_start, start)
    freq_stop = spec.f[-1] / order
    if stop is not None:
        freq_stop = min(freq_stop, stop)
    divisor = functools.reduce(np.lcm, range(1, order+1))
    freq_step = spec.df / divisor
    freq = np.arange(freq_start, freq_stop, freq_step)
    amp = np.ones_like(freq)
    for i in range(order):
        amp *= np.interp(freq, spec.f / (i + 1), spec.pxx)
    idx = np.argmax(amp)
    fundamental_frequency = freq[idx]

    # Correct octave error only if threshold is given.
    if threshold is None:
        return fundamental_frequency
    common_factors = []
    pxx_max = np.interp(fundamental_frequency, spec.f, spec.pxx)
    for i in range(2, order + 1):
        ampi = np.interp(fundamental_frequency / i, spec.f, spec.pxx)
        if ampi / pxx_max > threshold:
            common_factors.append(i)
    if common_factors:
        fundamental_frequency /= functools.reduce(np.lcm, common_factors)
    return fundamental_frequency


def find_fundamental_brute(spec: Spectrum,
                           start: float,
                           stop: float,
                           step: float) -> float:
    """Find fundamental tonal frequency by brute force.

    The spectrum is expressed by spl(f), where f is the frequency and
    spl is the amplitude. This method uses an array of fs where the
    starting frequency, stoping frequency and step size are given by
    `start`, `stop` and `step`. The corresponding spls are
    interpolated if necessary. The mean of the spls are then obtained
    and the fundamental frequency with maximum mean spl is returned as the
    fundamental tonal frequency.

    Parameters
    ----------
    spec: Spectrum
        The spectrum to process.
    start, stop, step: float
        Defining the points for brute search.

    Returns
    -------
    fundamental_frequency: float
        The fundamental tonal frequency.
    """
    fs = np.arange(start, stop, step)
    mean = []
    max_f = spec.x[-1]
    for i, f in enumerate(fs):
        f_targets = f * (np.arange(max_f // f) + 1)
        spls = np.interp(f_targets, spec.f, spec.spl)
        mean.append(spls.mean())
    return fs[np.argmax(mean)]


def find_fundamental_cepstrum(spec: Spectrum,
                              start: float,
                              stop: float):
    """Use cepstrum to find fundamental tonal frequency.

    Parameters
    ----------
    spec: Spectrum
        The spectrum to process.
    start, stop: float
        Defining the possible range for fundamental frequency.

    Returns
    -------
    fundamental_frequency: float
        The fundamental tonal frequency.
    """
    f = as_linrange(spec.x)
    ceps = abs(np.fft.rfft(spec.spl))
    stop_idx = round(f[-1] / start)
    start_idx = round(f[-1] / stop)
    idx = np.argmax(ceps[start_idx: stop_idx])
    return f[-1] / (idx + start_idx)


find_fundamental = find_fundamental_hps
