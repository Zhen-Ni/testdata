#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Optional, Literal, Union, Tuple, overload, Iterable
import scipy.signal as signal

from .xydata import Spectrum, LinRange, LogRange, XYData, Storage
from .channel import SpectrumChannel
from .misc import as_int


__all__ = ('get_spl', 'get_spectrum', 'psd', 'csd',
           'OctaveBand')

P_REF = 2e-5
OCTAVE_REFERENCE_FREQUENCY = 1000.


@overload
def get_decibel(s: float,
                s_ref: float = ...
                ) -> float: ...


@overload
def get_decibel(s: Iterable[float],
                s_ref: float = ...
                ) -> npt.NDArray[float]: ...


@overload
def get_decibel(s: XYData,
                s_ref: float = ...
                ) -> XYData: ...


def get_decibel(s, s_ref=1.):
    """Calculate the decibel.

    The expression for decibel is:
    L = 10 * log10 (abs(s / s_ref))

    Parameters
    ----------
    s : float or array_like or XYData object
        Signal intensity or power.
    s_ref: float, optional
        Reference signal intensity or power.

    Returns
    -------
    signal_level: float or array_like or XYData
        Decibel of the input signal.
    """
    if isinstance(s, XYData):
        y = get_decibel(s.y, s_ref)
        return XYData(s.x, y).derive(type(s))
    else:
        return 10 * np.log10(abs(np.asarray(s) / s_ref))


@overload
def get_spl(p: float,
            type: Literal['power', 'pressure'] = ...,
            p_ref: Optional[float] = ...
            ) -> float: ...


@overload
def get_spl(p: Iterable[float],
            type: Literal['power', 'pressure'] = ...,
            p_ref: Optional[float] = ...
            ) -> npt.NDArray[float]: ...


@overload
def get_spl(p: XYData,
            type: Literal['power', 'pressure'] = ...,
            p_ref: Optional[float] = ...
            ) -> XYData: ...


def get_spl(p, type='power', p_ref=None):
    """Calculate the sound power/pressure level (SPL).

    The expression for SPL for sound power is:
    SPL = 10 * log10(p / p_ref ** 2)
    where p stands for sound power.

    The expression for SPL for sound pressure is:
    SPL = 20 * log10(p / p_ref)
    where p stands for sound pressure.

    The constant `p_ref` is 2e-5 by default, which can be modified by
    global variable P_REF.

    Parameters
    ----------
    p : float or array_like or XYData object
        Sound power or sound pressure.
    type : {'power', 'pressure'}, optional
        Selects between calculating the sound power level and sound
        pressure level. Defaults to 'power'.
    p_ref: float, optional
        The reference sound power/pressure for calculating SPL. Defaults
        to 2e-5.
    """
    p_ref = P_REF if p_ref is None else p_ref
    if type == 'power':
        return get_decibel(p, p_ref ** 2)
    elif type == 'pressure':
        return 2 * get_decibel(p, p_ref)
    else:
        raise ValueError("type must be 'power' or 'pressure'.")


def psd(x: Storage,
        y: Storage,
        df: float = 1.,
        window: Union[str, tuple, npt.ArrayLike] = 'hann',
        return_onesided: bool = True,
        scaling: Literal['density', 'spectrum'] = 'density',
        normalization: Literal['power', 'amplitude'] = 'power',
        **kwargs
        ) -> Spectrum:
    """Calculate the power spectral density of input data.

    This function is a wrapper of scipy.signal.welch. It estimates the
    power spectral density of the input xydata using the Welch's
    method.

    Parameters
    ----------
    x: Storage
        The sampling points.
    y: Storage
        Storage objects containing the data.
    df: float, optional
        The frequency resolution. Defaults to 1.0.
    window: str or tuple or array_like, optional
        The desired window to use. See scipy.signal.welch for more
        information. Defaults to 'hann'.
    return_onesided: bool, optional
        If `True` and input data is real, return a one-sided spectrum.
        Defaults to `True`.
    scaling: {'density', 'spectrum'}, optional
        Selects between computing the power spectral density and the
        power spectrum. Defaults to 'density'.
    normalization: {'power', 'amplitude'}, optional
        Selects between the normalization method for the window
        function. Defaults to 'power'.
    kwargs: dict, optional
        Other keyword arguments to be passed to scipy.signal.welch.

    Returns
    -------
    spectrum: Spectrum
        The desired power spectrum (density), with additional
        arguments stored in its `info`.
    """
    return csd(x, y, y,
               df, window, return_onesided, scaling, normalization,
               **kwargs)


def csd(x: Storage,
        y1: Storage,
        y2: Storage,
        df: float = 1.,
        window: Union[str, tuple, npt.ArrayLike] = 'hann',
        return_onesided: bool = True,
        scaling: Literal['density', 'spectrum'] = 'density',
        normalization: Literal['power', 'amplitude'] = 'power',
        **kwargs
        ) -> Spectrum:
    """Calculate the cross-spectral density of input data.

    This function is a wrapper of scipy.signal.csd. It estimates the
    cross-spectral density of the input xydata using the Welch's
    method.

    Parameters
    ----------
    x: Storage
        The sampling points.
    y1: Storage
        Storage objects containing the data.
    y2: Storage
        Storage objects containing the data.
    df: float, optional
        The frequency resolution. Defaults to 1.0.
    window: str or tuple or array_like, optional
        The desired window to use. See scipy.signal.csd for more
        information. Defaults to 'hann'.
    return_onesided: bool, optional
        If `True` and input data is real, return a one-sided spectrum.
        Defaults to `True`.
    scaling: {'density', 'spectrum'}, optional
        Selects between computing the cross power spectral density and the
        cross power spectrum. Defaults to 'density'.
    normalization: {'power', 'amplitude'}, optional
        Selects between the normalization method for the window
        function. Defaults to 'power'.
    kwargs: dict, optional
        Other keyword arguments to be passed to scipy.signal.csd.

    Returns
    -------
    spectrum: Spectrum
        The desired cross power spectrum (density), with additional
        arguments stored in its `info`.
    """
    x = LinRange.from_storage(x)
    if not len(x) == len(y1) == len(y2):
        raise ValueError('size of x, y1 and y2 should be the same')
    dt = x.step
    fs = 1 / dt
    nperseg = as_int(fs / df)
    # In scipy.signal.welch, scaling='density' uses power
    # normalization, and scaling='spectrum' uses amplitude
    # normalization.
    if normalization == 'power':
        f, pxy = signal.csd(y1, y2, fs, window, nperseg,
                            return_onesided=return_onesided,
                            scaling='density', **kwargs)
        if scaling == 'spectrum':
            pxy *= df
    elif normalization == 'amplitude':
        f, pxy = signal.csd(y1, y2, fs, window, nperseg,
                            return_onesided=return_onesided,
                            scaling='spectrum', **kwargs)
        if scaling == 'density':
            pxy /= df
    else:
        raise ValueError('normalization must be "power" or "amplitude"')

    # The order of the frequency and pxy should be adjusted if a two-sided
    # spectrum is returned by `csd`.
    index = np.argsort(f)
    f = f[index]
    pxy = pxy[index]

    freq = LinRange.from_storage(f)
    return Spectrum(freq, pxy,
                    dict(df=df, window=window,
                         return_onesided=return_onesided,
                         scaling=scaling,
                         normalization=normalization, **kwargs))


def get_spectrum(xydata: XYData,
                 df: float = 1.,
                 window: Union[str, tuple, npt.ArrayLike] = 'hann',
                 return_onesided: bool = True,
                 scaling: Literal['density', 'spectrum'] = 'density',
                 normalization: Literal['power', 'amplitude'] = 'power',
                 **kwargs
                 ) -> Spectrum:
    """Calculate the spectrum of input xydata.

    This function is a wrapper of scipy.signal.welch. It estimates the
    power spectral (density) of the input xydata using the Welch's
    method.

    Parameters
    ----------
    xydata: XYData
        The input data in time domain.
    df: float, optional
        The frequency resolution. Defaults to 1.
    window: str or tuple or array_like, optional
        The desired window to use. See scipy.signal.welch for more
        information. Defaults to 'hann'.
    return_onesided: bool, optional
        If `True` and input data is real, return a one-sided spectrum.
        Defaults to `True`.
    scaling: {'density', 'spectrum'}, optional
        Selects between computing the power spectral density and the
        power spectrum. Defaults to 'density'.
    normalization: {'power', 'amplitude'}, optional
        Selects between the normalization method for the window
        function. Defaults to 'power'.
    kwargs: dict, optional
        Other keyword arguments to be passed to scipy.signal.welch.

    Returns
    -------
    spectrum: Spectrum
        The desired power spectrum (density), with additional arguments
        stored in its `info`.
    """
    return csd(xydata.x, xydata.y, xydata.y,
               df, window, return_onesided, scaling, normalization,
               **kwargs)


OctaveRatioType = Literal['base10', 'base2']


class OctaveBand:
    """Octave band manager.

    The definition of octave band and the corresponding midband
    frequency and bandedge frequencies are defined by national
    standart GB/T 3241 as follows.

    The octave ratio `G` is defined as:
    'base2': G = 2
    'base10': G = 10 ** 0.3

    The bandwidth designator '1/b' is a positive int.

    Thus, the midband frequency is defined as: fm = G ** (x / b) * fr,
    if b is odd fm = G ** ((2x + 1) / 2b) * fr, if b is even where x
    is the index, and fr is the reference frequency. According to the
    standard, fr is set to 1000 Hz by default.

    The lower and upper bandedge frequencies are:
    f1 = G ** -(1 / 2b) * fm
    and
    f2 = G ** (1 / 2b) * fm

    The most commonly used one-third octave band is:
    OctaveBand(3, 'base10', 1000.)

    Parameters
    ----------
    inv_designator : positive int
        Inverse of the bandwidth designator. Defaults to 3.
    octave_ratio : {'base2', 'base10}
        Octave ratio selection. Defaults to 'base10'.
    reference_frequency : float or None
        The reference_frequency. If None is given, it is set to
        OCTAVE_REFERENCE_FREQUENCY. Defaults to None.
    """

    def __init__(self,
                 inv_designator: int = 3,
                 octave_ratio: OctaveRatioType = 'base10',
                 reference_frequency: Optional[float] = None):
        if octave_ratio == 'base2':
            ratio = 2.
        elif octave_ratio == 'base10':
            ratio = 10 ** 0.3
        else:
            raise ValueError('octave_ratio must be "2" or "10"')
        inv_designator = int(inv_designator)
        factor = ratio ** (1 / (2 * inv_designator))
        self._fr = OCTAVE_REFERENCE_FREQUENCY if \
            reference_frequency is None else reference_frequency
        self._ratio = ratio
        self._inv_designator = inv_designator
        self._factor = factor

    @property
    def reference_frequency(self):
        return self._fr

    @property
    def octave_ratio(self):
        return self._ratio

    @overload
    def midband_frequency(self,
                          index: int) -> float: ...

    @overload
    def midband_frequency(self,
                          index: Iterable[int]) -> npt.NDArray[float]: ...

    def midband_frequency(self,
                          index: int | Iterable[int]
                          ) -> float | npt.NDArray[float]:
        """Get the midband frequency of the given index.

        Parameters
        ----------
        index : int or Iterable[int]
            The index of the octave band.

        Returns
        -------
        fm : float or np.ndarray
            The midband frequency.
        """
        if np.iterable(index):
            index = np.asarray(index)
        if self._inv_designator & 1:  # odd
            fm = self._factor ** (2 * index) * self._fr
        else:                   # even
            fm = self._factor ** (2 * index + 1) * self._fr
        return fm

    @overload
    def bandedge_frequency(self, index: int) -> Tuple[float, float]: ...

    @overload
    def bandedge_frequency(self,
                           index: Iterable[int]
                           ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]: ...

    def bandedge_frequency(self,
                           index: int | Iterable[int]
                           ) -> Tuple[float, float] | Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Get the bandedge frequenc.

       Parameters
        ----------
        index : int or Iterable[int]
            The index of the octave band.

        Returns
        -------
        f1 : float or np.ndarray
            The lower bandedge frequency.
        f2 : float or np.ndarray
            The upper bandedge frequency.
        """
        fm = self.midband_frequency(index)
        f1 = fm / self._factor
        f2 = fm * self._factor
        return f1, f2

    @overload
    def index(self, frequency: float) -> int: ...

    @overload
    def index(self, frequency: Iterable[float]) -> npt.NDArray[int]: ...

    def index(self,
              frequency: float | Iterable[float]
              ) -> int | npt.NDArray[int]:
        """Get the octave index band of the given frequency."""
        quotient = frequency / self._fr
        index = np.log(quotient) / np.log(self.octave_ratio)
        if self._inv_designator & 1:  # odd
            index *= self._inv_designator
        else:                   # even
            index *= self._inv_designator - 0.5
        if np.iterable(frequency):
            return np.array(np.round(index), dtype=int)
        else:
            return as_int(index)

    def power(self,
              spectrum: Union[Spectrum, SpectrumChannel],
              lower_frequency: Optional[float] = None,
              upper_frequency: Optional[float] = None,
              kind: Literal['linear', 'cubic'] = 'linear'
              ) -> Spectrum:
        """Get the total power of octave bands of given spectrum.

        The result is calculated by integrating the input spectrum
        along the x-axis between octave bandedge frequencies. Thus,
        the input spectrum should be power density.  Do not use
        kind='cubic' unless you know exactly what you are doing.
        """
        from scipy.interpolate import interp1d
        freq = spectrum.f
        pxx = spectrum.pxx
        # pxx values with frequency < 0 are ignored
        pxx = pxx[freq > 0]
        freq = freq[freq > 0]
        start = (self.index(freq[0]) if
                 lower_frequency is None else
                 self.index(lower_frequency) - 1)
        end = (self.index(freq[-1]) if
               upper_frequency is None else
               self.index(upper_frequency) + 1)
        index = np.arange(start, end)
        # Use ranged storage to save space here. It can be substitued
        # by `x = self.midband_frequency(index[1:])` for more
        # accuracy.
        x = LogRange(len(index)-1,
                     self._factor ** 2,
                     self.midband_frequency(index[1]))
        x_bounds = self.bandedge_frequency(index)[1]
        func_y = interp1d(freq, pxx, kind=kind,
                          copy=False, assume_sorted=True)
        y = []
        i = 0        # index for freq and pxx
        # Dealing with different octave bands seperately.
        for j in range(len(index) - 1):
            f1, f2 = x_bounds[j], x_bounds[j + 1]
            xj, yj = [f1], [func_y(f1)]
            while freq[i] < f2:
                if freq[i] < f1:
                    i += 1
                    continue
                xj.append(freq[i])
                yj.append(pxx[i])
                i += 1
            xj.append(f2)
            yj.append(func_y(f2))
            yi = _octave_power_quad_helper(xj, yj, kind)
            y.append(yi)
        return Spectrum(x, y)


def _octave_power_quad_helper(x: npt.ArrayLike,
                              y: npt.ArrayLike,
                              kind: Literal['linear', 'cubic']):
    """Integrate y(x)."""
    if kind == 'linear':
        from scipy.integrate import trapezoid
        return trapezoid(y, x)
    elif kind == 'cubic':
        from scipy.integrate import simpson
        return simpson(y, x)
    else:
        raise ValueError('kind should be "linear" or "cubic"')
