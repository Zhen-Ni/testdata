#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Optional, Literal, Union, overload, Iterable
import scipy.signal as signal

from .xydata import Spectrum, XYData, Storage, as_linrange
from .misc import as_int


__all__ = ('get_spl', 'get_spectrum', 'psd', 'csd')

P_REF = 2e-5


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
    x = as_linrange(x)
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

    freq = as_linrange(f)
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
