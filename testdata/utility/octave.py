#!/usr/bin/env python3

from __future__ import annotations

from typing import Literal, Optional, Iterable, overload
import numpy as np
import numpy.typing as npt

from ..core import (ArrayLike, LogRange, SpectrumChannel, Spectrum,
                    Storage, as_storage, Array)
from ..core.misc import as_int

__all__ = ('OctaveBand', )

OCTAVE_REFERENCE_FREQUENCY = 1000.

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
    def midband_frequency(self, index: ArrayLike[int]) -> Array[float]: ...

    def midband_frequency(self, index):
        """Get the midband frequency of the given index.

        Parameters
        ----------
        index : int or Collection[int]
            The index of the octave band.

        Returns
        -------
        fm : float or Array
            The midband frequency.
        """
        if np.iterable(index):
            index = as_storage(index)
        if self._inv_designator & 1:  # odd
            fm = self._factor ** (2 * index) * self._fr
        else:                   # even
            fm = self._factor ** (2 * index + 1) * self._fr
        return fm

    @overload
    def bandedge_frequency(self, index: int) -> tuple[float, float]: ...

    @overload
    def bandedge_frequency(self,
                           index: ArrayLike[int]
                           ) -> tuple[Array[float], Array[float]]: ...

    def bandedge_frequency(self, index):
        """Get the bandedge frequenc.

       Parameters
        ----------
        index : int or Collection[int]
            The index of the octave band.

        Returns
        -------
        f1 : float or Array
            The lower bandedge frequency.
        f2 : float or Array
            The upper bandedge frequency.
        """
        fm = self.midband_frequency(index)
        f1 = fm / self._factor
        f2 = fm * self._factor
        return f1, f2

    @overload
    def index(self, frequency: float) -> int: ...

    @overload
    def index(self, frequency: ArrayLike[float]) -> Storage[int]: ...

    def index(self, frequency):
        """Get the octave index band of the given frequency."""
        quotient = frequency / self._fr
        index = np.log(quotient) / np.log(self.octave_ratio)
        if self._inv_designator & 1:  # odd
            index = index * self._inv_designator
        else:                   # even
            index = index * self._inv_designator - 0.5
        if np.iterable(frequency):
            return as_storage(np.round(index), dtype=int)
        else:
            return as_int(index)

    def power(self,
              spectrum: Spectrum | SpectrumChannel,
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
