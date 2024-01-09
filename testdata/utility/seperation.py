#!/usr/bin/env python3

"""Seperate signal into broadband and tonal components. (Experimental)"""
from typing import overload
import numpy as np
from scipy import signal

from ..core import (Storage, ArrayLike, XYData, Spectrum, LinRange,
                    as_storage)


__all__ = ('get_broadband_empirical', )

BROADBAND_EMPITICAL_THRESHOLD = 1.5


@overload
def get_broadband_empirical(data: Spectrum,
                            order: int,
                            threshold: float = ...
                            ) -> Spectrum: ...


@overload
def get_broadband_empirical(data: ArrayLike,
                            order: int,
                            threshold: float = ...
                            ) -> Storage: ...


def get_broadband_empirical(data,
                            order,
                            threshold=BROADBAND_EMPITICAL_THRESHOLD):
    if isinstance(data, XYData):
        if not isinstance(data.x, LinRange):
            raise ValueError('Input should have linranged x')
        filtered_y = np.array(data.y)
    else:
        filtered_y = np.asarray(data)
    b = [1 / order] * order
    index = np.ones_like(filtered_y)
    while index.any():
        # Use convolve to save about 40% computation time than using
        # signal.filtfilt.
        y_ref = signal.convolve(filtered_y, b, mode='same')
        index = abs(filtered_y / y_ref) > threshold
        filtered_y[index] = y_ref[index]
    if isinstance(data, XYData):
        return Spectrum(data.x, filtered_y)
    else:
        return as_storage(filtered_y)
