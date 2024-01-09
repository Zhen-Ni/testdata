#!/usr/bin/env python3


"""Muodule for basic test data structure."""

from __future__ import annotations
from typing import (Union, Literal, TypeVar, Collection, Sequence,
                    Generic, Type, Any, overload, final)
import abc

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import numpy.typing as npt

from .misc import Scalar, InfoDict, cached_property, parse_slice


__all__ = ('ArrayLike',
           'Storage', 'Array', 'LinRange', 'LogRange',
           'XYData', 'Spectrum',
           'as_storage', 'as_linrange', 'as_logrange'
           )


ScalarType = TypeVar('ScalarType', int, float, complex)
ArrayLike = Sequence[ScalarType] | Collection[ScalarType]


class Storage(Sequence, NDArrayOperatorsMixin, Generic[ScalarType]):
    __slots__ = ()

    @property
    @abc.abstractmethod
    def dtype(self) -> Type[ScalarType]: pass

    @overload
    def __getitem__(self, index: int) -> ScalarType: ...
    @overload
    def __getitem__(self, index: slice) -> Storage[ScalarType]: ...

    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @abc.abstractmethod
    def __len__(self) -> int: pass

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} object> "
                f"size: {len(self)}, "
                f"dtype: {self.dtype}")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Storage):
            if self.dtype != other.dtype:
                return False
            if len(self) != len(other):
                return False
            return (np.asarray(self) == np.asarray(other)).all()
        if isinstance(other, Sequence):
            return self == as_storage(other)
        return False

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __array__(self, dtype: Union[Type[ScalarType], str,
                                     npt.DTypeLike, None] = None
                  ) -> npt.NDArray:
        """Convert self to np.ndarray.

        Always copy the internal storage to avoid modification.Derived
        classes are encouraged to overwrite this method to improve
        performance.
        """
        if dtype is None:
            dtype = self.dtype
        return np.array(list(self), dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs) -> Storage:
        """Add support for numpy universal function calls."""
        inputs = []
        for arg in args:
            if isinstance(arg, Storage):
                inputs.append(np.asarray(arg))
            else:
                inputs.append(arg)
        result_array = getattr(ufunc, method)(*inputs, **kwargs)
        return as_storage(result_array)


class ArrayBase(Storage, Generic[ScalarType]):
    @abc.abstractmethod
    def _asarray(self) -> npt.NDArray:
        """Access internal array storage."""
        pass

    @final
    def __array__(self, dtype: Union[Type[ScalarType], str,
                                     npt.DTypeLike, None] = None
                  ) -> npt.NDArray:
        # Get a copy of self._array to avoid modification.
        if dtype is None:
            return self._asarray().copy()
        return np.array(self._asarray, dtype=dtype)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ArrayBase):
            if self.dtype != other.dtype:
                return False
            if len(self) != len(other):
                return False
            return (self._asarray() == other._asarray()).all()
        return super().__eq__(other)


class Array(ArrayBase, Generic[ScalarType]):
    """A wrapper for efficient data storage.
    """

    __slots__ = ('_array', )

    def __init__(self,
                 array: ArrayLike[ScalarType],
                 dtype: Union[Type[ScalarType], str,
                              npt.DTypeLike, None] = None):
        # Always copy the input iterable object.
        if (isinstance(array, Collection) and
                not isinstance(array, Sequence)):
            array = list(array)
        self._array = np.array(array, dtype=dtype)

    @property
    def dtype(self) -> Type[ScalarType]:
        return self._array.dtype.type

    def _asarray(self) -> npt.NDArray:
        return self._array

    @staticmethod
    def frombytes(b: bytes,
                  dtype: Union[Type[ScalarType], str,
                               npt.DTypeLike, None] = None
                  ) -> Array[ScalarType]:
        array = np.frombuffer(b, dtype=dtype)
        return Array(array, dtype)

    def tobytes(self) -> bytes:
        return self._array.tobytes()

    def __len__(self) -> int:
        return len(self._array)

    @overload
    def __getitem__(self, index: int) -> ScalarType: ...
    @overload
    def __getitem__(self, index: slice) -> ArraySlice[ScalarType]: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            if -len(self) <= index < len(self):
                return self._array[index]
            raise IndexError('list index out of range')
        if isinstance(index, slice):
            return ArraySlice(self, index)
        raise TypeError('indices must be integers or slices')

    def __reduce__(self):
        return (self.__class__.frombytes,
                (self.tobytes(), self.dtype))


class ArraySlice(ArrayBase, Generic[ScalarType]):
    __slots__ = ('_storage', '_start', '_step', '_size')

    def __init__(self, storage: Array, index: slice):
        self._storage = storage
        size, step_idx, start_idx = parse_slice(len(self._storage), index)
        self._start = start_idx
        self._step = step_idx
        self._size = size

    @property
    def dtype(self) -> Type[ScalarType]:
        return self._storage.dtype

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> ScalarType: ...
    @overload
    def __getitem__(self, index: slice) -> ArraySlice[ScalarType]: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            if -len(self) <= index < len(self):
                return self._storage[self._start + self._step * index]
            raise IndexError('list index out of range')
        if isinstance(index, slice):
            if isinstance(self, ArraySlice):
                secondary_size, secondary_step, secondary_start = \
                    parse_slice(len(self), index)
                start = self._start + self._step * secondary_start
                step = self._step * secondary_step
                size = secondary_size
                stop = start + step * size
                return ArraySlice(self._storage,
                                  slice(start, stop, step))
            return ArraySlice(self, index)
        raise TypeError('indices must be integers or slices')

    def _asarray(self) -> npt.NDArray:
        start = self._start
        step = self._step
        stop = self._start + self._step * self._size
        if stop < 0:
            stop -= len(self._storage)
        return self._storage._array[start: stop: step]


class RangedStorage(Storage, Generic[ScalarType]):
    """Ranged storage."""

    __slots__ = ('_size', '_dtype', '_step', '_start')

    @property
    def dtype(self) -> Type[ScalarType]:
        return self._dtype

    def __init__(self,
                 size: int,
                 step: ScalarType,
                 start: ScalarType = 0,
                 dtype: Union[Type[ScalarType], str,
                              npt.DTypeLike] = None):
        self._size = int(size)
        self._dtype = np.dtype(np.result_type(step, start) if
                               dtype is None else dtype).type
        self._step: ScalarType = self._dtype(step)
        self._start: ScalarType = self._dtype(start)

    @property
    def size(self) -> int:
        return self._size

    @property
    def step(self) -> ScalarType:
        return self._step

    @property
    def start(self) -> ScalarType:
        return self._start

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} object> "
                f"size: {self.size}, "
                f"step: {self.step}, "
                f"start: {self.start}, "
                f"dtype: {self.dtype}")

    def __len__(self) -> int:
        return self.size

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RangedStorage):
            if self.dtype != other.dtype:
                return False
            if not ((self.start == other.start) and
                    (self.step == other.step) and
                    (self.size == other.size)):
                return False
            return True
        return super().__eq__(other)


class LinRange(RangedStorage, Generic[ScalarType]):
    """A wrapper for Storage with linspaced data.

    It is a wrapper for class Storage. When the data is evenly spaced,
    only some key information are necessary for recovering the
    data. Thus, it saves disk space when pickling.

    Parameters
    ----------
    size: int
        The length of the data.
    step: int or float
        The difference of two adjacent data.
    start: int or float, optional
        The start value of data. Defaults to 0.
    dtype: npt.DTypeLike, optional
        Data type for storage. None for auto detection. Defaults to None.
    """

    def __array__(self, dtype: Union[Type[ScalarType], str,
                                     npt.DTypeLike, None] = None
                  ) -> npt.NDArray:
        result = np.arange(self.size, dtype=self.dtype)
        result *= self.step
        result += self.start
        return result

    @overload
    def __getitem__(self, index: int) -> ScalarType: ...
    @overload
    def __getitem__(self, index: slice) -> LinRange[ScalarType]: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            if -len(self) <= index < len(self):
                if index < 0:
                    index = len(self) + index
                return self.start + self.step * index
            raise IndexError('list index out of range')
        if isinstance(index, slice):
            size, step, start = parse_slice(self.size, index)
            return LinRange(size,
                            self.step * step,
                            self.start + self.step * start)
        raise TypeError('indices must be integers or slices')


class LogRange(RangedStorage, Generic[ScalarType]):
    """A wrapper for Storage with logspaced data.

    It is a wrapper for class Storage. When the data is spaced evenly
    on a log scale, only some key information are necessary for
    recovering the data. Thus, it saves disk space when pickling.

    Parameters
    ----------
    size: int
        The length of the data.
    step: int or float
        The quotient of two adjacent data.
    start: int or float, optional
        The start value of data. Defaults to 1.
    dtype: npt.DTypeLike, optional
        Data type for storage. None for auto detection. Defaults to None.
    """

    def __array__(self, dtype: Union[Type[ScalarType], str,
                                     npt.DTypeLike, None] = None
                  ) -> npt.NDArray:
        result = self.start * self.step ** np.arange(self.size)
        return np.asarray(result, dtype=self.dtype)

    @overload
    def __getitem__(self, index: int) -> ScalarType: ...
    @overload
    def __getitem__(self, index: slice) -> LogRange[ScalarType]: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            if -len(self) <= index < len(self):
                if index < 0:
                    index = len(self) + index
                return self.start * self.step ** index
            raise IndexError('list index out of range')
        if isinstance(index, slice):
            size, step, start = parse_slice(self.size, index)
            return LogRange(size,
                            self.step ** step,
                            self.start * self.step ** start)
        raise TypeError('indices must be integers or slices')


XType = TypeVar('XType', int, float, complex)
YType = TypeVar('YType', int, float, complex)

XYDataType = TypeVar('XYDataType', bound='XYData')


class XYPoint(Generic[XType, YType]):
    def __init__(self, x: XType, y: YType):
        self._x: XType = x
        self._y: YType = y

    @property
    def x(self) -> XType:
        return self._x

    @property
    def y(self) -> YType:
        return self._y


class XYData(Generic[XType, YType]):
    """Data structure for test data.

    TestData contains paired x and y values and additional
    information.
    """

    def __init__(self,
                 x: ArrayLike[XType],
                 y: ArrayLike[YType],
                 info: InfoDict = {},
                 ):
        _x = as_storage(x) if not isinstance(x, Storage) else x
        _y = as_storage(y) if not isinstance(y, Storage) else y
        if len(_x) != len(_y):
            raise ValueError('length of x and y should be the same')
        self._x: Storage[XType] = _x
        self._y: Storage[YType] = _y
        self._info = info

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object> size: {len(self.x)}"

    def __len__(self) -> int:
        return len(self._x)

    def derive(self,
               NewXYDataType: Type[XYDataType]
               ) -> XYDataType:
        if not issubclass(NewXYDataType, XYData):
            raise TypeError('"NewXYDataType" should be '
                            'subclass of "XYDataType"')
        self.__class__ = NewXYDataType
        return self

    @property
    def x(self) -> Storage[XType]:
        return self._x

    @property
    def y(self) -> Storage[YType]:
        return self._y

    @property
    def info(self) -> InfoDict:
        return self._info

    def __getstate__(self):
        return self.x, self.y, self.info

    def __setstate__(self, state):
        self.__init__(*state)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, XYData):
            return (self.x == other.x and
                    self.y == other.y and
                    self.info == other.info)
        return False

    @overload
    def __getitem__(self, index: int) -> XYPoint[XType, YType]: ...
    @overload
    def __getitem__(self, index: slice) -> XYData[XType, YType]: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            return XYPoint(self.x[index], self.y[index])
        if isinstance(index, slice):
            x = self.x[index]
            y = self.y[index]
            return XYData(x, y)
        raise TypeError('indices must be integers or slices')


class SpectrumPoint(XYPoint, Generic[XType, YType]):
    @property
    def f(self) -> XType:
        return self.x

    @property
    def pxx(self) -> YType:
        return self.y


class Spectrum(XYData, Generic[XType, YType]):
    """Data structure for Spectrum.

    It is a wrapper for XYData, but with more functions for spectral
    analysis.
    """
    @overload
    def __getitem__(self, index: int) -> SpectrumPoint[XType, YType]: ...
    @overload
    def __getitem__(self, index: slice) -> Spectrum[XType, YType]: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            return SpectrumPoint(self.x[index], self.y[index])
        if isinstance(index, slice):
            x = self.x[index]
            y = self.y[index]
            return Spectrum(x, y)
        raise TypeError('indices must be integers or slices')

    @property
    def df(self) -> Scalar:
        if isinstance(self.x, LinRange):
            return self.x.step
        raise AttributeError('only linrange x has attribute "df"')

    @property
    def f(self) -> npt.NDArray:
        return np.asarray(self.x)

    @property
    def pxx(self) -> npt.NDArray:
        return np.asarray(self.y)

    @cached_property
    def decibel(self) -> npt.NDArray:
        """Get the decibel.

        The decibel is calculated using the item with key 's_ref'
        stored in the `info` of the instance. If 's_ref' does not
        exist, the default value provided by get_decibel will be used.

        See Also
        --------
        get_decibel, get_spl
        """
        from .tools import get_decibel
        s_ref = self.info.get('s_ref')
        if s_ref is None:
            raise AttributeError('s_ref is not defined')
        return np.asarray(get_decibel(self.pxx, s_ref))

    @cached_property
    def spl(self) -> npt.NDArray:
        """Get the sound power level (SPL).

        The sound pressure level is calculated using the item with key
        'p_ref' stored in the `info` of the instance. If 'p_ref' does
        not exist, the default value P_REF will be used.

        See Also
        --------
        get_spl, get_decibel
        """
        from .tools import get_spl
        return np.asarray(get_spl(self.pxx, 'power',
                                  p_ref=self.info.get('p_ref')))

    @staticmethod
    def from_time_data(xydata: XYData,
                       df: float = 1.,
                       window: Union[str, tuple, npt.ArrayLike] = 'hann',
                       return_onesided: bool = True,
                       scaling: Literal['density', 'spectrum'] = 'density',
                       normalization: Literal['power', 'amplitude'] = 'power',
                       **kwargs) -> Spectrum:
        """Calculate the spectrum of input xydata.

        See Also
        --------
        get_spectrum
        """
        from .tools import get_spectrum
        return get_spectrum(xydata=xydata,
                            df=df,
                            window=window,
                            return_onesided=return_onesided,
                            scaling=scaling,
                            normalization=normalization,
                            **kwargs)


_AS_STORAGE_LENGTH_THRESHOLD = 5                  # must be larger than 3


def as_storage(x: ArrayLike[ScalarType],
               dtype: Union[Type[ScalarType], str, npt.DTypeLike,
                            None] = None
               ) -> Storage[ScalarType]:
    if len(x) < _AS_STORAGE_LENGTH_THRESHOLD:
        return Array(x, dtype)
    it = iter(x)
    a = next(it)
    b = next(it)
    c = next(it)
    result: Storage[ScalarType]
    if np.isclose(a + c, b + b):
        try:
            result = as_linrange(x, dtype)
        except ValueError:
            pass
        else:
            return result
    elif np.isclose(b * b, a * c):
        try:
            result = as_logrange(x, dtype)
        except ValueError:
            pass
        else:
            return result
    return Array(x, dtype)


def as_linrange(x: ArrayLike[ScalarType],
                dtype: Union[Type[ScalarType], str, npt.DTypeLike,
                             None] = None
                ) -> LinRange[ScalarType]:
    if isinstance(x, LinRange):
        return LinRange(x.size, x.step, x.start, x.dtype)
    _x = list(x)
    start = _x[0]
    stop = _x[-1]
    if len(_x) != 1:
        step = (stop - start) / (len(_x) - 1)
    else:
        step = 0
    size = len(_x)
    result: LinRange[ScalarType] = LinRange(size, step, start, dtype)
    if not np.allclose(_x, np.asarray(result)):
        raise ValueError('cannot construct LinRange object from given input')
    return result


def as_logrange(x: ArrayLike[ScalarType],
                dtype: Union[Type[ScalarType], str, npt.DTypeLike,
                             None] = None
                ) -> LogRange[ScalarType]:
    if isinstance(x, LogRange):
        return LogRange(x.size, x.step, x.start, x.dtype)
    _x = list(x)
    start = _x[0]
    stop = _x[-1]
    if len(_x) != 1:
        step = (stop / start) ** (1 / (len(_x) - 1))
    else:
        step = 0
    size = len(_x)
    result: LogRange[ScalarType] = LogRange(size, step, start, dtype)
    if not np.allclose(_x, np.asarray(result)):
        raise ValueError('cannot construct LogRange object from given input')
    return result
