#!/usr/bin/env python3


"""Muodule for basic test data structure."""

from __future__ import annotations
from typing import (Union, Literal, Iterable, overload, TypeVar,
                    Generic, Type, Any)
import abc

import numpy as np
import numpy.typing as npt

from .misc import Real, InfoDict, cached_property


__all__ = ('Array', 'LinRange', 'LogRange',
           'XYData', 'Spectrum',
           'as_linrange',
)


T = TypeVar('T', int, float, complex)
RealType = TypeVar('RealType', int, float)


class Storage(abc.ABC, Generic[T]):
    __slots__ = ()

    @property
    @abc.abstractmethod
    def dtype(self) -> Type[T]: pass

    @abc.abstractmethod
    def __array__(self) -> npt.NDArray:
        """Convert self to np.ndarray."""
        pass

    @abc.abstractmethod
    def __repr__(self) -> str: pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> T: pass

    @abc.abstractmethod
    def __len__(self) -> int: pass

    # Might be unnecessary.
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def __eq__(self, other) -> bool: pass

    def __ne__(self, other: Any) -> bool:
        return not (self == other)


class Array(Storage, Generic[T]):
    """A wrapper for efficient data storage.
    """

    __slots__ = ('_array', )

    def __init__(self,
                 iterable: Iterable[T],
                 dtype: Union[Type[T], str, npt.DTypeLike, None] = None):
        # Always copy the input iterable object.
        self._array = np.array(iterable, dtype=dtype)

    @property
    def dtype(self) -> Type[T]:
        return self._array.dtype.type

    @staticmethod
    def frombytes(b: bytes,
                  dtype: Union[Type[T], str, npt.DTypeLike, None] = None
    ) -> Array[T]:
        array = np.frombuffer(b, dtype=dtype)
        return Array(array, dtype)

    def __array__(self) -> npt.NDArray[T]:
        # Get a copy of self._array to avoid modification.
        return self._array.copy()

    def tobytes(self) -> bytes:
        return self._array.tobytes()

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, index: int) -> T:
        if not isinstance(index, int):
            raise TypeError('index must be int')
        if -len(self) <= index < len(self):
            return self._array[index]
        else:
            raise IndexError('list index out of range')

    def __repr__(self):
        return (f"<{self.__class__.__name__} object> "
                f"size: {len(self)}', "
                f"dtype: {self.dtype}")

    def __reduce__(self):
        return (self.__class__.frombytes,
                (self.tobytes(), self.dtype))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.dtype != other.dtype:
            return False
        return (self._array == other._array).all()


StorageType = TypeVar('StorageType', bound=Storage)


class RangedStorage(Storage, abc.ABC):

    __slots__ = ()
    
    """Ranged storage."""
    @property
    @abc.abstractmethod
    def size(self) -> int: pass

    @property
    @abc.abstractmethod
    def step(self) -> Real: pass

    @property
    @abc.abstractmethod
    def start(self) -> Real: pass

    @classmethod
    @abc.abstractmethod
    def from_storage(cls: Type[StorageType],
                     other: Storage) -> StorageType: pass

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} object> "
                f"size: {self.size}, "
                f"step: {self.step}, "
                f"start: {self.start}, "
                f"dtype: {self.dtype}")

    def __len__(self) -> int:
        return self.size

    
class LinRange(RangedStorage, Generic[RealType]):
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
    typecode: npt.DTypeLike, optional
        Data type for storage. None for auto detection. Defaults to None.
    """

    __slots__ = ('_size', '_dtype', '_step', '_start')

    def __init__(self,
                 size: int,
                 step: RealType,
                 start: RealType = 0,
                 dtype: Union[Type[RealType], str, npt.DTypeLike] = None):
        self._size = int(size)
        self._dtype = np.dtype(np.result_type(step, start) if
                               dtype is None else dtype).type
        self._step = self._dtype(step)
        self._start = self._dtype(start)

    @property
    def dtype(self) -> Type[RealType]:
        return self._dtype

    def __array__(self) -> npt.NDArray[RealType]:
        return np.arange(self.size) * self.step + self.start

    def __getitem__(self, index: int) -> RealType:
        if not isinstance(index, int):
            raise TypeError('index must be int')
        if -len(self) <= index < len(self):
            if index < 0:
                index = len(self) - index
            return self.start + self.step * index
        else:
            raise IndexError('list index out of range')

    @property
    def size(self) -> int:
        return self._size

    @property
    def step(self) -> RealType:
        return self._step

    @property
    def start(self) -> RealType:
        return self._start

    def __reduce__(self):
        return (self.__class__,
                (self.size, self.step, self.start, self.dtype))

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.dtype != other.dtype:
            return False
        if (self.start == other.start and
            self.step == other.step and
            self.size == other.size):
            return True
        return False

    @classmethod
    def from_storage(cls: Type[LinRange],
                     other: Storage) -> LinRange:
        if isinstance(other, cls):
            return other
        start = other[0]
        stop = other[-1]
        if len(other) == 1:
            step = 0
        else:
            step = (stop - start) / (len(other) - 1)
        size = len(other)
        result = LinRange(size, step, start, other.dtype)
        if not np.allclose(other, np.asarray(result)):
            raise ValueError('cannot construct LinRange '
                             'object from given input')
        return result


class LogRange(RangedStorage, Generic[RealType]):
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
        The start value of data. Defaults to 0.
    typecode: npt.DTypeLike, optional
        Data type for storage. None for auto detection. Defaults to None.
    """

    __slots__ = ('_size', '_dtype', '_step', '_start')

    def __init__(self,
                 size: int,
                 step: RealType,
                 start: RealType = 0,
                 dtype: Union[Type[T], str, npt.DTypeLike] = None):
        self._size = int(size)
        self._dtype = np.dtype(np.result_type(step, start) if
                               dtype is None else dtype).type
        self._step = self._dtype(step)
        self._start = self._dtype(start)

    @property
    def dtype(self) -> Type[RealType]:
        return self._dtype

    def __array__(self) -> npt.NDArray[RealType]:
        return self.start * self.step ** np.arange(self.size)

    def __getitem__(self, index: int) -> RealType:
        if not isinstance(index, int):
            raise TypeError('index must be int')
        if -len(self) <= index < len(self):
            if index < 0:
                index = len(self) - index
            return self.start * self.step ** index
        else:
            raise IndexError('list index out of range')

    @property
    def size(self) -> int:
        return self._size

    @property
    def step(self) -> RealType:
        return self._step

    @property
    def start(self) -> RealType:
        return self._start

    def __reduce__(self):
        return (self.__class__,
                (self.size, self.step, self.start, self.dtype))

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.dtype != other.dtype:
            return False
        if (self.start == other.start and
            self.step == other.step and
            self.size == other.size):
            return True
        return False

    @classmethod
    def from_storage(cls: Type[LogRange],
                     other: Storage) -> LogRange:
        if isinstance(other, cls):
            return other
        start = other[0]
        stop = other[-1]
        if start == 0 or len(other) == 1:
            step = 0
        else:
            step = (stop / start) ** (1 / (len(other) - 1))
        size = len(other)
        result = LogRange(size, step, start, other.dtype)
        if not np.allclose(other, np.asarray(result)):
            raise ValueError('cannot construct LogRange '
                             'object from given input')
        return result


XYDataType = TypeVar('XYDataType', bound='XYData')


class XYData:
    """Data structure for test data.

    TestData contains paired x and y values and additional
    information. The x data must be evenly spaced.
    """

    __slots__ = ('_x', '_y', '_info')

    def __init__(self,
                 x: Iterable,
                 y: Iterable,
                 info: InfoDict = {},
                 ):
        _x = Array(x) if not isinstance(x, Storage) else x
        _y = Array(y) if not isinstance(y, Storage) else y
        if len(_x) != len(_y):
            raise ValueError('length of x and y should be the same')
        self._x = _x
        self._y = _y
        self._info = info

    def __repr__(self):
        return f"<{self.__class__.__name__} object> size: {len(self.x)}"

    def derive(self,
               NewXYDataType: Type[XYDataType]
    ) -> XYDataType:
        if not issubclass(NewXYDataType, XYData):
            raise TypeError('"NewXYDataType" should be '
                            'subclass of "XYDataType"')
        self.__class__ = NewXYDataType
        return self

    @property
    def x(self) -> Storage:
        return self._x

    @property
    def y(self) -> Storage:
        return self._y

    @property
    def info(self) -> InfoDict:
        return self._info

    def __getstate__(self):
        return self.x, self.y, self.info

    def __setstate__(self, state):
        self.__init__(*state)

    def __eq__(self, other: XYData) -> bool:
        return (self.x == other.x and
                self.y == other.y and
                self.info == other.info)


class Spectrum(XYData):
    """Data structure for Spectrum.

    It is a wrapper for XYData, but with more functions for spectral
    analysis.
    """

    __slots__ = ('__dict__')

    @property
    def df(self) -> Real:
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
        return get_decibel(self.pxx, s_ref)

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
        return get_spl(self.pxx, 'power', p_ref=self.info.get('p_ref'))

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


@overload
def as_linrange(x: LinRange[RealType]) -> LinRange[RealType]: ...


@overload
def as_linrange(x: Iterable[RealType],
                dtype: Union[Type[RealType], str, npt.DTypeLike, None] = ...
) -> LinRange[RealType]: ...


def as_linrange(x: LinRange[RealType] | Iterable[RealType],
                dtype: Union[Type[RealType], str, npt.DTypeLike,
                             None] = None
) -> LinRange[RealType]:
    if isinstance(x, LinRange):
        return x
    _x = tuple(x)
    start = _x[0]
    stop = _x[-1]
    if len(_x) != 1:
        step = (stop - start) / (len(_x) - 1)
    else:
        step = 0
    size = len(_x)
    result = LinRange(size, step, start, dtype)
    if not np.allclose(_x, np.asarray(result)):
        raise ValueError('cannot construct LinRange object from given input')
    return result
