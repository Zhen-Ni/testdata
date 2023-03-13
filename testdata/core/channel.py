#!/usr/bin/env python3

"""Define channels"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypeVar, Type, Union, Literal
import copy
import numpy as np
import numpy.typing as npt

from .xydata import XYData, Spectrum, LinRange

if TYPE_CHECKING:
    from .misc import Real, InfoDict
    from .section import Section
    from ..io.loader import LoaderInfo


__all__ = ('Channel', 'SpectrumChannel', 'ChannelType')

    
ChannelType = TypeVar('ChannelType', bound='Channel')


class Channel:
    """Channel for testdata.

    Each channel records source data and corresponding information.
    The source data is stored as an xydata. All data in channel are
    picklable.  If the source data is very large, it can be discarded
    when pickling. To restore source data, loader info can be stored
    in the channel instance and can be used to recover source data
    from source file.

    Parameters
    ----------
    name : str
        The name for the channel. This value is used for identify this
        channel from others in the same section.
    records: dict, optional
        Dict for storing additional information. It can also be used to
        store other information for derived classes.
    source_data : XYData, optional
        The source data to be stored. If None, it can be loaded by
        loader using loader_info. Defaults to None.
    loader_info: LoaderInfo, optional
        Information for loader to restore source data.
    is_source_reserved: bool or None, optional
        Whether to store source data when pickling. If source data is
        very large and loader_info can be provided, this value is
        suggested to set to False to optimize file size when
        pickling. If set to True, the source data will be pickled. If
        set to None, it depends on loader_info. If loader_info is
        provided, source data will not be saved (equal to False). If
        loader_info is None, source data will be pickled (equal to
        True).  Defaults to None.
    """
    def __init__(self,
                 name: str,
                 records: Optional[InfoDict] = None,
                 source_data: Optional[XYData] = None,
                 loader_info: Optional[LoaderInfo] = None,
                 is_source_reserved: Optional[bool] = None,
    ):
        self._name = name
        self._records = {} if records is None else records
        self._source_data = source_data
        self._loader_info = loader_info
        if is_source_reserved is None:
            is_source_reserved = True if loader_info is None else False
        self._is_source_reserved = is_source_reserved

        # self._section is controlled by Section object.
        self._section: Optional[Section]
        self._set_section(None)

    @classmethod
    def from_channel(cls: Type[ChannelType],
                     other: Channel) -> ChannelType:
        """Construct a new Channel instance.

        This functions works like a copy constructor. However, the new
        object shares resources with the given channel.
        """
        result = object.__new__(cls)
        result._name = other._name
        result._records = other._records
        result._source_data = other._source_data
        result._loader_info = other._loader_info
        result._is_source_reserved = other._is_source_reserved
        result._section = None
        return result

    def derive(self,
               NewChannelType: Type[ChannelType]
    ) -> ChannelType:
        """Convert the instance's type to `NewChannelType`.

        After calling this method, the instance's type will be changed
        to `NewChannelType`.
        """
        if not issubclass(NewChannelType, Channel):
            raise TypeError('"NewChannelType" should be subclass of "Channel"')
        self.__class__ = NewChannelType
        return self

    def __repr__(self) -> str:
        return (f'<{self.__class__.__name__} object> name: {self.name}, '
                f'section: {self.section}')

    def __str__(self) -> str:
        if self.section:
            return f'{self.section.name}-{self.name}'
        else:
            return f'{self.name}'

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name

    @property
    def records(self) -> InfoDict:
        return self._records

    @property
    def source_data(self) -> Optional[XYData]:
        return self._source_data

    @property
    def loader_info(self) -> Optional[LoaderInfo]:
        return self._loader_info

    @property
    def section(self) -> Optional[Section]:
        return self._section

    def _set_section(self, section: Optional[Section]) -> None:
        self._section = section

    def reserve_source(self, flag: Optional[bool] = None) -> bool:
        """Interface for whether to store source data.

        Parameters
        ----------
        flag : {True, False, None}, optional
            Set whether the source data will be stored when
            pickling. If set to True, the source data will be
            stored. If set to False, the source data will not be
            stored. If set to None, do nothing but returns the current
            status. Defaults to None.

        Returns
        -------
        flag : bool
            Shows whether to store source data after running this
            method.  If True, the source data will be stored. If
            false, the source data will not be stored,
        """
        if flag is not None:
            self._is_source_reserved = flag
        return self._is_source_reserved

    def vacuum(self) -> None:
        """Remove source_data if it is not reserved."""
        if not self.reserve_source():
            self.set_source_data(None)

    def get_source_data(self) -> XYData:
        """Get source data.

        This method will definitely return the source data, or raise
        an error, which is trigged by `update_source_data`.
        """
        # Guarantees self._source_data has a non-None value.
        if self._source_data is None:
            self.update_source_data()
        return self._source_data

    def set_source_data(self: ChannelType,
                        data: Optional[XYData] = None
    ) -> ChannelType:
        """Set the source data to given value manually."""
        self._source_data = data
        return self

    def update_source_data(self: ChannelType) -> ChannelType:
        """Fetch source data.

        Raises AttributeError if cannot get source
        data. self._source_data is not None if this method returns.
        """
        if self.loader_info is None:
            raise AttributeError('no loader_info')
        from ..io.loader import load_data
        data = load_data(self.loader_info)
        self.set_source_data(data)
        return self

    def __getstate__(self) -> dict:
        # Minimize stored data to save storage space.  Only basic
        # information is pickled.  Do not need to save self._section,
        # as it is managed by section objects and will be set when
        # adding the channel to section. When recovering channel from
        # bytes, self._section is always set to None.
        state = dict(_name=self._name,
                     _records=self._records,
                     _source_data=self._source_data,
                     _loader_info=self._loader_info,
                     _is_source_reserved=self._is_source_reserved)       
        if not self._is_source_reserved:
            state['_source_data'] = None
        return state

    def __setstate__(self: ChannelType, state: dict):
        self.__dict__.update(state)
        self._set_section(None)

    def __copy__(self: ChannelType) -> ChannelType:
        state = copy.copy(self.__dict__)
        res = object.__new__(type(self))
        res.__setstate__(state)
        return res

    def __deepcopy__(self: ChannelType, memo=None) -> ChannelType:
        state = copy.deepcopy(self.__dict__, memo)
        res = object.__new__(type(self))
        res.__setstate__(state)
        return res


class SpectrumChannel(Channel):
    """A spectrum channel is a channel with spectrum data.

    The source data is assumed to be time domain data and the spectrum
    data can be calculated and managed by this class. The spectrum data
    is stored in channel._records.
    """
    @property
    def spectrum(self) -> Optional[Spectrum]:
        return self._records.get('spectrum')

    @spectrum.setter
    def spectrum(self, spectrum: Spectrum):
        self._records['spectrum'] = spectrum

    def get_spectrum(self) -> Spectrum:
        # Make sure self.spectrum is not None.
        if self.spectrum is None:
            self.update_spectrum()
        return self.spectrum

    def update_spectrum(self,
                        df: float = 1.,
                        window: Union[str, tuple, npt.ArrayLike] = 'hann',
                        return_onesided: bool = True,
                        scaling: Literal['density', 'spectrum'] = 'density',
                        normalization: Literal['power', 'amplitude'] = 'power',
                        **kwargs) -> SpectrumChannel:
        self.spectrum = Spectrum.from_time_data(self.get_source_data(),
                                                df,
                                                window,
                                                return_onesided,
                                                scaling,
                                                normalization,
                                                **kwargs)
        return self

    @property
    def dt(self) -> Real:
        data = self.get_source_data()
        if isinstance(data.x, LinRange):
            return data.x.step
        raise AttributeError('only source data with linrange x '
                             'has no attribute "dt"')

    @property
    def t(self) -> npt.NDArray:
        return np.asarray(self.get_source_data().x)

    @property
    def y(self) -> npt.NDArray:
        return np.asarray(self.get_source_data().y)

    @property
    def df(self) -> Real:
        return self.get_spectrum().df

    @property
    def f(self) -> npt.NDArray:
        return self.get_spectrum().f

    @property
    def pxx(self) -> npt.NDArray:
        return self.get_spectrum().pxx

    @property
    def decibel(self) -> npt.NDArray:
        return self.get_spectrum().decibel

    @property
    def spl(self) -> npt.NDArray:
        return self.get_spectrum().spl
