#!/usr/bin/env python3

"""Define Section"""

from __future__ import annotations
from typing import (TYPE_CHECKING, TypeVar, Iterable, Any, List,
                    Tuple, Type, Optional)
if TYPE_CHECKING:
    from .misc import InfoDict

import re

from .channel import Channel

__all__ = ('Section', 'SectionType'
)

SectionType = TypeVar('SectionType', bound='Section')


DEFAULT_NAME = 'unnamed section'


class Section:
    """Test data section.

    Each Section corresponds to a certain test condition and contains
    several channels.
    """
    def __init__(self,
                 name: Optional[str] = None,
                 records: InfoDict = None,
                 channels: Iterable[Channel] = ()
                 ):
        self._name = DEFAULT_NAME if name is None else name
        self._records = {} if records is None else records
        self._channels: List[Channel] = []
        self.extend_channel(channels)

    def __repr__(self):
        return f'<{self.__class__.__name__} object> name: {self.name}'

    def __str__(self):
        return f'{self.name}'

    def derive(self,
               NewSectionType: Type[SectionType]
    ) -> SectionType:
        """Convert the instance's type to `NewSectionType`.

        After calling this method, the instance's type will be changed
        to `NewSectionType`.
        """
        if not issubclass(NewSectionType, Section):
            raise TypeError('"NewSectionType" should be subclass of "Section"')
        self.__class__ = NewSectionType
        return self

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def records(self) -> dict:
        return self._records

    @property
    def channels(self) -> Tuple[Channel, ...]:
        return tuple(self._channels)

    def append_channel(self,
                       channel: Channel) -> None:
        # channel._section indicates which section the channel belongs
        # to, which is managed by this class.  A channel can only
        # belong to one section.  Set the channel's section to self
        # when adding the channel.
        if channel.section is not None:
            raise AttributeError(f"{channel} belongs to other section")
        channel._set_section(self)
        self._channels.append(channel)

    def extend_channel(self, channels: Iterable[Channel]) -> None:
        for channel in channels:
            if channel.section is not None:
                raise AttributeError("not all input channels are isolated")
        for channel in channels:
            channel._set_section(self)
        self._channels.extend(channels)

    def insert_channel(self, index: int, channel: Channel) -> None:
        if channel.section is not None:
            raise AttributeError(f"{channel} belongs to other section")
        channel._set_section(self)
        self._channels.insert(index, channel)

    def pop_channel(self: SectionType,
                    index: int = -1) -> Channel:
        channel = self._channels.pop(index)
        channel._set_section(None)
        return channel

    def clear_channel(self) -> List[Channel]:
        channels = [i for i in self.channels]
        self._channels.clear()
        for channel in channels:
            channel._set_section(None)
        return channels

    def __getitem__(self, index: Any):
        """Get items in channels or records.

        This method is designed for easy to use, do not rely too much
        on it. No corresponding __setitem__ should be defined.
        """
        if isinstance(index, int):
            return self._channels[index]
        elif isinstance(index, str):
            return self._records[index]
        else:
            raise TypeError('index should be int or string')

    def __iter__(self):
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def find_channels(self, name: str) -> List[Channel]:
        result = []
        pattern = re.compile(name)
        for channel in self.channels:
            if re.search(pattern, channel.name):
                result.append(channel)
        return result

    def find_records(self, name: str) -> InfoDict:
        result = {}
        pattern = re.compile(name)
        for key, value in self.records.items():
            if re.search(pattern, key):
                result[key] = value
        return result

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        # channel.section is set to None when unpickling. Need to set
        # it here.
        for channel in self.channels:
            channel._set_section(self)
