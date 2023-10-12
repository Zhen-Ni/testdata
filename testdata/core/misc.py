#!/usr/bin/env python3

from typing import Union, Dict, Any

__all__ = ('Real', 'Scalar', 'InfoDict')


Real = Union[int, float]
Scalar = Union[int, float, complex]
InfoDict = Dict[str, Any]


_NOT_FOUND = object()


class cached_property:
    """This is the non-reentrant version of functools.chaned_property."""

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property "
                "to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without "
                "calling __set_name__ on it.")
        try:
            cache = instance.__dict__
        # not all objects have __dict__ (e.g. class defines slots)
        except AttributeError:
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[self.attrname] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on "
                    f"{type(instance).__name__!r} instance "
                    f"does not support item assignment for "
                    f"caching {self.attrname!r} property."
                )
                raise TypeError(msg) from None
        return val


def as_int(x: Union[int, float]) -> int:
    """Round input value and cast it into int value."""
    return int(round(x))
