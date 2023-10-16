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


def parse_slice(container_length: int, index: slice
                ) -> tuple[int, int, int]:
    """Parse the slice index of a container into size, step and start pairs.

    For sequencial container `c`, its items can be reached by
    `c[index]` where `index` can be either int or slice object. If
    `index` is a slice object, it is in the form of `[start: stop:
    step]`, and either of the three values can be None. This function
    parses the slice index object into `size`, `step` and `start`
    pairs.
    """
    step_idx = 1 if index.step is None else index.step
    if step_idx == 0:
        raise ValueError('slice step cannot be zero')
    start_idx = index.start
    if start_idx is None:
        start_idx = 0 if step_idx > 0 else container_length - 1
    else:
        if start_idx < 0:
            start_idx = container_length + start_idx
        if start_idx < 0:
            start_idx = 0
        if start_idx >= container_length:
            start_idx = container_length - 1
    stop_idx = index.stop
    if stop_idx is None:
        stop_idx = -1 if step_idx < 0 else container_length
    else:
        if stop_idx < 0:
            stop_idx = container_length + stop_idx
        if stop_idx < -1:
            stop_idx = -1
        if stop_idx > container_length:
            stop_idx = container_length
    size = ((stop_idx - start_idx) // step_idx +
            bool((stop_idx - start_idx) % step_idx))
    if size < 0:
        size = 0
    return size, step_idx, start_idx
