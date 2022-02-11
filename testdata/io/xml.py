#!/usr/bin/env python3

import base64
import xml.etree.ElementTree as ET
from typing import Union, Any

import numpy as np

from .. import core
from ..core.xydata import Storage


__all__ = 'dump_element', 'load_element'


def dump(data: Any, file):
    element = dump_element(data)
    etree = ET.ElementTree(element)
    etree.write(file)


def load(file) -> Any:
    etree = ET.parse(file)
    return load_element(etree.getroot())


def dumps(data: Any) -> bytes:
    return ET.tostring(dump_element(data))


def loads(code: bytes) -> Any:
    return load_element(ET.fromstring(code))


def dump_element(data: Any) -> ET.Element:
    if isinstance(data, (type(None), bool,
                         int, float, complex,
                         bytes, str,
                         tuple, list, set, dict)):
        return _dump_builtin(data)
    elif isinstance(data, Storage):
        return _dump_storage(data)
    elif isinstance(data, core.XYData):
        return _dump_xydata(data)
    elif isinstance(data, core.Channel):
        return _dump_channel(data)
    elif isinstance(data, core.Section):
        return _dump_section(data)
    else:
        raise TypeError(f'type "{type(data)}" is not supported')


def load_element(element: ET.Element) -> Any:
    if element.tag == 'builtin':
        return _load_builtin(element)
    elif element.tag == 'storage':
        return _load_storage(element)
    elif element.tag == 'xydata':
        return _load_xydata(element)
    elif element.tag == 'channel':
        return _load_channel(element)
    elif element.tag == 'section':
        return _load_section(element)
    else:
        raise AttributeError(f'tag {element.tag} is not supported')


def _dump_builtin(data: Union[None, bool,
                              int, float, complex,
                              bytes, str,
                              tuple, list, set, dict]
) -> ET.Element:
    if isinstance(data, type(None)):
        element = ET.Element('builtin', {'type': 'None'})
    elif isinstance(data, bool):
        element = ET.Element('builtin', {'type': 'bool'})
        element.text = f'{int(data)}'
    elif isinstance(data, int):
        element = ET.Element('builtin', {'type': 'int'})
        element.text = f'{data}'
    elif isinstance(data, float):
        element = ET.Element('builtin', {'type': 'float'})
        element.text = f'{data}'
    elif isinstance(data, complex):
        element = ET.Element('builtin', {'type': 'complex'})
        element.text = f'{data}'
    elif isinstance(data, bytes):
        element = ET.Element('builtin', {'type': 'bytes'})
        element.text = base64.encodebytes(data).decode()
    elif isinstance(data, str):
        element = ET.Element('builtin', {'type': 'str'})
        element.text = f'{data}'
    elif isinstance(data, tuple):
        element = ET.Element('builtin', {'type': 'tuple'})
        element.extend([dump_element(i) for i in data])
    elif isinstance(data, list):
        element = ET.Element('builtin', {'type': 'list'})
        element.extend([dump_element(i) for i in data])
    elif isinstance(data, set):
        element = ET.Element('builtin', {'type': 'set'})
        element.extend([dump_element(i) for i in data])
    elif isinstance(data, dict):
        element = ET.Element('builtin', {'type': 'dict'})
        element.extend([dump_element((key, value)) for
                        key, value in data.items()])
    else:
        assert 0, "This should not happen"
    return element


def _load_builtin(element: ET.Element) -> Union[None, bool,
                                                int, float, complex,
                                                bytes, str,
                                                tuple, list, set, dict]:
    if element.attrib['type'] == 'None':
        res = None
    elif element.attrib['type'] == 'bool':
        res = bool(int(element.text))
    elif element.attrib['type'] == 'int':
        res = int(element.text)
    elif element.attrib['type'] == 'float':
        res = float(element.text)
    elif element.attrib['type'] == 'complex':
        res = complex(element.text)
    elif element.attrib['type'] == 'bytes':
        res = base64.decodebytes(element.text.encode())
    elif element.attrib['type'] == 'str':
        res = element.text
    elif element.attrib['type'] == 'tuple':
        res = tuple([load_element(i) for i in element])
    elif element.attrib['type'] == 'list':
        res = [load_element(i) for i in element]
    elif element.attrib['type'] == 'set':
        res = set([load_element(i) for i in element])
    elif element.attrib['type'] == 'dict':
        res = dict([load_element(i) for i in element])
    else:
        assert 0, "This should not happen"
    return res


def _dump_storage(storage: Storage) -> ET.Element:
    # Special storage formats for LinRange or LogRange.
    # Use Array for other storage types.
    dtype = np.dtype(storage.dtype).name
    if isinstance(storage, core.LinRange):
        element = ET.Element('storage', {'type': 'LinRange',
                                         'dtype': dtype,
                                         'size': f'{storage.size}',
                                         'step': f'{storage.step}',
                                         'start': f'{storage.start}'})
    elif isinstance(storage, core.LogRange):
        element = ET.Element('storage', {'type': 'LogRange',
                                         'dtype': dtype,
                                         'size': f'{storage.size}',
                                         'step': f'{storage.step}',
                                         'start': f'{storage.start}'})
    elif isinstance(storage, Storage):
        element = ET.Element('storage', {'type': 'Array',
                                         'dtype': dtype})
        element.text = ', '.join([f'{i}' for i in storage])
    else:
        assert 0, "This should not happen"
    return element


def _load_storage(element: ET.Element) -> Storage:
    """Load storage from xml element.
    
    Please make sure element.tag is "storage". It will not be checked
    in this funciton.
    """
    # Special storage formats for LinRange or LogRange.
    # Array is used for other storage types.
    dtype = np.dtype(element.attrib['dtype']).type
    res: Storage
    if element.attrib['type'] == 'LinRange':
        size = int(element.attrib['size'])
        step = dtype(element.attrib['step'])
        start = dtype(element.attrib['start'])
        res = core.LinRange(size, step, start, dtype)
    elif element.attrib['type'] == 'LogRange':
        size = int(element.attrib['size'])
        step = dtype(element.attrib['step'])
        start = dtype(element.attrib['start'])
        res = core.LogRange(size, step, start, dtype)
    elif element.attrib['type'] == 'Array':
        data = element.text.split(',')
        res = core.Array(dtype(data))
    else:
        assert 0, "This should not happen"
    return res


def _dump_xydata(data: core.XYData) -> ET.Element:
    element = ET.Element('xydata')
    element.append(_dump_storage(data.x))
    element.append(_dump_storage(data.y))
    element.append(dump_element(data.info))
    return element


def _load_xydata(element: ET.Element) -> core.XYData:
    x = _load_storage(element[0])
    y = _load_storage(element[1])
    info = load_element(element[2])
    return core.XYData(x, y, info)


def _dump_channel(channel: core.Channel) -> ET.Element:
    element = ET.Element('channel', {'name': channel.name})
    element.append(dump_element(channel.source_data))
    element.append(dump_element(channel.records))
    return element


def _load_channel(element: ET.Element) -> core.Channel:
    name = element.attrib['name']
    source_data = load_element(element[0])
    records = load_element(element[1])
    return core.Channel(name, records, source_data)


def _dump_section(section: core.Section) -> ET.Element:
    element = ET.Element('section', {'name': section.name})
    element.append(dump_element(section.records))
    element.append(dump_element(section.channels))
    return element


def _load_section(element: ET.Element) -> core.Section:
    name = element.attrib['name']
    records = load_element(element[0])
    channels = load_element(element[1])
    return core.Section(name, records, channels)
