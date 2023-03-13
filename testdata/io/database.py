#!/usr/bin/env python3

"""Manage saving and opening of sections using sqlite3.

Section can be saved in databases managed by sqlite3. A database can
contain multiple sections. The database contains one table `section`,
which stores pickled section data.
"""


from __future__ import annotations
from typing import Optional, List, Any, Generator, Literal
import sqlite3
import urllib.parse
import pickle
import zlib
import re

from ..core.section import Section


__all__ = 'dumps', 'loads', 'SectionDB'


def dumps(obj: Any, compress=False) -> bytes:
    bytes_ = pickle.dumps(obj)
    if compress:
        bytes_ = zlib.compress(bytes_)
    return bytes_


def loads(bytes_: bytes, compress=False) -> Any:
    if compress:
        bytes_ = zlib.decompress(bytes_)
    return pickle.loads(bytes_)


URI_MODES = 'ro', 'rw', 'rwc', 'memory'
DEFAULT_TIMEOUT = 5.


class SectionDB:
    """Database manager for section objects.

    Section objects are stored in database managed by sqlite3. All
    sections are saved in table named `section`. Each record in the
    table contains two values: `name` and `section`. The `name` is the
    primary key and is the same as the name of the section object. The
    `section` is the pickled section object.

    Parameters
    ----------
    dbname : str
        The filename of the database.
    mode : {'ro', 'rw', 'rwc', 'memory', optional
        Uniform Resource Identifiers (URI) mode. Mode 'ro' for read-only,
        'rw' for read-write, 'rwc' for read-write and created if it does
        not exist, and 'memory' for database that is a pure in-memory that
        never interacts with disk. Defaults to 'ro'.
    timeout : float, optional
        How long the connection should wait for the lock to go away until
        raising an exception. If set to None, it is defined by 
        DEFAULT_TIMEOUT, which is 5.0 (5 secondes )by default. If set to
        0, an expection will be raised immediately if the database is locked.
        Setting timeout to negative values has the same effect as setting it
        to 0. Defaults to None.

    References
    ----------
    [1] About URI: https://www.sqlite.org/uri.html
    """
    def __init__(self,
                 dbname: str,
                 mode: Literal['ro', 'rw', 'rwc', 'memory'] = 'ro',
                 timeout: Optional[float] = None):
        # In case dbname contains special characters, which causes
        # ambiguity in Uniform Resource Identifiers (URI). See
        # https://stackoverflow.com/questions/5607551/how-to-urlencode-a-querystring-in-python
        self._dbname = urllib.parse.quote(dbname)

        if mode in URI_MODES:
            self._mode = mode
        else:
            raise ValueError(f'mode must be in {URI_MODES}')
        self._timeout = DEFAULT_TIMEOUT if timeout is None else timeout
        self._con: Optional[sqlite3.Connection] = None
        self._cur: Optional[sqlite3.Cursor] = None

    @property
    def con(self):
        if self._con is None:
            raise AttributeError('connect is not established')
        return self._con

    @property
    def cur(self):
        if self._cur is None:
            self._cur = self.con.cursor()
        return self._cur
        
    def connect(self) -> None:
        if self._con is not None:
            self.close()
        # sqlite3 adds URI support since python version 3.4. Using URI
        # features can connect a database with different modes, like
        # read-only (ro), read-write (rw), read-write-create (rwc),
        # etc. See
        # https://docs.python.org/3/library/sqlite3.html#sqlite3.connect
        self._con = sqlite3.connect(f'file:{self._dbname}?mode={self._mode}',
                                    timeout=self._timeout,
                                    uri=True)
        self.create_table()

    def commit(self) -> None:
        self.con.commit()

    def close(self) -> None:
        if self._con:
            con = self.con
            self._con = None
            self._cur = None
            con.commit()
            con.close()
            self._con = None

    def vacuum(self) -> None:
        """Shrink the size of the database."""
        self.cur.execute('VACUUM')
            
    def __enter__(self) -> SectionDB:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def create_table(self) -> None:
        self.cur.execute("""CREATE TABLE IF NOT EXISTS section
                            (name CHAR PRIMARY KEY, section BLOB)""")

    def insert(self, section: Section) -> int:
        bytes_ = dumps(section)
        self.cur.execute("INSERT INTO section VALUES (?,?)",
                         (section.name, bytes_))
        return self.cur.rowcount

    def update(self, section: Section) -> int:
        bytes_ = dumps(section)
        self.cur.execute("UPDATE section SET section=? WHERE name=?",
                         (bytes_, section.name))
        return self.cur.rowcount

    def select(self, name: str) -> Optional[Section]:
        self.cur.execute("SELECT * FROM section WHERE name=?", (name,))
        if res := self.cur.fetchone():
            return loads(res[1])

    def delete(self, name: str) -> int:
        self.cur.execute("DELETE FROM section WHERE name=?", (name,))
        return self.cur.rowcount

    def list(self, pattern: Optional[str] = None) -> List[str]:
        self.cur.execute("SELECT DISTINCT name FROM section")
        res = self.cur.fetchall()
        if pattern is None:
            return [i[0] for i in res]
        pattern_ = re.compile(pattern)
        return [i[0] for i in res if re.search(pattern_, i[0])]

    def iterfind(self, pattern: str) -> Generator[Section, None, None]:
        for name in self.list(pattern):
            # In case the the database changes during iteration and
            # res becomes None.
            if res := self.select(name):
                yield res

    def findall(self, pattern: str) -> List[Section]:
        return list(self.iterfind(pattern))

    def __iter__(self):
        return self.iterfind('')
