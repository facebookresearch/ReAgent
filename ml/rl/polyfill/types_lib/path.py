#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import re
import uuid


_URI_SCHEME_RE = re.compile(r"([-+.a-zA-Z0-9]+)://(.*)$")


def _new_file_name(base, filename=None, extension=None):
    filename = filename if filename is not None else str(uuid.uuid4())
    if extension is not None:
        filename = filename + "." + extension
    return os.path.join(base, filename)


def _new_dir_name(base, dirname=None):
    dirname = dirname if dirname else str(uuid.uuid4())
    return os.path.join(base, dirname)


class Path:
    def __init__(self, path=None, filename=None, extension=None, basepath=None):
        if path is None:
            if basepath is not None:
                base_dir = basepath
            elif filename is None:
                base_dir = self._root_dir()
            else:
                base_dir = _new_dir_name(self._root_dir())
        else:
            uri_scheme, short_path = self._split_uri_scheme(path)
            if uri_scheme is not None and uri_scheme not in self.uri_schemes:
                raise ValueError(
                    "Cannot initialise a {} with uri scheme {}".format(
                        self.type_name, uri_scheme
                    )
                )
            normalized_path = self._normalize_path(uri_scheme, short_path)
            if filename is None and extension is None:
                self._path = normalized_path
                return
            else:
                base_dir = normalized_path

        self.do_create_dir(base_dir)
        self._path = _new_file_name(base_dir, filename, extension)

    @classmethod
    def _normalize_path(cls, uri, short_path):
        return short_path

    @staticmethod
    def _split_uri_scheme(s):
        match = _URI_SCHEME_RE.match(str(s))
        if match is None:
            return None, s
        else:
            return match.group(1).lower(), match.group(2)

    def __str__(self):
        return self.__repr__()

    def str(self):
        return self.__repr__()

    def __repr__(self):
        return self._path

    def __eq__(self, other):
        return isinstance(other, Path) and self._path == other.path

    def __hash__(self):
        return hash(self._path)

    @classmethod
    def _root_dir(cls):
        return NotImplementedError()

    def do_create_dir(self, dir_):
        raise NotImplementedError()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value
