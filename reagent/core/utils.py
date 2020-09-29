#!/usr/bin/env python3


class lazy_property(object):
    """
    More or less copy-pasta: http://stackoverflow.com/a/6849299
    Meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self._fget = fget
        self.__doc__ = fget.__doc__
        self.__name__ = fget.__name__

    def __get__(self, obj, obj_cls_type):
        if obj is None:
            return None
        value = self._fget(obj)
        setattr(obj, self.__name__, value)
        return value
