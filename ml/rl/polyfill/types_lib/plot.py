#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import base64
import struct
from typing import Generic, Iterable, List, Optional, Tuple, TypeVar, Union

from ml.rl.polyfill.exceptions import NonRetryableTypeError


NumericType = Union[int, float]
InputPointType = TypeVar("InputPointType")
PlotPointType = TypeVar("PlotPointType", bound=Union[List, Tuple])


class Plot(Generic[InputPointType, PlotPointType]):
    def __init__(self):
        self._series = []
        self._options = {"xAxis": {}, "yAxis": {}}
        self._format = "2f"

    def add_series(self, points, name=None, point_names=None):
        # type: (Iterable[InputPointType], Optional[str], Optional[List[str]]) -> None
        plot_points = self._process_points(points)
        series = {
            "base64_points": base64.b64encode(
                # pyre-fixme[32]: Variable argument must be an iterable.
                b"".join(struct.pack(self._format, *p) for p in plot_points)
            ).decode("ascii"),
            "length": len(plot_points),
        }
        if name:
            series["name"] = name
        if point_names is not None:
            if not isinstance(point_names, list):
                point_names = list(point_names)
            assert len(plot_points) == len(point_names)
            if all(isinstance(name, (int, float)) for name in point_names):
                series["base64_point_names"] = base64.b64encode(
                    b"".join(struct.pack("f", point_name) for point_name in point_names)
                ).decode("ascii")
            else:
                # pyre-fixme[6]: Expected `Union[int, str]` for 2nd param but got
                #  `List[str]`.
                series["point_names"] = point_names
        self._series.append(series)

    def _process_points(self, points):
        # type: (Iterable[InputPointType]) -> List[PlotPointType]
        raise NonRetryableTypeError("Using PLOT directly is not supported")

    def set_title(self, title):
        self._options["title"] = str(title)


SeriesPointType = Union[List[NumericType], Tuple[NumericType]]


class PointSeriesPlot(Plot[SeriesPointType, SeriesPointType]):
    # pyre-fixme[14]: `_process_points` overrides method defined in `Plot`
    #  inconsistently.
    def _process_points(self, points):
        # type: (Iterable[SeriesPointType]) -> List[SeriesPointType]
        processed = []
        for item in points:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], (int, float))
                and isinstance(item[1], (int, float))
            ):
                processed.append(item)
            else:
                raise NonRetryableTypeError("All points must be pairs of numbers.")
        return processed


# Marker types to distinguish line plots from scatter plots
class LinePlot(PointSeriesPlot):
    pass


HistogramPointType = Tuple[float, float]


class Histogram(Plot[NumericType, HistogramPointType]):
    def __init__(
        self,
        bins=None,  # type: Optional[int]
        bin_start=None,  # type: Optional[NumericType]
        bin_size=None,  # type: Optional[NumericType]
        normalize=False,  # type: bool
    ):
        # type: (...) -> None
        super(Histogram, self).__init__()
        if bins is None and bin_start is None and bin_size is None:
            bins = 25
        elif bins is None and (bin_start is None or bin_size is None):
            raise NonRetryableTypeError("Specify both bin start and size")
        elif bins is not None and (bin_start is not None or bin_size is not None):
            raise NonRetryableTypeError("Specify the number of bins OR the start/size")
        self._bins = bins
        self._bin_start = float(bin_start) if bin_start is not None else None
        self._bin_size = float(bin_size) if bin_size is not None else None
        self._normalize = normalize

    def _process_points(self, points):
        # type: (Iterable[NumericType]) -> List[HistogramPointType]
        processed = []  # type: List[NumericType]
        for item in points:
            if isinstance(item, (int, float)):
                if self._bin_start is None or item >= self._bin_start:
                    processed.append(item)
                else:
                    raise NonRetryableTypeError(
                        "All points must be bigger than bin_start"
                    )
            else:
                raise NonRetryableTypeError("All points must be numbers")
        processed.sort()

        if self._bins is not None:
            # pyre-fixme[6]: Expected `float` for 1st param but got `Optional[int]`.
            bin_size = (float(processed[-1]) - float(processed[0])) / self._bins
            bin_start = float(processed[0])
        elif self._bin_size is not None and self._bin_start is not None:
            # to make the the typehinting happy - we alreay ensured either bins
            # or start + size
            bin_size = self._bin_size
            bin_start = self._bin_start

        bins = []  # type: List[Tuple[float, float]]
        bin_count = 0  # type: float
        for point in processed:
            # pyre-fixme[18]: Global name `bin_start` is undefined.
            # pyre-fixme[18]: Global name `bin_size` is undefined.
            while bin_start + bin_size < point:
                bins.append((bin_start, bin_count))
                bin_start += bin_size
                bin_count = 0

            if self._normalize:
                bin_count += 1.0 / len(processed)
            else:
                bin_count += 1

        bins.append((bin_start, bin_count))
        return bins
