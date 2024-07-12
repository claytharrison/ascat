#!/usr/bin/env python3

import numpy.typing as npt
import xarray as xr

from ascat.read_native.xarray.indices import GridIndex
from matplotlib import pyplot as plt


@xr.register_dataset_accessor("gridds")
@xr.register_dataarray_accessor("gridds")
class GridAccessor:
    """
    Based heavily on the DGGSAccessor class from xdggs.
    """
    def __init__(self, obj):
        self._obj = obj

        index = None
        name = ""
        for k, idx in obj.xindexes.items():
            if isinstance(idx, GridIndex):
                if index is not None:
                    raise ValueError(
                        "Only one GridIndex per dataset or dataarray is supported"
                    )
                index = idx
                name = k
        self._name = name
        self._index = index

    @property
    def index(self):
        """Returns the GridIndex instance for this Dataset or DataArray.

        Raise a ``ValueError`` if no such index is found.
        """
        if self._index is None:
            raise ValueError("no GridIndex found on this Dataset or DataArray")
        return self._index

    @property
    def coord(self):
        """Returns the indexed DGGS (location ids) coordinate as a DataArray.

        Raise a ``ValueError`` if no such coordinate is found on this Dataset or DataArray.

        """
        if not self._name:
            raise ValueError(
                "no coordinate with a GridIndex found on this Dataset or DataArray"
            )
        return self._obj[self._name]

    def sel_latlon(self, latitude, longitude):
        """Select location ids from latitude/longitude data.

        Parameters
        ----------
        latitude : array-like
            Latitude coordinates (degrees).
        longitude : array-like
            Longitude coordinates (degrees).

        Returns
        -------
        subset
            A new :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`
            with the nearest location_ids to the input latitude/longitude data points.

        """
        grid_indexers = {self._name: self.index._latlon2gpi(latitude, longitude)}
        return self._obj.sel(grid_indexers)

    # def assign_latlon_coords(self) -> xr.Dataset | xr.DataArray:
    #     """Return a new Dataset or DataArray with new "latitude" and "longitude"
    #     coordinates representing the grid cell centers."""

    #     lat_data, lon_data = self.index.cell_centers
    #     return self._obj.assign_coords(
    #         latitude=(self.index._dim, lat_data),
    #         longitude=(self.index._dim, lon_data),
    #     )

    def sel_bbox(self, bbox):
        """Select grid cells from a bounding box.

        Parameters
        ----------
        bbox : tuple
            A tuple of (min_lat, max_lat, min_lon, max_lon) in degrees.

        Returns
        -------
        subset
            A new :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`
            with all cells that intersect the input bounding box.

        """
        grid_indexers = {self._name: self.index._bbox2gpi(bbox)}
        return self._obj.sel(grid_indexers)

    def plot(self, variable, **kwargs):
        """Plot the grid cells."""
        lats, lons = self.index.gpis.T
        return plt.scatter(lons, lats, c=self._obj[variable], **kwargs)
        # return self._obj.plot.scatter(x=lons, y=lats, **kwargs)
        # return self.index.plot(**kwargs)
        #
