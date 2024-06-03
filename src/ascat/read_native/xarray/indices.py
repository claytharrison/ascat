#!/usr/bin/env python3
# Everything in this file has been adapted from the source code of xdggs.
# It is intended for demo purposes only at the moment.

from collections.abc import Hashable, Mapping
from typing import Any, Union

import numpy as np
import xarray as xr

from fibgrid.realization import FibGrid
from xarray.indexes import Index, PandasIndex
from xarray.core.indexing import merge_sel_results

# from xdggs.utils import _extract_cell_id_variable, register_dggs


class GridIndex(Index):

    def __init__(self, cell_ids, dim):
        self._dim = dim

        if isinstance(cell_ids, PandasIndex):
            self._pd_index = cell_ids
        else:
            self._pd_index = PandasIndex(cell_ids, dim)

    @classmethod
    def from_variables(cls, variables, *, options):
        raise NotImplementedError()

    def create_variables(self, variables):
        return self._pd_index.create_variables(variables)

    def isel(self, indexers):
        new_pd_index = self._pd_index.isel(indexers)
        if new_pd_index is not None:
            return self._replace(new_pd_index)
        return None

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        return self._pd_index.sel(labels, method=method, tolerance=tolerance)

    def _replace(self, new_pd_index: PandasIndex):
        raise NotImplementedError()

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        """convert latitude / longitude points to cell ids."""
        raise NotImplementedError()

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        """convert cell ids to latitude / longitude (cell centers)."""
        raise NotImplementedError()

    @property
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cellid2latlon(self._pd_index.index.values)

    def equals(self, other):
        return self._pd_index.equals(other._pd_index)

    def join(self, other, how="inner"):
        raise NotImplementedError()

    def reindex_like(self, other):
        raise NotImplementedError()

    def to_pandas_index(self):
        return self._pd_index.index

class FibGridCache:
    def __init__(self):
        self.grids = {}

    def fetch_or_store(self, spacing):
        """
        Fetch a FibGrid object from the cache given a resolution,
        or store a new one.

        Parameters
        ----------
        """
        key = str(spacing)
        if key not in self.grids:
            self.grids[key] = FibGrid(spacing)

        return self.grids[key]

fibgrid_cache = FibGridCache()

class FibGridIndex(GridIndex):

    def __init__(
        self,
        cell_ids,
        dim,
        spacing,
    ):
        super().__init__(cell_ids, dim)
        self._spacing = spacing
        self._fibgrid = fibgrid_cache.fetch_or_store(spacing)

    @classmethod
    def from_variables(
        cls,
        variables,
        *,
        options,
    ):
        _, var = next(iter(variables.items()))
        dim = next(iter(var.dims))

        spacing = var.attrs.get("spacing", options.get("spacing"))
        return cls(var.data, dim, spacing)

    def _replace(self, new_pd_index):
        return type(self)(new_pd_index, self._dim, self._spacing)

    def _latlon2cellid(self, lat, lon):
        # return coordinates_to_cells(lat, lon, self._spacing, radians=False)
        gpi, _ = self._fibgrid.find_nearest_gpi(lon, lat)
        if isinstance(gpi, np.ma.MaskedArray):
            return gpi.compressed()
        return gpi

    def _cellid2latlon(self, cell_ids):
        # return cells_to_coordinates(cell_ids, radians=False)
        lons, lats = self._fibgrid.gpi2lonlat(cell_ids)
        return (np.vstack([lats.compressed(), lons.compressed()]).T)

    def _bbox2cellid(self, bbox):
        gpis = self._fibgrid.get_bbox_grid_points(*bbox)
        if isinstance(gpis, np.ma.MaskedArray):
            return gpis.compressed()
        return gpis

    def _repr_inline_(self, max_width):
        return f"FibGridIndex(spacing={self._spacing})"

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        results = []

        for k, v in labels.items():
            if k == "location_id":
                # special logic
                lookup_vector = np.zeros(self._fibgrid.gpis.max()+1, dtype=bool)
                lookup_vector[v] = True
                idx_locs = lookup_vector[self._pd_index.index.values]
                # idx_locs = np.where(idx_locs)[0]
                results.append(self._pd_index.sel({k: idx_locs}))
            else:
                results.append(self._pd_index.sel({k: v}, method=method, tolerance=tolerance))
        return merge_sel_results(results)

    def join(self, other, how="inner"):
        #simply concatenate them
        index = self._pd_index.index.append(other._pd_index.index)
        coord_dtype = np.result_type(self._pd_index.coord_dtype, other._pd_index.coord_dtype)

        return type(self)(index, self._dim, spacing=self._spacing)

    def reindex_like(self, other):
        lookup_vector = np.zeros(self._fibgrid.gpis.max()+1, dtype=bool)
        lookup_vector[other._pd_index.index.values] = True
        idx_locs = lookup_vector[self._pd_index.index.values]
        return {self._dim: self._pd_index.sel({self._dim: idx_locs})}
        # return type(self)(self._pd_index.sel({self._dim: idx_locs}), self._dim, spacing=self._spacing)


class FibGridRaggedArrayIndex(FibGridIndex):

    def __init__(
        self,
        cell_ids,
        dim,
        spacing,
    ):
        super().__init__(cell_ids, dim, spacing)
        self._ds_location_ids = None

    def _repr_inline_(self, max_width):
        return f"FibGridIndex(spacing={self._spacing})"

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        results = []

        for k, v in labels.items():
            if k == "location_id":
                # special logic
                lookup_vector = np.zeros(self._fibgrid.gpis.max()+1, dtype=bool)
                lookup_vector[v] = True
                idx_locs = lookup_vector[self._pd_index.index.values]
                # idx_locs = np.where(idx_locs)[0]
                results.append(self._pd_index.sel({k: idx_locs}))
            else:
                results.append(self._pd_index.sel({k: v}, method=method, tolerance=tolerance))
        return merge_sel_results(results)

        # return self._pd_index.sel(labels, method=method, tolerance=tolerance)

