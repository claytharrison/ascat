#!/usr/bin/env python3

import unittest
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np
import dask

from pyresample.geometry import SwathDefinition, AreaDefinition
from fibgrid.realization import FibGrid

from pygeogrids.netcdf import load_grid

import ascat.read_native.generate_test_data as gtd

# from ascat.read_native.swath_collection import SwathGridFiles
# from ascat.read_native.cell_collection import RaggedArrayFiles
from ascat.read_native.xarray.indices import GridIndex, FibGridCache, FibGridIndex, FibGridRaggedArrayIndex
import ascat.read_native.xarray.accessor

class TestGridIndex(unittest.TestCase):
    def setUp(self):
        # self.gpis = [0, 0, 1, 1, 1, 2, 4, 4, 5]
        # self.temp_data = [20, 22, 12, 12, 13, 16, 30, 31, -4]
        self.gpis = np.array([0, 1, 2, 4, 5])
        self.temp_data = np.array([20, 13, 16, 31, -4])
        self.dim = "obs"

    def test_init(self):
        gi = GridIndex(self.gpis, self.dim)
        self.assertEqual(gi._dim, self.dim)
        np.testing.assert_array_equal(gi._pd_index.index.values, self.gpis)

    def test_from_variables(self):
        gi = GridIndex(self.gpis, self.dim)
        with self.assertRaises(NotImplementedError):
            gi.from_variables(None, options=None)

    def test_create_variables(self):
        xarray_variable = xr.DataArray(np.random.rand(6), dims=[self.dim])
        gi = GridIndex(self.gpis, self.dim)
        index_var = gi.create_variables(xarray_variable)
        # print(index_var)
        # TODO not sure yet what to assert here, come back to this
        # print(xarray_variable)
        # self.assertEqual

    def test_isel(self):
        pass
        # gi = GridIndex(self.gpis, self.dim)
        # new_gpis = [0, 1, 1, 2, 4, 4, 5]
        # new_gi = gi.isel({self.dim: slice(0, 7, 2)})
        # self.assertEqual(new_gi._pd_index.index.values.tolist(), new_gpis)

    def test_sel(self):
        # doesn't work
        gi = GridIndex(self.gpis, self.dim)
        # new_gpis = [0, 1, 1, 2, 4, 4, 5]
        gi_selection = gi.sel({"placeholder": [2, 4]})
        np.testing.assert_array_equal(gi_selection.dim_indexers[self.dim], np.array([2, 3]))
        np.testing.assert_array_equal(self.gpis[gi_selection.dim_indexers[self.dim]], np.array([2, 4]))

    def test_equals(self):
        gi = GridIndex(self.gpis, self.dim)
        gi2 = GridIndex(self.gpis, self.dim)
        gi3 = GridIndex(self.gpis, "abcdefg")
        gi4 = GridIndex(np.array([9,8,6,7]), self.dim)
        self.assertTrue(gi.equals(gi2))
        self.assertFalse(gi.equals(gi3))
        self.assertFalse(gi.equals(gi4))

    def test_to_pandas_index(self):
        gi._pd_index
        import pandas as pd
        gi = GridIndex(self.gpis, self.dim)
        self.assertTrue(isinstance(gi.to_pandas_index(), pd.Index))


class TestFibGridCache(unittest.TestCase):
    def test_init(self):
        pass

class TestFibGridIndex(unittest.TestCase):

    def setUp(self):
        self.swath_path = "tests/ascat_test_data/hsaf/h129/swaths"

    def test_init(self):
        pass

    def test_from_variables(self):
        pass

    def test_create_variables(self):
        pass

    def test_isel(self):
        pass

    def test_sel(self):
        pass

    def test_replace(self):
        pass

    def test_latlon2gpi(self):
        pass

    def test_gpi2latlon(self):
        pass

    def test_cell_centers(self):
        pass

class TestFibGridRaggedArrayIndex(unittest.TestCase):
    def setUp(self):
        self.cell_path = "tests/ascat_test_data/hsaf/h129/stack_cells"
        # self.rgf = RaggedArrayFiles(self.cell_path, "h129")

    def test_init(self):
        pass

    def test_from_variables(self):
        pass

    def test_create_variables(self):
        pass

    def test_isel(self):
        pass

    def test_sel(self):
        pass

    def test_replace(self):
        pass

    def test_latlon2gpi(self):
        pass

    def test_gpi2latlon(self):
        pass

    def test_cell_centers(self):
        pass

    def test_equals(self):
        pass

    def test_join(self):
        pass

    def test_reindex_like(self):
        first_ds = Path(self.cell_path).rglob("*.nc").__next__()
        first_ds = xr.open_dataset(first_ds)
        first_ds["location_id"] = first_ds["location_id"][first_ds["locationIndex"]]
        first_ds = first_ds.set_coords("location_id").set_xindex("location_id", FibGridRaggedArrayIndex, spacing=6.25)
        # print(first_ds)
        # print(first_ds.gridds.index)
        # print(first_ds.gridds.index.cell_centers.shape)
        # print(first_ds.gridds.index.join(first_ds.gridds.index).cell_centers.shape)
        # print(first_ds.gridds.index.reindex_like(first_ds.gridds.index))
        # print(xarray_reduce(first_ds["surface_soil_moisture"], first_ds["location_id"], func="mean"))
