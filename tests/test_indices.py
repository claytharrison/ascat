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

    def test_latlon2cellid(self):
        pass

    def test_cellid2latlon(self):
        pass

    def test_cell_centers(self):
        pass

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

    def test_latlon2cellid(self):
        pass

    def test_cellid2latlon(self):
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

    def test_latlon2cellid(self):
        pass

    def test_cellid2latlon(self):
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
