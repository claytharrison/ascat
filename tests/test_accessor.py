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

from ascat.read_native import xarray_ext as xae
from ascat.read_native.swath_collection import SwathGridFiles
from ascat.read_native.cell_collection import RaggedArrayFiles
import ascat.read_native.xarray.indices as idxs

from xarray.core.indexes import PandasIndex

import ascat.read_native.xarray.accessor

from flox.xarray import xarray_reduce

class TestGridAccessor(unittest.TestCase):
    def setUp(self):
        # self.tempdir = TemporaryDirectory()
        # self.tempdir_path = Path(self.tempdir.name)
        # gen_dummy_swathfiles(self.tempdir_path)
        self.swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        self.cell_path = "/home/charriso/p14/data-write/USERS/charriso/h121_merged/metop_abc/"
        self.sgf = SwathGridFiles.from_product_id(self.swath_path, "h129")
        self.rgf = RaggedArrayFiles(self.cell_path, product_id="h121_v1.0")


    # def tearDown(self):
    #     # self.tempdir.cleanup()

    def test_init(self):
        pass

    def test_index(self):
        pass

    def test_coord(self):
        pass

    def test_sel_latlon(self):
        pass

    def test_plot(self):
        from matplotlib import pyplot as plt
        import cartopy.crs as ccrs
        first_cell = Path(self.cell_path).rglob("*.nc").__next__()
        cells_ds = xr.open_dataset(first_cell)
        # bounds = (43, 51, 11, 21) #latmin, latmax, lonmin, lonmax
        # dates = (np.datetime64(datetime(2020, 12, 1)), np.datetime64(datetime(2020, 12, 15)))
        # cells_ds = self.rgf.extract(bbox=bounds, date_range=dates)
        cells_ds["location_id"] = cells_ds["location_id"][cells_ds["locationIndex"]]
        cells_ds = (cells_ds
                    .set_coords("location_id")
                    .set_xindex("location_id", idxs.FibGridRaggedArrayIndex, spacing=12.5)
                    )
                    # .set_xindex("time", PandasIndex))

        # subset = cells_ds.sel(location_id=3157185)
        # day = cells_ds.sel(time="2020-12-01")
        day = cells_ds.sel(obs=(cells_ds["time"] > np.datetime64("2020-12-01")) & (cells_ds["time"] < np.datetime64("2020-12-15")))
        print(day)
        # avg_sm = xarray_reduce(day["surface_soil_moisture"], day["time"].dt.hour, func="mean")
        avg_sm = xarray_reduce(day["surface_soil_moisture"], day["location_id"], func="mean")
        print(avg_sm)
        # print(cells_ds.gridds.index)
        # print(cells_ds.gridds.coord)
        # print(subset.gridds.index)
        # print(subset.gridds.coord)
        # print(subset.gridds)
        # print(subset.gridds.index.cell_centers.T)
        # print(day)
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # ax.coastlines()
        # ax.gridlines(draw_labels=True)
        # ax.set_extent([-60, -45, 5, -10])
        # day.gridds.plot("surface_soil_moisture")
        # plt.show()
        # print(cells_ds.gridds.sel_latlon(45, 15))
