# Copyright (c) 2024, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import multiprocessing as mp
from functools import partial
from pathlib import Path

from tqdm import tqdm

import xarray as xr
import numpy as np
import dask.array as da


from ascat.file_handling import MultiFileHandler
from ascat.read_native.product_info import cell_io_catalog
from ascat.read_native.product_info import grid_cache
from ascat.utils import get_grid_gpis
from ascat.utils import append_to_netcdf
from ascat.utils import create_variable_encodings


class RaggedArrayCell:
    """
    Class to read and merge ragged array cell files.
    """
    def __init__(self, filename, data=None, chunks=1_000_000):
        self.filename = filename
        self.ds = data
        self.chunks = chunks

    def read(self, date_range=None, valid_gpis=None, lookup_vector=None, **kwargs):
        preprocessor = kwargs.pop("preprocessor", False)
        ds = xr.open_dataset(self.filename, **kwargs)
        if preprocessor:
            ds = preprocessor(ds)

        ds = self._ensure_obs(ds)
        ds = ds.chunk({"obs": self.chunks})
        if lookup_vector is not None:
            ds = self._trim_to_gpis(ds, lookup_vector=lookup_vector)
        elif valid_gpis is not None:
            ds = self._trim_to_gpis(ds, gpis=valid_gpis)
        ds = self._ensure_indexed(ds)
        if date_range is not None:
            ds = self._trim_var_range(ds, "time", *date_range)
        # should I do it this way or just return the ds without having it be a class attribute?
        self.ds = ds
        return self.ds

    # def read_period(self, dt_start, dt_end):
    #     data = self.read()
    #     return data
    @staticmethod
    def _indexed_or_contiguous(ds):
        if "locationIndex" in ds:
            return "indexed"
        return "contiguous"

    @staticmethod
    def _trim_var_range(ds, var_name, var_min, var_max, end_inclusive=False):
        # if var_name in ds:
        if end_inclusive:
            mask = (ds[var_name] >= var_min) & (ds[var_name] <= var_max)
        else:
            mask = (ds[var_name] >= var_min) & (ds[var_name] < var_max)
        return ds.sel(obs=mask.compute())

    @staticmethod
    def _ensure_obs(ds):
        # basic heuristic - if obs isn't present, assume it's instead "time"
        if "obs" not in ds.dims:
            ds = ds.rename_dims({"time": "obs"})
        # other possible heuristics:
        # - if neither "obs" nor "time" is present, assume the obs dim is the one that's
        #  not "locations".
        return ds

    def _ensure_indexed(self, ds):
        """
        Convert a contiguous dataset to indexed dataset,
        if necessary. Indexed datasets pass through.

        Ragged array type is determined by the presence of
        either a "row_size" or "locationIndex" variable
        (for contiguous and indexed arrays, respectively).

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset in indexed or contiguous ragged array format.

        Returns
        -------
        xarray.Dataset
            Dataset in indexed ragged array format.
        """
        if ds is None or "locationIndex" in ds.data_vars:
            return ds

        # remove nans
        row_size = np.where(ds["row_size"].data > 0, ds["row_size"].data,
                            0)

        locationIndex = np.repeat(np.arange(row_size.size), row_size)
        ds["locationIndex"] = ("obs", locationIndex)
        ds = ds.drop_vars(["row_size"])

        # return ds[self._var_order(ds)]
        # we're not going to ensure var_order anymore until it's found to be necessary
        return ds

    @staticmethod
    def _ensure_contiguous(ds):
        """
        Convert an indexed dataset to contiguous dataset,
        if necessary. Contiguous datasets pass through.

        Ragged array type is determined by the presence of
        either a "row_size" or "locationIndex" variable
        (for contiguous and indexed arrays, respectively).

        Parameters
        ----------
        ds : xarray.Dataset, Path
            Dataset in indexed or contiguous ragged array format.

        Returns
        -------
        xarray.Dataset
            Dataset in contiguous ragged array format.
        """
        if ds is None or "row_size" in ds.data_vars:
            return ds

        if not ds.chunks:
            ds = ds.chunk({"obs": 1_000_000})

        ds = ds.sortby(["locationIndex", "time"])
        idxs, sizes = np.unique(ds.locationIndex.values, return_counts=True)
        row_size = np.zeros_like(ds.location_id.data)
        row_size[idxs] = sizes
        ds["row_size"] = ("locations", row_size)
        ds = ds.drop_vars(["locationIndex"])
        return ds

    @staticmethod
    def _only_locations(ds):
        """Return a dataset with only the variables that aren't in the obs-dimension.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset with only the locations-dimensional variables.
        """
        return ds[[
            var
            for var in ds.variables
            if "obs" not in ds[var].dims
            and var not in ["row_size", "locationIndex"]
        ]]

    def merge(self, data):
        """
        Merge datasets with different locations dimensions.

        Parameters
        ----------
        data : list of xarray.Dataset
            Datasets to merge.

        Returns
        -------
        xarray.Dataset
            Merged dataset.
        """
        if data == []:
            return None

        # need a way to short-circuit this if the datasets have all the same location_ids?
        location_vars, location_sorter = self._location_vars_from_ds_list(data)

        merged_ds = xr.combine_nested(
            [self._preprocess(ds, location_vars, location_sorter)
             for ds in data],
            concat_dim="obs",
            data_vars="minimal",
            coords="minimal",
            combine_attrs="drop_conflicts",
        )

        # # Move these elsewhere?
        # merged_ds = trim_dates(merged_ds, date_range)
        # merged_ds = self._trim_var_range(merged_ds, "time", *date_range)
        # merged_ds = self._trim_to_gpis(merged_ds, valid_gpis)

        return merged_ds

    def _location_vars_from_ds_list(self, data):
        # if all datasets have same location_id, we can just return the first one
        locs_merged = xr.combine_nested(
            [self._only_locations(ds) for ds in data], concat_dim="locations"
        )

        _, idxs = np.unique(
            locs_merged["location_id"], return_index=True
        )

        location_vars = {
            var: locs_merged[var][idxs]
            for var in locs_merged.variables
        }

        location_sorter = np.argsort(location_vars["location_id"].values)

        locs_merged.close()

        return location_vars, location_sorter

    @staticmethod
    def _preprocess(ds, location_vars, location_sorter):
        """Pre-processing to be done on a component dataset so it can be merged with others.

        Assumes `ds` is an indexed ragged array. (Re)-calculates the `locationIndex`
        values for `ds` with respect to the `location_id` variable for the merged
        dataset, which may include locations not present in `ds`.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        location_vars : dict
            Dictionary of ordered location variable DataArrays for the merged data.
        location_sorter : numpy.ndarray
            Result of `np.argsort(location_vars["location_id"])`, used to calculate
            the `locationIndex` variable. Calculated outside this function to avoid
            re-calculating it for every dataset being merged.

        Returns
        -------
        xarray.Dataset
            Dataset with pre-processing applied.
        """
        # First, we need to calculate the locationIndex variable, based
        # on the location_id variable that will go on the final merged dataset.
        # This should have been stored in self.location_vars["location_id"] at some
        # point before reaching this function, along with all the other
        # locations-dimensional variables in the combined dataset.

        # if "locations" is in the dataset dimensions, then we have
        # a multi-location dataset.
        if "locations" in ds.dims:
            # TODO maybe we can speed this up with a sel?
            ds = ds.dropna(dim="locations", subset=["location_id"])
            locationIndex = location_sorter[np.searchsorted(
                location_vars["location_id"].values,
                ds["location_id"].values[ds["locationIndex"]],
                sorter=location_sorter,
            )]
            ds = ds.drop_dims("locations")

        # if not, we just have a single location, and logic is different
        else:
            locationIndex = location_sorter[np.searchsorted(
                location_vars["location_id"].values,
                np.repeat(ds["location_id"].values, ds["locationIndex"].size),
                sorter=location_sorter,
            )]

        ds["locationIndex"] = ("obs", locationIndex)

        # Next, we put the locations-dimensional variables on the dataset,
        # and set them as coordinates.
        for var, var_data in location_vars.items():
            ds[var] = ("locations", var_data.values)
        ds = ds.set_coords(["lon", "lat", "alt", "time"])

        try:
            # Need to reset the time index if it's already there, but I can't
            # figure out how to test if time is already an index except like this
            ds = ds.reset_index("time")
        except ValueError:
            pass

        return ds

    def _trim_to_gpis(self, ds, gpis=None, lookup_vector=None):
        """Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return
        if gpis is None and lookup_vector is None:
            return ds
        if len(gpis) == 0 and (lookup_vector is None or len(lookup_vector)==0):
            return ds
        if self._indexed_or_contiguous(ds) == "indexed":
            if gpis is None:
                ds_location_ids = ds["location_id"].data[ds["locationIndex"].data]
                obs_idx = lookup_vector[ds_location_ids]
                locations_idx = np.unique(ds["locationIndex"][obs_idx])

                # then trim out any gpis in the dataset not in gpis
                ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
                new_locationIndex = np.searchsorted(ds["location_id"].data,
                                                    ds_location_ids[obs_idx])
                # and add the new locationIndex
                ds["locationIndex"] = ("obs", new_locationIndex)

            else:
                # first trim out any gpis not in the dataset from the gpi list
                gpis = np.intersect1d(gpis, ds["location_id"].values, assume_unique=True)

                # this is a list of the locationIndex values that correspond to the gpis we're keeping
                locations_idx = np.searchsorted(ds["location_id"].values, gpis)
                # this is the indices of the observations that have any of those locationIndex values
                obs_idx = np.isin(ds["locationIndex"], locations_idx)

                # now we need to figure out what the new locationIndex vector will be once we drop all the other location_ids
                old_locationIndex = ds["locationIndex"].values
                new_locationIndex = np.searchsorted(
                    locations_idx,
                    old_locationIndex[np.isin(old_locationIndex, locations_idx)]
                )

                # then trim out any gpis in the dataset not in gpis
                ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
                # and add the new locationIndex
                ds["locationIndex"] = ("obs", new_locationIndex)
        if self._indexed_or_contiguous(ds) == "contiguous":
            if len(gpis) == 1:
                idx = np.where(ds.location_id==gpis[0])[0][0]
                start = int(ds.row_size.isel(locations=slice(0,idx)).sum().values)
                end = int(start + ds.row_size.isel(locations=idx).values)
                return ds.isel(obs=slice(start, end), locations=idx)
            else:
                if lookup_vector is None:
                    idxs = np.where(np.isin(timeseries.location_id, gpis))[0]
                else:
                    idxs = np.where(
                        lookup_vector[np.repeat(timeseries.location_id, timeseries.row_size)]
                    )[0]
                if idxs.size > 0:
                    starts = [int(timeseries.row_size.isel(locations=slice(0,i)).sum().values)
                                for i in idxs]
                    ends = [int(start + timeseries.row_size.isel(locations=i).values)
                            for start, i in zip(starts, idxs)]
                    obs = np.concatenate([range(start, end) for start, end in zip(starts, ends)])
                    locations = [i for i in idxs]
                    return timeseries.isel(obs=obs, locations=locations)

        return ds

    def write(self,
              filename=None,
              ra_type="indexed",
              mode="w",
              postprocessor=None,
              **kwargs):
        """
        Write data to a netCDF file.

        Parameters
        ----------
        filename : str
            Output filename.
        ra_type : str, optional
            Type of ragged array to write. Default is "contiguous".
        **kwargs : dict
            Additional keyword arguments passed to xarray.to_netcdf().
        """
        filename = filename or self.filename

        if ra_type not in ["contiguous", "indexed"]:
            raise ValueError("ra_type must be 'contiguous' or 'indexed'")
        out_ds = self.ds
        if ra_type == "contiguous":
            out_ds = self._ensure_contiguous(out_ds)

        if postprocessor is not None:
            out_ds = postprocessor(out_ds)

        # out_ds = out_ds[self._var_order(out_ds)]

        # custom_variable_attrs = self._kwargs.get(
        #     "attributes", None) or self.custom_variable_attrs
        # custom_global_attrs = self._kwargs.get(
        #     "global_attributes", None) or self.custom_global_attrs
        # out_ds = self._set_attributes(out_ds, custom_variable_attrs,
        #                               custom_global_attrs)

        # custom_variable_encodings = kwargs.pop(
        #     "encoding", None) or self.custom_variable_encodings
        out_encoding = kwargs.pop("encoding", {})
        out_encoding = create_variable_encodings(out_ds, out_encoding)
        #
        out_ds.encoding["unlimited_dims"] = ["obs"]

        for var, var_encoding in out_encoding.items():
            if "_FillValue" in var_encoding and "_FillValue" in out_ds[
                    var].attrs:
                del out_ds[var].attrs["_FillValue"]

        if mode == "a" and ra_type == "indexed":
            if not Path(filename).exists():
                out_ds.to_netcdf(filename, **kwargs)
            else:
                append_to_netcdf(filename, out_ds, unlimited_dim="obs")
            return

        out_ds.to_netcdf(filename,
                         encoding=out_encoding,
                         **kwargs)


class OrthoMultiCell:
    """
    Class to read and merge orthomulti cell files.
    """
    def __init__(self, filename, chunks=None):
        self.filename = filename
        if chunks is None:
            chunks = {"time": 1000, "locations": 1000}
        self.chunks = chunks
        self.ds = None

    def read(self, date_range=None, valid_gpis=None, lookup_vector=None, **kwargs):
        if self.ds is not None:
            self.ds.close()
        ds = xr.open_dataset(self.filename, **kwargs)
        # ds = ds.set_index(locations="location_id")
        ds = ds.chunk(self.chunks)
        # do these after merging?
        if date_range is not None:
            ds = ds.sel(time=slice(*date_range))
        if valid_gpis is not None:
            # ds = ds.sel(locations=valid_gpis)
            ds = self._trim_to_gpis(ds, gpis=valid_gpis)
        elif lookup_vector is not None:
            ds = self._trim_to_gpis(ds, lookup_vector=lookup_vector)
        # should I do it this way or just return the ds without having it be a class attribute?
        self.ds = ds
        return self.ds

    def merge(self, data):
        """
        Merge datasets with different locations dimensions.

        Parameters
        ----------
        data : list of xarray.Dataset
            Datasets to merge.

        Returns
        -------
        xarray.Dataset
            Merged dataset.
        """
        if data == []:
            return None

        merged_ds = xr.concat(
            data,
            dim="locations",
            combine_attrs="drop_conflicts",
        )
        return merged_ds

    @staticmethod
    def _trim_to_gpis(ds, gpis=None, lookup_vector=None):
        """Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return None
        if gpis is None and lookup_vector is None:
            pass

        elif gpis is None:
            ds_location_ids = ds["location_id"].data
            locs_idx = lookup_vector[ds_location_ids]
            ds = ds.sel(locations=locs_idx)
        else:
            ds = ds.where(ds["location_id"].isin(gpis).compute(), drop=True)

        return ds


class CellGridFiles(MultiFileHandler):
    """
    Managing pygeogrid-ed files with a cell number in the filename.
    """
    def __init__(
        self,
        root_path,
        cls,
        fn_templ,
        sf_templ,
        grid_name,
        cls_kwargs=None,
        err=True,
        fn_read_fmt=None,
        sf_read_fmt=None,
        fn_write_fmt=None,
        sf_write_fmt=None,
        fmt_kwargs=None,
        read_kwargs=None,
        cache_size=0,
    ):
        """
        Initialize CellGridFiles class.

        Parameters
        ----------
        root_path : str
            Root path.
        cls : class
            Class reading/writing files.
        fn_templ : str
            Filename template (e.g. "{date}_ascat.nc").
        sf_templ : dict, optional
            Subfolder template defined as dictionary (default: None).
        grid_name : str
            Name of the grid used by the files as stored in the grid_cache.
        cls_kwargs : dict, optional
            Class keyword arguments (default: None).
        err : bool, optional
            Set true if a file error should be re-raised instead of
            reporting a warning.
            Default: False
        fn_read_fmt : str or function, optional
            Filename format for read operation.
        sf_read_fmt : str or function, optional
            Subfolder format for read operation.
        fn_write_fmt : str or function, optional
            Filename format for write operation.
        sf_write_fmt : str or function, optional
            Subfolder format for write operation.
        cache_size : int, optional
            Number of files to keep in memory (default=0).
        """
        super().__init__(root_path, cls, fn_templ, sf_templ, cls_kwargs, err,
                         cache_size)

        self.fn_read_fmt = fn_read_fmt
        self.sf_read_fmt = sf_read_fmt
        self.fn_write_fmt = fn_write_fmt
        self.sf_write_fmt = sf_write_fmt
        self.fmt_kwargs = fmt_kwargs or {}
        self.read_kwargs = read_kwargs or {}

        grid_info = grid_cache.fetch_or_store(grid_name)
        self.grid_name = grid_name
        self.grid = grid_info["grid"]
        if (grid_info["attrs"] is not None) and ("grid_sampling_km" in grid_info["attrs"]):
            self.grid_sampling_km = grid_info["attrs"]["grid_sampling_km"]
        else:
            self.grid_sampling_km = None

    @classmethod
    def from_product_id(cls, root_path, product_id, **kwargs):
        """
        Create a new CellFiles object from a product_id.
        """
        product_id = product_id.upper()
        if product_id in cell_io_catalog:
            product_class = cell_io_catalog[product_id]
            return cls.from_product_class(root_path, product_class, **kwargs)
        error_str = f"Product {product_id} not recognized. Valid products are"
        error_str += f" {', '.join(cell_io_catalog.keys())}."
        raise ValueError(error_str)

    def _fmt(self, *fmt_args, **fmt_kwargs):
        """
        Format filenames/filepaths.
        """
        if callable(self.fn_read_fmt):
            fn_read_fmt = self.fn_read_fmt(*fmt_args, **fmt_kwargs)
        else:
            fn_read_fmt = self.fn_read_fmt

        if callable(self.sf_read_fmt):
            sf_read_fmt = self.sf_read_fmt(*fmt_args, **fmt_kwargs)
        else:
            sf_read_fmt = self.sf_read_fmt

        if callable(self.fn_write_fmt):
            fn_write_fmt = self.fn_write_fmt(*fmt_args, **fmt_kwargs)
        else:
            fn_write_fmt = self.fn_write_fmt

        if callable(self.sf_write_fmt):
            sf_write_fmt = self.sf_write_fmt(*fmt_args, **fmt_kwargs)
        else:
            sf_write_fmt = self.sf_write_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _merge_data(self, data):
        """
        Merge datasets after reading area. Needs to be overwritten
        by child class, otherwise data is returned as is.

        Parameters
        ----------
        data : list
            Data.

        Returns
        -------
        data : list
            Merged data.
        """
        return self.fid.merge(data)

    # we'll get rid of search_period and read_period and just have search/read
    # Then we can have several arguments that will allow us to search by different criteria

    def spatial_search(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
            # mask_and_scale=True,
            # date_range=None,
            # **kwargs,
            # timestamp,
            # search_date_fmt="%Y%m%d*",
            # date_field="date",
            # date_field_fmt="%Y%m%d",
            # return_date=False
            fmt_kwargs=None,
    ):
        """
        Search files for cells matching a spatial criterion.

        Parameters
        ----------
        cell : int or list of int
            Grid cell number to read.
        location_id : int or list of int
            Location id.
        coords : tuple of numeric or tuple of iterable of numeric
            Tuple of (lon, lat) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        fmt_kwargs = fmt_kwargs or self.fmt_kwargs
        if cell is not None:
            # guarantee cell is a list
            matched_cells = cell
            if not isinstance(matched_cells, list):
                matched_cells = [matched_cells]
        elif location_id is not None:
            # guarantee location_id is a list
            if not isinstance(location_id, list):
                location_id = [location_id]
            matched_cells = self._cells_for_location_id(location_id)
        elif coords is not None:
            matched_cells = self._cells_for_coords(coords)
        elif bbox is not None:
            matched_cells = self._cells_for_bbox(bbox)
        elif geom is not None:
            matched_cells = self._cells_for_geom(geom)
        else:
            matched_cells = self.grid.arrcell

        matched_cells = np.unique(matched_cells)
        # self.grid.allpoints

        filenames = []
        for c in matched_cells:
            fn_read_fmt, sf_read_fmt, _, _ = self._fmt(c, **fmt_kwargs)
            filenames += sorted(self.fs.search(fn_read_fmt, sf_read_fmt))

        return filenames

    def _cells_for_location_id(self, location_id):
        """
        Get cells for location_id.

        Parameters
        ----------
        location_id : int
            Location id.

        Returns
        -------
        cells : list of int
            Cells.
        """
        cells = self.grid.gpi2cell(location_id)
        return cells

    def _cells_for_coords(self, coords):
        """
        Get cells for coordinates.

        Parameters
        ----------
        coords : tuple
            Coordinates (lon, lat)

        Returns
        -------
        cells : list of int
            Cells.
        """
        # gpis, _ = self.grid.find_nearest_gpi(*coords)
        gpis = get_grid_gpis(self.grid, coords=coords)
        cells = self._cells_for_location_id(gpis)
        return cells

    def _cells_for_bbox(self, bbox):
        """
        Get cells for bounding box.

        Parameters
        ----------
        bbox : tuple
            Bounding box.

        Returns
        -------
        cells : list of int
            Cells.
        """
        # gpis = self.grid.get_bbox_grid_points(*bbox)
        gpis = get_grid_gpis(self.grid, bbox=bbox)
        cells = self._cells_for_location_id(gpis)
        return cells

    def _cells_for_geom(self, geom):
        """
        Get cells for bounding box.

        Parameters
        ----------
        bbox : tuple
            Bounding box.

        Returns
        -------
        cells : list of int
            Cells.
        """
        gpis = get_grid_gpis(self.grid, geom=geom)
        cells = self._cells_for_location_id(gpis)
        return cells


    def _apply_func_to_file(self,
                            filename,
                            func,
                            out_dir,
                            func_kwargs=None,
                            write_kwargs=None,
                            write_func=None):
        func_kwargs = func_kwargs or {}
        write_kwargs = write_kwargs or {}
        fid = self.cls(Path(self.root_path)/filename)
        ds = fid.read(mask_and_scale=True)
        if func is not None:
            fid.ds = func(ds, **func_kwargs)
        out_filename = out_dir / Path(filename).relative_to(self.root_path)
        if write_func is not None:
            write_func(fid.ds, out_filename, **write_kwargs)
        else:
            fid.write(out_filename, **write_kwargs)
        # out_filename = Path(out_dir)/Path(filename).name
        # ds.to_netcdf(out_filename)
        fid.ds.close()
        ds.close()
        return

    def reprocess(self,
                  out_dir,
                  func,
                  cell=None,
                  location_id=None,
                  coords=None,
                  bbox=None,
                  num_processes=1,
                  write_kwargs=None,
                  write_func=None,
                  **func_kwargs):
        """
        Reprocess all files into a new directory, preserving subdirectory structure,
        by reading them in as xarrays and applying function "func" to them.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_kwargs = write_kwargs or {}

        filenames = self.spatial_search(
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox
        )

        if num_processes == 1:
            for filename in tqdm(filenames):
                self._apply_func_to_file(filename,
                                         func,
                                         out_dir,
                                         func_kwargs,
                                         write_kwargs,
                                         write_func)
        else:
            ctx = mp.get_context("forkserver")
            pool = ctx.Pool(processes=num_processes)
            convert_func = partial(
                self._apply_func_to_file,
                func=func,
                out_dir=out_dir,
                func_kwargs=func_kwargs,
                write_kwargs=write_kwargs,
                write_func=write_func,
            )
            r = list(tqdm(pool.imap_unordered(convert_func,
                                              filenames,
                                              chunksize=2),
                          total=len(filenames)))

            pool.close()
            pool.join()

    def extract(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
            # mask_and_scale=True,
            max_coord_dist=np.inf,
            date_range=None,
            fmt_kwargs=None,
            **kwargs,
    ):
        """
        Read data matching a spatial and temporal criterion.

        Parameters
        ----------
        cell : int or list of int
            Grid cell number to read.
        location_id : int or list of int
            Location id.
        coords : tuple of numeric or tuple of iterable of numeric
            Tuple of (lon, lat) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        max_coord_dist : float
            The maximum distance a coordinate's nearest grid point can be from it to be
            selected.
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        fmt_kwargs = fmt_kwargs or self.fmt_kwargs
        filenames = self.spatial_search(
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
            geom=geom,
            fmt_kwargs=fmt_kwargs,
        )
        if cell is not None:
            valid_gpis = None
            lookup_vector = None
        else:
            valid_gpis, lookup_vector = get_grid_gpis(
                self.grid,
                cell,
                location_id,
                coords,
                bbox,
                geom,
                max_coord_dist,
                return_lookup=True
            )

        data = []

        for filename in filenames:
            self._open(filename)
            d = self.fid.read(
                date_range=date_range,
                valid_gpis=valid_gpis,
                lookup_vector=lookup_vector,
                **{**self.read_kwargs,
                   **kwargs},
            )
            if d is not None:
                data.append(d)

        if data:
            data = self._merge_data(data)
            data.attrs["grid_name"] = self.grid_name
            return data

        return None

    def append_to_disk(self,
                       out_dir,
                       cell=None,
                       location_id=None,
                       coords=None,
                       bbox=None,):
        """
        Append cell grid files to existing cell grid files on disk.

        Parameters
        ----------
        out_dir : str
            Output directory. Assumed that all files in the directory are compatible
            with the files being appended to them (in dimensions, variables, naming, etc.).
        num_processes : int, optional
            Number of processes to use for conversion.
            Default: -1 (use all available cores).
        """
        filenames = self.spatial_search(
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox
        )

        for filename in filenames:
            self._open(filename)
            self.fid.write(out_dir/filename, mode="a", ra_type="indexed")

    def _merge_cell_out(self, cell, out_dir, fmt_kwargs, **write_kwargs):
        data = self.extract(cell=cell, fmt_kwargs=fmt_kwargs)
        data.load()
        if data is not None:
            fid = self.cls(None, data=data)
            filename = self.ft.build_basename(self.fn_read_fmt(cell))
            fid.write(out_dir/filename, **write_kwargs)

    def merge_out(self,
                  out_dir,
                  cells=None,
                  num_processes=1,
                  fmt_kwargs=None,
                  **write_kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cells = cells or self.grid.get_cells()
        fmt_kwargs = fmt_kwargs or self.fmt_kwargs
        if num_processes == 1:
            for cell in tqdm(cells):
                self._merge_cell_out(cell, out_dir, fmt_kwargs, **write_kwargs)
        else:
            ctx = mp.get_context("forkserver")
            # with ctx.Pool(processes=num_processes) as pool:
            pool = ctx.Pool(processes=num_processes)
            _merge_func = partial(
                self._merge_cell_out,
                out_dir=out_dir,
                fmt_kwargs=fmt_kwargs,
                **write_kwargs
            )
            r = list(tqdm(pool.imap_unordered(_merge_func, cells, chunksize=2),
                            total = len(cells)))
            pool.close()
            pool.join()





class RaggedArrayFiles(CellGridFiles):

    @classmethod
    def from_product_class(cls, root_path, product_class, **kwargs):
        # if anything in kwargs, it will overwrite the defaults
        grid_name = kwargs.pop("grid_name", product_class.grid_name)
        sf_templ = kwargs.pop("sf_pattern", product_class.sf_pattern) \
                    or {"sat_str": "{sat}"}
        fn_read_fmt = kwargs.pop("fn_read_fmt", product_class.fn_read_fmt)
        sf_read_fmt = kwargs.pop("sf_read_fmt", product_class.sf_read_fmt)

        init_options = {
            "root_path": root_path,
            "cls": RaggedArrayCell,
            "fn_templ": "{cell_id}.nc",
            "sf_templ": sf_templ,
            "grid_name": grid_name,
            "fn_read_fmt": fn_read_fmt,
            "sf_read_fmt": sf_read_fmt,
        }
        # we want any kwargs to override the defaults from the product class
        init_options = {**init_options, **kwargs}
        return cls(**init_options)

    def convert_dir_to_contiguous(self,
                                  out_dir,
                                  cell=None,
                                  location_id=None,
                                  coords=None,
                                  bbox=None,
                                  num_processes=None):
        """
        Convert a directory of indexed ragged array files to contiguous ragged array files.

        Parameters
        ----------
        out_dir : str
            Output directory.
        num_processes : int, optional
            Number of processes to use for conversion.
            Default: -1 (use all available cores).
        """

        # filenames = self.spatial_search(
        #     cell=cell,
        #     location_id=location_id,
        #     coords=coords,
        #     bbox=bbox
        # )
        # if num_processes == 1:
        #     for filename in filenames:
        #         self._apply_func_to_file(filename, func, out_dir)
        # else:
        #     ctx = mp.get_context("forkserver")
        #     pool = ctx.Pool(processes=num_processes)
        #     convert_func = partial(
        #         self._apply_func_to_file,
        #         func=func,
        #         out_dir=out_dir
        #     )
        #     pool.map(convert_func, filenames)
        #     pool.close()
        #     pool.join()

        self.reprocess(out_dir,
                       None,
                       cell=cell,
                       location_id=location_id,
                       coords=coords,
                       bbox=bbox,
                       num_processes=num_processes,
                       write_kwargs={"ra_type": "contiguous"})

    def extract(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            # geom=None,
            # mask_and_scale=True,
            max_coord_dist=np.inf,
            date_range=None,
            fmt_kwargs=None,
            **kwargs,
    ):
        """
        Read data matching a spatial and temporal criterion.

        Parameters
        ----------
        cell : int or list of int
            Grid cell number to read.
        location_id : int or list of int
            Location id.
        coords : tuple of numeric or tuple of iterable of numeric
            Tuple of (lon, lat) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        max_coord_dist : float
            The maximum distance a coordinate's nearest grid point can be from it to be
            selected.
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        data = super().extract(
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
            max_coord_dist=max_coord_dist,
            date_range=date_range,
            fmt_kwargs=fmt_kwargs,
            **kwargs,
        )
        return data




class OrthoMultiArrayFiles(CellGridFiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_product_class(cls, root_path, product_class):#, all_sats=False):
        grid_name = product_class.grid_name
        sf_templ = {"sat_str": "{sat}"} #if all_sats else None
        # sf_read_fmt = {"sat_str": {"sat": "metop_[abc]"}} #if all_sats else None
        init_options = {
            "root_path": root_path,
            "cls": OrthoMultiCell,
            "fn_templ": "{cell_id}.nc",
            "sf_templ": sf_templ,
            "grid_name": grid_name,
            "fn_read_fmt": product_class.fn_read_fmt,
            "sf_read_fmt": product_class.sf_read_fmt,
        }
        init_options = {**init_options}
        return cls(**init_options)

    # @staticmethod
    # def _fn_read_fmt(cell, sat=None):
    #     return {"cell_id": f"{cell:04d}"}

    # @staticmethod
    # def _sf_read_fmt(cell, sat=None):
    #     if sat is None:
    #         return None
    #     return {"sat_str": {"sat": sat}}
