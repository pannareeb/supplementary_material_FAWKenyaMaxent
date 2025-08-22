
import ee
import geemap.core as geemap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import json
import geopandas as gpd
import os
import time

from glob import glob
import rasterio
from rasterio.mask import mask

def feature_to_ee_geometry(feature):
  """Converts a GeoJSON Feature to an ee.Geometry object."""
  # Check if the feature has a geometry and if it's not None
  if feature.get('geometry') and feature['geometry'].get('coordinates'):
    return ee.Geometry(feature['geometry'])
  else:
    return None


def gee_export_avg(collection, boundary ,bands_to_export,
                   output_dir='data/climate_data/gee_avg', 
                   method = 'mean', export_scale = 9000, nodata_value = -9999, asctask = False):
  # boundary = ee.FeatureCollection(ee_geometries)

  # # Define the output directory for the raster files
  # output_dir = '11_GEEoutput/Output7JUL_nodata9999_20172019_2'
  os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

  collection_clipped = collection.filterBounds(boundary)
  # Get the list of available band names
  band_names = collection_clipped.first().bandNames().getInfo()

  # Define the scale for exporting (adjust as needed for your resolution)
  export_scale = export_scale # meters
  nodata_value = nodata_value

  # Iterate through each band and export it
  for band_name in bands_to_export:
      if band_name in band_names:
          # Select the specific band from the collection
          if method == 'mean':
            band_image = collection_clipped.select(band_name).mean()
          elif method == 'max':
            band_image = collection_clipped.select(band_name).max()
          elif method == 'min':
            band_image = collection_clipped.select(band_name).min()
          elif method == 'median':
            band_image = collection_clipped.select(band_name).median()
          else:
              raise ValueError("Unsupported method")
          # Unmask the image: masked pixels will be set to -9999
          band_image_unmasked = band_image.unmask(nodata_value)

          # Define export parameters for GeoTIFF
          tif_task = ee.batch.Export.image.toDrive(
              image=band_image_unmasked,
              description=f'{band_name}_raster_tif',
              folder=output_dir,
              fileNamePrefix=f'{band_name}_raster_tif',
              scale=export_scale,
              region=boundary.geometry(),
              fileFormat='GeoTIFF',
              formatOptions={'noData': nodata_value} 
          )

          # Start the GeoTIFF export task
          tif_task.start()
          print(f'Started export task for {band_name} (GeoTIFF). Task ID: {tif_task.id}')

          if asctask:
            # Define export parameters for ASCII Grid (.asc)
            asc_task = ee.batch.Export.image.toDrive(
                image=band_image_unmasked,
                description=f'{band_name}_raster_asc',
                folder=output_dir,
                fileNamePrefix=f'{band_name}_raster_asc',
                scale=export_scale,
                region=boundary.geometry(),
                fileFormat='GeoTIFF', # Export as GeoTIFF first
                formatOptions={
                    'cloudOptimized': False # Not needed for conversion to ASCII
                }
            )

            # Start the GeoTIFF export task (to be converted to ASCII later)
            asc_task.start()
            print(f'Started intermediate export task for {band_name} (ASCII). Task ID: {asc_task.id}')

  # You will need to manually check the Google Drive folder or use ee.data.listTasks()
  # to monitor the progress and completion of these tasks.
  # Once the GeoTIFFs for ASCII conversion are ready, you'll need to download them
  # and use a tool like gdal_translate or a Python library (e.g., rasterio)
  # to convert the GeoTIFF to ASCII Grid (.asc) format.

  print("Export tasks initiated. Check your Google Drive for progress.")



def gee_export_selected_months(collection_id, boundary, bands_to_export, year_month_list,
                           output_dir='data/climate_data/gee_selected_months', export_scale=9000, nodata_value=-9999):
    os.makedirs(output_dir, exist_ok=True)
    collection = ee.ImageCollection(collection_id)
    name = collection_id.split('/')[-1]
    export_scale = export_scale # meters
    nodata_value = nodata_value

    # Iterate through each year-month combination
    for ym in year_month_list:
        year, month = map(int, ym.split('-'))

        # Filter to specific year-month and region
        filtered = collection \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .filter(ee.Filter.calendarRange(month, month, 'month')) \
            .filterBounds(boundary)

        count = filtered.size().getInfo()
        if count != 1:
            print(f"[SKIP] {ym}: found {count} images (expected 1)")
            continue

        image = filtered.first().select(bands_to_export).unmask(nodata_value)
        filename = f'{name}_{year}_{month:02d}'

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
            folder=output_dir,
            fileNamePrefix=filename,
            scale=export_scale,
            region=boundary.geometry(),
            fileFormat='GeoTIFF',
            formatOptions={'noData': nodata_value} 
        )

        task.start()
        print(f"[EXPORTING] {filename} (Task ID: {task.id})")
        time.sleep(2)

# def asc_to_tiff(asc_folder, output_folder):
#     from glob import glob
#     import os
#     import rasterio

#     os.makedirs(output_folder, exist_ok=True)

#     asc_files = glob(os.path.join(asc_folder, '*.asc'))
#     for asc_path in asc_files:
#         try:
#             with rasterio.open(asc_path) as src:
#                 data = src.read(1)
#                 profile = src.profile
#                 profile.update({
#                     'driver': 'GTiff',
#                     'count': 1,
#                     'dtype': data.dtype,
#                     'nodata': src.nodata,
#                     'crs': src.crs if src.crs else None,
#                     'transform': src.transform
#                 })
#                 output_name = os.path.splitext(os.path.basename(asc_path))[0] + '.tif'
#                 output_path = os.path.join(output_folder, output_name)

#                 with rasterio.open(output_path, 'w', **profile) as dst:
#                     dst.write(data, 1)

#             print(f"Saved: {output_path}")
#         except Exception as e:
#             print(f"Error processing {asc_path}: {e}")




def clip_rasters_to_tiff(raster_folder, output_folder, mask_gdf):
    os.makedirs(output_folder, exist_ok=True)
    raster_files = glob(os.path.join(raster_folder, '*.tif'))

    for raster_path in raster_files:
        try:
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                nodata_val = src.nodata if src.nodata is not None else -9999

                # Reproject mask if needed
                if mask_gdf.crs != raster_crs:
                    mask_proj = mask_gdf.to_crs(raster_crs)
                else:
                    mask_proj = mask_gdf

                geoms = [feature["geometry"] for feature in mask_proj.__geo_interface__["features"]]

                out_image, out_transform = mask(src, geoms, crop=True, nodata=nodata_val)

                # If masked array, fill with nodata_val
                if isinstance(out_image, np.ma.MaskedArray):
                    out_image = out_image.filled(nodata_val)

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": nodata_val
                })

            base_name = os.path.splitext(os.path.basename(raster_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_clipped.tif")

            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(out_image)

            print(f"✅ Saved: {output_path}")

        except Exception as e:
            print(f"❌ Error processing {raster_path}: {e}")

    print("All rasters clipped and saved as GeoTIFF!")


def check_raster_properties(raster_path, nodata_value=-9999):
    try:
        with rasterio.open(raster_path) as src:
            print(f"\n--- Checking Raster: {raster_path} ---")

            # Get pixel size (resolution)
            pixel_size = src.res
            print(f"Pixel size (width, height): {pixel_size}")

            # Get nodata value from metadata or fallback
            raster_nodata = src.nodata #if src.nodata is not None else nodata_value
            print(f"Raster's official NoData value (if set): {src.nodata}")
            print(f"Using NoData value for counting: {raster_nodata}")

            # Read the first band with masking applied
            band = src.read(1, masked=True)

            # Total pixels
            total_pixels = band.size
            print(f"Total number of pixels: {total_pixels}")

            # Valid pixels = pixels NOT masked
            valid_pixels = (~band.mask).sum() if np.ma.is_masked(band) else total_pixels
            print(f"Number of valid pixels: {valid_pixels}")

            # Percentage valid pixels
            percentage_valid = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            print(f"Percentage of valid pixels: {percentage_valid:.2f}%")

            print("--- End Check ---")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening raster file {raster_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {raster_path}: {e}")






def split_multiband_rasters_by_date(input_folder, output_root, band_names=None):
    """
    Splits multi-band GeoTIFFs into single-band TIFFs.
    Saves each time step's bands into its own subfolder (e.g. output_root/2021-03/).

    Assumes filenames contain a date token like YYYY_MM or YYYY-MM.

    Parameters:
        input_folder (str): Folder containing multi-band .tif files.
        output_root (str): Root folder where time-specific subfolders will be created.
        band_names (list, optional): List of band names (must match band count).
    """
    os.makedirs(output_root, exist_ok=True)
    raster_paths = glob(os.path.join(input_folder, "*.tif"))

    for raster_path in raster_paths:
        try:
            with rasterio.open(raster_path) as src:
                band_count = src.count

                # Validate or default band names
                if band_names:
                    if len(band_names) != band_count:
                        raise ValueError(f"Expected {band_count} band names, got {len(band_names)}")
                    name_list = band_names
                else:
                    name_list = [f"band{idx}" for idx in range(1, band_count + 1)]

                # Extract base name and infer date (e.g. from 'bio_2021_03.tif' -> '2021-03')
                base_name = os.path.splitext(os.path.basename(raster_path))[0]
                date_token = None
                for part in base_name.split('_'):
                    if part.isdigit() and len(part) == 4:
                        year = part
                    elif part.isdigit() and len(part) == 2:
                        month = part
                if 'year' in locals() and 'month' in locals():
                    date_token = f"{year}-{month}"
                else:
                    raise ValueError(f"Cannot extract YYYY-MM date from filename: {base_name}")

                output_folder = os.path.join(output_root, date_token)
                os.makedirs(output_folder, exist_ok=True)

                # Save each band to its time-stamped folder
                for band_idx, band_name in enumerate(name_list, start=1):
                    band_data = src.read(band_idx)
                    band_profile = src.profile.copy()
                    band_profile.update({
                        'count': 1,
                        'dtype': band_data.dtype,
                        'driver': 'GTiff',
                        'nodata': src.nodata,
                    })

                    output_path = os.path.join(output_folder, f"{band_name}.tif")
                    with rasterio.open(output_path, 'w', **band_profile) as dst:
                        dst.write(band_data, 1)

                    print(f"✅ Saved: {output_path}")

        except Exception as e:
            print(f"❌ Error processing {raster_path}: {e}")

    print("🎉 All multi-band rasters split into time folders!")


import re
import shutil

def group_raster_by_month(input_folder, output_folder):
    """
    Flattens all raster files from subdirectories in input_folder and groups them into
    monthly folders (e.g. '01', '02', ..., '12') based on the MM in the filename pattern *_MM.tif.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get all .tif files recursively
    raster_files = glob(os.path.join(input_folder, '**', '*.tif'), recursive=True)

    for raster_path in raster_files:
        filename = os.path.basename(raster_path)

        # Match the MM (month) from pattern *_MM.tif
        match = re.search(r'_(\d{2})\.tif$', filename)
        if not match:
            print(f"⚠️ Skipping (no month found): {filename}")
            continue

        month = match.group(1)
        month_folder = os.path.join(output_folder, month)
        os.makedirs(month_folder, exist_ok=True)

        # Copy the file into the corresponding month folder
        destination = os.path.join(month_folder, filename)
        shutil.copy2(raster_path, destination)
        print(f"✅ {filename} ➡️ {month}/")

    print("🎉 Grouping complete.")

def set_nodata_value_in_geotiff(input_path, nodata_value=-9999, save_to_folder=None):
    """
    Sets both the metadata and pixel values of NoData in a GeoTIFF to a specified value.
    Can overwrite the original file or save to a new folder.

    Parameters:
    - input_path: str, path to the input GeoTIFF file.
    - nodata_value: numeric, value to set as NoData in the metadata (default -9999).
    - save_to_folder: str or None, if provided, saves the modified file into this folder.
    """
    try:
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            data = src.read()

            # Determine current nodata value
            current_nodata = src.nodata
            if current_nodata is not None:
                # Replace existing NoData values with new one
                data = np.where(data == current_nodata, nodata_value, data)
            else:
                # Use raster mask if nodata not explicitly set
                mask = src.read_masks(1) == 0
                data[:, mask] = nodata_value

            profile.update(nodata=nodata_value)

            # Determine output path
            if save_to_folder:
                os.makedirs(save_to_folder, exist_ok=True)
                filename = os.path.basename(input_path)
                output_path = os.path.join(save_to_folder, filename)
            else:
                output_path = input_path + ".tmp.tif"

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)

        # Replace original file only if not saving to another folder
        if not save_to_folder:
            os.replace(output_path, input_path)
            print(f"✅ NoData value updated and file overwritten: {input_path}")
        else:
            print(f"✅ NoData value updated and saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error while processing {input_path}: {e}")
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)

def mean_rasters_by_keyword_recursive(root_folder, keyword, output_path='mean_raster.tif'):
    # Recursively find all .tif files with the keyword in filename
    raster_files = sorted(glob(os.path.join(root_folder, '**', f'*{keyword}*.tif'), recursive=True))
    
    if len(raster_files) == 0:
        raise FileNotFoundError(f"No rasters with keyword '{keyword}' found in {root_folder} or subfolders.")

    data_stack = []
    ref_profile = None

    for i, raster_path in enumerate(raster_files):
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype('float32')
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            data_stack.append(data)

            if ref_profile is None:
                ref_profile = src.profile.copy()

    # Stack and compute pixel-wise mean
    stacked = np.stack(data_stack)
    mean_data = np.nanmean(stacked, axis=0)

    # Update profile
    ref_profile.update(dtype='float32', count=1, nodata=np.nan)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save output
    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        dst.write(np.nan_to_num(mean_data, nan=ref_profile['nodata']), 1)

    print(f"✅ Recursive mean raster saved to: {output_path}")

# Example usage:
# mean_rasters_by_keyword_recursive('data/climate_data/', keyword='bio1', output_path='bio1_mean_recursive.tif')



# def clip_raster_to_asc(raster_folder, output_folder, mask_gdf):
#     from rasterio.mask import mask
#     from glob import glob
#     import os
#     import rasterio
#     import numpy as np

#     raster_files = glob(os.path.join(raster_folder, '*.tif'))
#     os.makedirs(output_folder, exist_ok=True)

#     for raster_path in raster_files:
#         try:
#             with rasterio.open(raster_path) as src:
#                 raster_crs = src.crs

#                 if mask_gdf.crs != raster_crs:
#                     mask_gdf_proj = mask_gdf.to_crs(raster_crs)
#                 else:
#                     mask_gdf_proj = mask_gdf

#                 geoms = [feature["geometry"] for feature in mask_gdf_proj.__geo_interface__["features"]]

#                 nodata_val = src.nodata if src.nodata is not None else -9999

#                 out_image, out_transform = mask(
#                     src,
#                     geoms,
#                     crop=True,
#                     nodata=nodata_val
#                 )

#                 # Optional: cast to float32 for compatibility with ASCII grids
#                 out_image = out_image.astype(np.float32)

#                 out_meta = src.meta.copy()
#                 out_meta.update({
#                     "driver": "AAIGrid",
#                     "dtype": 'float32',  # consistent with above cast
#                     "crs": src.crs,
#                     "nodata": nodata_val,
#                     "height": out_image.shape[1],
#                     "width": out_image.shape[2],
#                     "transform": out_transform,
#                     "count": out_image.shape[0]
#                 })

#             base_name = os.path.splitext(os.path.basename(raster_path))[0]
#             output_name = f"{base_name}_clipped.asc"
#             output_path = os.path.join(output_folder, output_name)

#             with rasterio.open(output_path, 'w', **out_meta) as dest:
#                 dest.write(out_image)

#             print(f"Clipped raster saved to: {output_path}")

#         except Exception as e:
#             print(f"Error processing {raster_path}: {e}")

#     print("✅ All rasters clipped and saved as .asc files!")


# def asc_to_tiff(asc_folder, output_folder):
#     from glob import glob
#     import os
#     import rasterio

#     os.makedirs(output_folder, exist_ok=True)

#     asc_files = glob(os.path.join(asc_folder, '*.asc'))
#     for asc_path in asc_files:
#         try:
#             with rasterio.open(asc_path) as src:
#                 data = src.read(1)
#                 profile = src.profile
#                 profile.update({
#                     'driver': 'GTiff',
#                     'count': 1,
#                     'dtype': data.dtype,
#                     'nodata': src.nodata,
#                     'crs': src.crs if src.crs else None,
#                     'transform': src.transform
#                 })
#                 output_name = os.path.splitext(os.path.basename(asc_path))[0] + '.tif'
#                 output_path = os.path.join(output_folder, output_name)

#                 with rasterio.open(output_path, 'w', **profile) as dst:
#                     dst.write(data, 1)

#             print(f"Saved: {output_path}")
#         except Exception as e:
#             print(f"Error processing {asc_path}: {e}")

