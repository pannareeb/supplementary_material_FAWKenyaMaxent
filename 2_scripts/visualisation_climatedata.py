
import os
import numpy as np
import rasterio
import rasterio.plot as rioplot
import matplotlib.pyplot as plt
from glob import glob

import os
import numpy as np
import rasterio
import rasterio.plot as rioplot
import matplotlib.pyplot as plt
from glob import glob


def visualise_rasters(
    path,
    scalelabel = None,
    nodata_value=None,
    output_folder=None,
    show_plot=True,
    save_plot=False,
    figsize=(6, 6),
    cmap_name='viridis',
    title_mode='file',      # 'file' or 'folder_file'
    vmin=None,              # <- NEW: colormap minimum
    vmax=None               # <- NEW: colormap maximum
):
    if os.path.isfile(path) and path.endswith('.tif'):
        raster_files = [path]
        print(f"🔹 Single raster file provided.")
    elif os.path.isdir(path):
        pattern = os.path.join(path, '*.tif')
        raster_files = glob(pattern)
        print(f"🔹 Folder provided. Pattern used: {pattern}")
    else:
        raise ValueError("❌ Invalid path: must be a .tif file or a folder containing .tif files.")

    print(f"🗂️ Found {len(raster_files)} raster file(s).")

    if save_plot and output_folder:
        os.makedirs(output_folder, exist_ok=True)

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(0, 0, 0, 0))  # transparent for nodata

    for raster in raster_files:
        print(f"\n📂 Processing: {raster}")
        try:
            with rasterio.open(raster) as src:
                band = src.read(1, masked=True)
                print("   ↪ src.nodata:", src.nodata)

                raster_nodata = src.nodata if src.nodata is not None else nodata_value
                if raster_nodata is None:
                    print(f"⚠️ No NoData value found or given for: {raster}")
                    band_masked = band
                else:
                    band_masked = np.ma.masked_equal(band, raster_nodata)
                # tell min max of the data
                flat_data = band_masked.compressed()
                if flat_data.size > 0:
                    v_0 = flat_data.min()
                    v_2 = np.percentile(flat_data, 2)
                    v_50 = np.percentile(flat_data, 50)
                    v_98 = np.percentile(flat_data, 98)
                    v_100 = flat_data.max()
                    print(f'min: {v_0}, 2%:{v_2}, 50%:{v_50}, 98%:{v_98}, max:{v_100}')

                # Determine title
                file_name = os.path.basename(raster).replace('.tif', '')
                folder_name = os.path.basename(os.path.dirname(raster))
                title = f"{folder_name}/{file_name}" if title_mode == 'folder_file' else file_name

                fig, ax = plt.subplots(figsize=figsize)
                im = ax.imshow(
                    band_masked,
                    cmap=cmap,
                    extent=rioplot.plotting_extent(src),
                    origin='upper',
                    vmin=vmin,
                    vmax=vmax
                )
                if scalelabel is not None:
                    fig.colorbar(im, ax=ax, label= scalelabel)
                else:
                    fig.colorbar(im, ax=ax, label='Pixel Value')

                ax.set_title(title)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                if save_plot and output_folder:
                    out_path = os.path.join(output_folder, f"{title.replace('/', '_')}.png")
                    fig.savefig(out_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Saved plot to: {out_path}")

                if show_plot:
                    plt.show()
                else:
                    plt.close()

        except Exception as e:
            print(f"❌ Failed to open or plot {raster}: {e}")

    return fig

def visualise_rasters_percent(
    path,
    nodata_value=None,
    output_folder=None,
    show_plot=True,
    save_plot=False,
    figsize=(6, 6),
    cmap_name='viridis',
    title_mode='file',  # 'file' or 'folder_file'
    use_percentile_clip=True,
    clip_range=(2, 98),  # percentiles for vmin and vmax
):
    if os.path.isfile(path) and path.endswith('.tif'):
        raster_files = [path]
        print(f"🔹 Single raster file provided.")
    elif os.path.isdir(path):
        pattern = os.path.join(path, '*.tif')
        raster_files = glob(pattern)
        print(f"🔹 Folder provided. Pattern used: {pattern}")
    else:
        raise ValueError("❌ Invalid path: must be a .tif file or a folder containing .tif files.")

    print(f"🗂️ Found {len(raster_files)} raster file(s).")

    if save_plot and output_folder:
        os.makedirs(output_folder, exist_ok=True)

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(0, 0, 0, 0))  # fully transparent

    for raster in raster_files:
        print(f"\n📂 Processing: {raster}")
        try:
            with rasterio.open(raster) as src:
                band = src.read(1, masked=True)

                print("   ↪ src.nodata:", src.nodata)

                raster_nodata = src.nodata if src.nodata is not None else nodata_value

                if raster_nodata is not None:
                    band_masked = np.ma.masked_equal(band, raster_nodata)
                else:
                    band_masked = band

                # Determine vmin, vmax based on percentiles if enabled
                vmin, vmax = None, None
                if use_percentile_clip:
                    flat_data = band_masked.compressed()
                    if flat_data.size > 0:
                        vmin = np.percentile(flat_data, clip_range[0])
                        vmax = np.percentile(flat_data, clip_range[1])
                        print(f"   🔍 Clipping at {clip_range[0]}–{clip_range[1]} percentiles: vmin={vmin}, vmax={vmax}")
                    else:
                        print("   ⚠️ Skipping clipping: no valid data")

                # Title
                file_name = os.path.basename(raster).replace('.tif', '')
                folder_name = os.path.basename(os.path.dirname(raster))
                title = f"{folder_name}/{file_name}" if title_mode == 'folder_file' else file_name

                fig, ax = plt.subplots(figsize=figsize)
                im = ax.imshow(
                    band_masked,
                    cmap=cmap,
                    extent=rioplot.plotting_extent(src),
                    origin='upper',
                    vmin=vmin,
                    vmax=vmax
                )
                fig.colorbar(im, ax=ax, label='Pixel Value')
                ax.set_title(title)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                if save_plot and output_folder:
                    out_path = os.path.join(output_folder, f"{title.replace('/', '_')}.png")
                    fig.savefig(out_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Saved plot to: {out_path}")

                if show_plot:
                    plt.show()
                else:
                    plt.close()

        except Exception as e:
            print(f"❌ Failed to open or plot {raster}: {e}")

    return fig


def visualise_rasters_lim(
    folder_path,
    nodata_value=None,
    output_folder=None,
    show_plot=True,
    save_plot=False,
    figsize=(6, 6),
    cmap_name='viridis',
    max_files=None):  # <- new

    pattern = os.path.join(folder_path, '*.tif')
    raster_files = sorted(glob(pattern))
    if max_files is not None:
        raster_files = raster_files[:max_files]

    if save_plot and output_folder:
        os.makedirs(output_folder, exist_ok=True)

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(0, 0, 0, 0))

    print("Pattern used:", pattern)
    total = len(sorted(glob(pattern)))
    print(f"Found {total} files; processing {len(raster_files)}.")

    for raster in raster_files:
        print(f"\nProcessing: {raster}")
        try:
            with rasterio.open(raster) as src:
                band = src.read(1, masked=True)

                print("raster src.nodata", src.nodata)
                raster_nodata = src.nodata if src.nodata is not None else nodata_value

                if raster_nodata is None:
                    print(f"⚠️ No NoData value found or given for: {raster}")
                    band_masked = band
                else:
                    band_masked = np.ma.masked_equal(band, raster_nodata)

                fig, ax = plt.subplots(figsize=figsize)
                im = ax.imshow(
                    band_masked,
                    cmap=cmap,
                    extent=rioplot.plotting_extent(src),
                    origin='upper'
                )
                fig.colorbar(im, ax=ax, label='Pixel Value')

                title = os.path.basename(raster).replace('.tif', '')
                ax.set_title(title)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                if save_plot and output_folder:
                    out_path = os.path.join(output_folder, f"{title}.png")
                    fig.savefig(out_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Saved plot to: {out_path}")

                if show_plot:
                    plt.show()
                else:
                    plt.close()

        except Exception as e:
            print(f"❌ Failed to open or plot {raster}: {e}")


def get_raster_stats(raster_path, custom_percentage = None, nodata_value=None):
    try:
        with rasterio.open(raster_path) as src:
            band = src.read(1, masked=True)

            # Use src.nodata if nodata_value not given
            nodata = src.nodata if src.nodata is not None else nodata_value
            if nodata is not None:
                band = np.ma.masked_equal(band, nodata)

            values = band.compressed()  # remove masked

            if values.size == 0:
                raise ValueError("No valid data in raster")

            stats = {
                'file': os.path.basename(raster_path),
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'range': np.max(values) - np.min(values),
                'p2': np.percentile(values, 2),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'p98': np.percentile(values, 98),
                'n': len(values),
            }
            if custom_percentage is not None:
                stats.update({f'p{p}': np.percentile(values, p) for p in custom_percentage})
            else:
                stats = stats
            return stats
    except Exception as e:
        print(f"❌ Error reading {raster_path}: {e}")
        return {'file': os.path.basename(raster_path), 'error': str(e)}
def get_raster_data_and_stats(raster_path, nodata_value=None):
    try:
        with rasterio.open(raster_path) as src:
            band = src.read(1, masked=True)

            nodata = src.nodata if src.nodata is not None else nodata_value
            if nodata is not None:
                band = np.ma.masked_equal(band, nodata)

            values = band.compressed()  # Flatten and drop nodata

            if values.size == 0:
                raise ValueError("No valid data in raster")

            stats = {
                'file': os.path.basename(raster_path),
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'range': np.ptp(values),
                'p2': np.percentile(values, 2),
                'p98': np.percentile(values, 98),
                'n': len(values),
            }

            return stats, values

    except Exception as e:
        print(f"❌ Error reading {raster_path}: {e}")
        return {'file': os.path.basename(raster_path), 'error': str(e)}, np.array([])

