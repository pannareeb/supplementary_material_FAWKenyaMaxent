import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json
import geopandas as gpd
import os
import time
from glob import glob
from sklearn import metrics


def compare_image_output(base_dir, keyword, n_cols , saveplot = False, savefolder = None, 
                         subdir_select = None, namefile_select = None, sortlength = False, addname = '',
                         labelmode = False):
  # base_dir = '/content/drive/My Drive/Colab Notebooks/FAW_climate_project/12_files_forMaxEnt_6JUL/ModelOut/ModelOut15JUL/'

  png_files = sorted(glob(os.path.join(base_dir, f'**/*{keyword}*.png'), recursive=True))
  tiff_files = sorted(glob(os.path.join(base_dir, f'**/*{keyword}*.tiff'), recursive=True))
  image_files = png_files + tiff_files
  if subdir_select is not None:
    joinname = '_'.join(subdir_select)
    png_files = []
    tiff_files = []
    for sub in subdir_select:
      pattern = os.path.join(base_dir, f'**/*{sub}*/*{keyword}*.png')
      png_files.extend(glob(pattern, recursive=True))
      pattern = os.path.join(base_dir, f'**/*{sub}*/*{keyword}*.tiff')
      tiff_files.extend(glob(pattern, recursive=True))
      image_files = png_files + tiff_files
      image_files = sorted(image_files)
      print('sorted')
  if namefile_select is not None:
    image_files = [f for f in image_files if namefile_select in f]
  if sortlength:
    image_files = sorted(image_files, key=lambda x: (len(x), x))

  print(f"Found {len(png_files)} .png files and {len(tiff_files)} .tiff files with {keyword} in the selected subdir (if specified)':")
  for image_file in image_files:
    print(image_file.split('/')[-1])

  # n_cols = 4
  n_rows = (len(image_files) + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
  axes = axes.flatten()  # Flatten the 2D array of axes

  for i, image_file in enumerate(image_files):
      img = plt.imread(image_file)
      axes[i].imshow(img)
      # axes[i].set_title(os.path.basename(png_file), fontsize=8)
      axes[i].axis('off')
  if labelmode:
    panel_labels = [f"M{row}{col}" for row in range(1, n_rows+1) for col in range(1, n_cols+1)]
    for i, ax in enumerate(axes):
      ax.text(0.2, 0.2, panel_labels[i], # x, y (axes coords)
          transform=ax.transAxes, fontsize=30, va="top", ha="left")


  # Hide any unused subplots
  for j in range(i + 1, len(axes)):
      axes[j].axis('off')

  plt.tight_layout()
  plt.show()
  if saveplot:
    if subdir_select is not None:
      if savefolder is not None:
        fig.savefig(f'{savefolder}/combined_{joinname}_{keyword}{addname}.png', dpi=300, bbox_inches='tight')
        print(f'save image with {subdir_select} and {keyword}{addname} name in "{savefolder}" folder')
      else:
        fig.savefig(f'{base_dir}/combined_{joinname}_{keyword}{addname}.png', dpi=300, bbox_inches='tight')
        print(f'save image with {subdir_select} and {keyword}{addname} name in base folder')
      # fig.savefig(f'{base_dir}/combined_{joinname}_{keyword}.svg')
    else:
      if savefolder is not None:
        fig.savefig(f'{savefolder}/combined_{keyword}{addname}.png', dpi=300, bbox_inches='tight')
        print(f'save image with {keyword}{addname} name in "{savefolder}" folder')
      else:
        fig.savefig(f'{base_dir}/combined_{keyword}{addname}.png', dpi=300, bbox_inches='tight')
        print(f'save image with {keyword}{addname} name in base folder')
      
      # fig.savefig(f"{base_dir}/combined_{keyword}.svg",bbox_inches="tight")
      
      
def compare_png_output(base_dir, keyword, n_cols , saveplot = False, subdir_select = None, namefile_select = None, sortlength = False):
  # base_dir = '/content/drive/My Drive/Colab Notebooks/FAW_climate_project/12_files_forMaxEnt_6JUL/ModelOut/ModelOut15JUL/'

  png_files = sorted(glob(os.path.join(base_dir, f'**/*{keyword}*.png'), recursive=True))
  if subdir_select is not None:
    joinname = '_'.join(subdir_select)
    png_files = []
    for sub in subdir_select:
      pattern = os.path.join(base_dir, f'**/*{sub}*/*{keyword}*.png')
      png_files.extend(glob(pattern, recursive=True))
      png_files = sorted(png_files)
      print('sorted')
  if namefile_select is not None:
    png_files = [f for f in png_files if namefile_select in f]
  if sortlength:
    png_files = sorted(png_files, key=lambda x: (len(x), x))

  print(f"Found {len(png_files)} .png files ending with {keyword} .png in the selected subdir (if specified)':")
  for png_file in png_files:
    print(png_file.split('/')[-1])

  # n_cols = 4
  n_rows = (len(png_files) + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
  axes = axes.flatten()  # Flatten the 2D array of axes

  for i, png_file in enumerate(png_files):
      img = plt.imread(png_file)
      axes[i].imshow(img)
      # axes[i].set_title(os.path.basename(png_file), fontsize=8)
      axes[i].axis('off')

  # Hide any unused subplots
  for j in range(i + 1, len(axes)):
      axes[j].axis('off')

  plt.tight_layout()
  plt.show()
  if saveplot:
    if subdir_select is not None:
      fig.savefig(f'{base_dir}/combined_{joinname}_{keyword}.png', dpi=300, bbox_inches='tight')
    else:
      fig.savefig(f'{base_dir}/combined_{keyword}.png', dpi=300, bbox_inches='tight')



def check_raster_properties(raster_path):
    try:
        with rasterio.open(raster_path) as src:
            print(f"\n--- Checking Raster: {raster_path} ---")

            # get pixel size (resolution)
            pixel_size = src.res
            print(f"Pixel size (width, height): {pixel_size}")

            # check nodata value from metadata
            raster_nodata = src.nodata
            print(f"Raster's NoData value (if set): {src.nodata}, used this value for counting")

            # read the first band with masking applied
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

def read_raster_as_vector(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(float)
        pixel_size = ds.res
        # print(f"Pixel size (width, height): {pixel_size}")
        return arr.flatten()


def compare_text_output(base_dir, keyword, createdf=False, full_line=False, subdir_select = None):
    search_string = keyword
    if subdir_select is not None:
      txt_files = []
      for sub in subdir_select:
        pattern = os.path.join(base_dir, f'**/*{sub}*/*ModelParms.txt')
        txt_files.extend(glob(pattern, recursive=True))
        txt_files = sorted(txt_files)
    else:
      file_pattern = os.path.join(base_dir, '**/*ModelParms.txt')
      txt_files = glob(file_pattern, recursive=True)
      txt_files = sorted(txt_files)

    if createdf:
        txt_df = pd.DataFrame(columns=['File', 'Text'])

    print(f"Found {len(txt_files)} .txt files ending with {keyword} .png in the selected subdir (if specified)':")

    for file_path in sorted(txt_files):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if search_string in line:
                        file_name = os.path.basename(file_path).replace('_ModelParms.txt', '')
                        text_output = line.strip() if full_line else line.strip().split(':')[-1]
                        print(f"{file_name}: {line.strip()}")
                        if createdf:
                            txt_df = pd.concat(
                                [txt_df, pd.DataFrame({'File': [file_name], 'Text': [text_output]})],
                                ignore_index=True
                            )
                        break  # Stop after finding the first occurrence in the file
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return txt_df if createdf else None

def create_feature_identity(x):
  feature_iden_base = x.columns.values
  linear_names = [f'lr_[{f}]' for f in feature_iden_base]
  product_names = [f"pd_[{a}]_[{b}]" for a, b in itertools.combinations(feature_iden_base, 2)]

  nhinge = 10
  hinge_names = []
  # Left hinge names
  for feat in feature_iden_base:
      for t in range(1, nhinge):  # thresholds 1..(nhinge-1)
          hinge_names.append(f"hgl_[{feat}]_t{t}")
  # Right hinge names
  for feat in feature_iden_base:
      for t in range(1, nhinge):  # thresholds 1..(nhinge-1)
          hinge_names.append(f"hgr_[{feat}]_t{t}")

  print('=> Size of linear_names, product_names, hinge_names:', len(linear_names),',',len(product_names), ',',len(hinge_names))
  feature_identity = linear_names + product_names + hinge_names
  print('=> Total:',len(feature_identity))
  return feature_identity

def create_hingethreshold_df(x, nhinge = 10):
  toformhignthresdf = x
  # print('x.describe():')
  # display(toformhignthresdf.describe())
  selfmin = np.min(toformhignthresdf, axis = 0)
  selfmax = np.max(toformhignthresdf, axis = 0)
  hingethreshold_df = pd.DataFrame(np.linspace(selfmin, selfmax,nhinge -1))
  print(f'Determining hinge thresholds with the input nhinge = {nhinge} by splitting min-max range of each variable at {nhinge-1} values')
  print(f'=> thres1 is the min, and thres{nhinge-1} is the max, see hingethreshold_df below')
  # display(hingethreshold_df)
  print('Creating a final fin_hingethreshold_df, set column by labels and row by thres(n)')
  fin_hingethreshold_df = hingethreshold_df.set_axis(toformhignthresdf.columns.values, axis=1)
  fin_hingethreshold_df.index = ["thres" + str(i+1) for i in fin_hingethreshold_df.index]
  return fin_hingethreshold_df
