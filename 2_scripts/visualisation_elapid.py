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
