
import os
import pandas as pd
import geopandas as gpd
import elapid as ela
from glob import glob
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go

import geopandas as gpd
from shapely.geometry import Point

import geojson
from pprint import pprint
import rasterio
from rasterio.mask import mask
import rasterio.plot as rioplot

import elapid as ela
from sklearn import metrics

import geemap.core as geemap
from datetime import datetime




def load_climatetif_to_dict_rec(climate_folder_path):
    """
    Recursively loads all .tif files from a root folder (including subfolders) into a dictionary.
    Keys preserve subfolder context using underscores.
    Values are full file paths.
    """
    raster_files = glob(os.path.join(climate_folder_path, '**', '*.tif'), recursive=True)

    print(f"\nFound {len(raster_files)} raster files in {climate_folder_path} >>")

    raster_dict = {}

    for raster_file in raster_files:
        print(raster_file)
        # Get relative path from root folder
        relative_path = os.path.relpath(raster_file, climate_folder_path)
        # Remove extension and replace os separators with underscore for flat label
        label = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        raster_dict[label] = raster_file

    print("\nLabels generated from raster filenames:")
    for label in raster_dict.keys():
        print(label)

    return raster_dict



def annotate_points_by_time(points_gdf, date_column, raster_base_folder, date_format="%Y-%m"):
    """
    Annotate points using rasters from a matching YYYY-MM folder based on their date.

    Parameters:
        points_gdf (GeoDataFrame): GeoDataFrame with point geometries and a date column.
        date_column (str): Name of the column containing datetime or parseable strings.
        raster_base_folder (str): Path to the folder containing date-named subfolders (e.g. "rasters/2013-01/").
        date_format (str): Format for subfolder names, default is "YYYY-MM".

    Returns:
        GeoDataFrame: Annotated points with environmental data for the correct time step.
    """
    points_gdf = points_gdf.copy()
    # make sure date column is datetime
    points_gdf[date_column] = pd.to_datetime(points_gdf[date_column])
    points_gdf["time_key"] = points_gdf[date_column].dt.strftime(date_format)

    annotated_dfs = []
    # tqdm for progress bar
    for time_key, group in tqdm(points_gdf.groupby("time_key"), desc= f"Annotating by {date_format}"):
        raster_folder = os.path.join(raster_base_folder, time_key)
        print(f"Checking for raster folder: {raster_folder}")
        if not os.path.exists(raster_folder):
            print(f"Warning: No raster folder found for {time_key}")
            continue

        raster_paths = sorted(glob(os.path.join(raster_folder, "*.tif")))
        if not raster_paths:
            print(f"Warning: No rasters in {raster_folder}")
            continue

        # generate labels based on file names (without .tif)
        labels = [os.path.splitext(os.path.basename(p))[0] for p in raster_paths]

        # annotate the group by env rasters
        try:
            annotated = ela.annotate(
                points=group,
                raster_paths=raster_paths,
                labels=labels,
                drop_na=True,
                quiet=True
            )
            annotated_dfs.append(annotated)
            print(f"\nAnnotated {len(group)} points for {date_format} ={time_key}")

        except Exception as e:
            print(f"=> Error annotating {time_key}: {e}")

    # combine all annotated groups (vertical stack, refresh index)
    if annotated_dfs:
        annotated_all = pd.concat(annotated_dfs, ignore_index=True)
        print(f"Total points annotated: {len(annotated_all)}")
        return gpd.GeoDataFrame(annotated_all, geometry='geometry', crs=points_gdf.crs)
    else:
        print("No annotations were performed.")
        return gpd.GeoDataFrame(columns=points_gdf.columns)
        
        
def calculate_kde_metrics(pres, bg, var_name):
    """Calculate overlap % and peak locations for two distributions."""
    import scipy.stats as stats

    pres, bg = pres[~np.isnan(pres)], bg[~np.isnan(bg)]
    kde_p, kde_b = stats.gaussian_kde(pres), stats.gaussian_kde(bg)

    x = np.linspace(min(pres.min(), bg.min()), max(pres.max(), bg.max()), 1000)
    dens_p, dens_b = kde_p(x), kde_b(x)

    result = {
    'var': var_name,
    'overlap_percent': np.trapz(np.minimum(dens_p, dens_b), x) * 100,
    'pres_peak': x[np.argmax(dens_p)],
    'bg_peak': x[np.argmax(dens_b)],
    'peak_diff': abs(x[np.argmax(dens_p)] - x[np.argmax(dens_b)])}
    return result

def pres_back_dis(pres_back_df, name, labels, outputpath = 'maxent_output',saveplot = False, bar = True, pad = 0.2, hspace=0.4, top=0.50):
  """
    Create a histrogram or a kdeplot of all env labels of two groups of points (presence and background) in the same plot
    kdeplot = distributions using kernel density estimation, similar to smoothed histogram.

    Parameters:
        pres_back_df dataframe [presencedata, bgdata]: data of each env variable for both presence and background points. col = each env var
        name str [namepresence,namebg,nameraster] : name for labelling plot
        labels str [env1, env2,...]: labels for each env var
        outputpath: root directory to save plot

    Returns:
        GeoDataFrame: Annotated points with environmental data for the correct time step.
    """
  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]
  titleplot = f'(Presence: {namepresence}\n Background: {namebg} under {nameraster})'

  # create output directory for all out files from this run
  output_model_dir = f'{outputpath}/ModelOut_{namepresence}_{namebg}_{nameraster}'
  os.makedirs(output_model_dir, exist_ok=True) # create the directory if it doesn't exist

  presencetoplot = pres_back_df[0]
  backgroundtoplot = pres_back_df[1]
  # presencetoplot = annotated[annotated['class'] == 1]
  # backgroundtoplot = annotated[annotated['class'] == 0]

  pair_colors = ['#b61458', '#61be03']
  n_vars = len(labels)
  ncols = 7
  nrows = (n_vars + ncols - 1) // ncols  # ensures enough rows for all labels

  fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
  axs = axs.ravel()  # flatten for easy indexing

  for i, label in enumerate(labels):
      pvar = presencetoplot[label]
      bvar = backgroundtoplot[label]
      if bar:
        axs[i].hist(
            [pvar, bvar],
            density=True,
            alpha=0.7,
            label=['presence', 'background'],
            color=pair_colors,
        )
      else:
        resultkde = calculate_kde_metrics(pvar,bvar,label)
        import seaborn as sns
        sns.kdeplot(pvar, ax=axs[i], label='presence', color=pair_colors[0], fill=True, alpha=0.5)
        sns.kdeplot(bvar, ax=axs[i], label='background', color=pair_colors[1], fill=True, alpha=0.5)
        axs[i].text(0.95, 0.95, f"Overlap: {resultkde['overlap_percent']:.1f}%\nPres peak: {resultkde['pres_peak']:.2f}\nBg peak: {resultkde['bg_peak']:.2f}", transform=axs[i].transAxes, ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

  # remove unused subplots
  for j in range(i + 1, len(axs)):
      fig.delaxes(axs[j])  # remove the empty axes

  # add shared legend in empty area (e.g. right below all plots)
  handles, lbls = axs[0].get_legend_handles_labels()
  fig.legend(handles, lbls, loc='upper left', ncol=2)
  fig.suptitle(f'Presence and background distributions\n{titleplot}', fontsize=12, y=1.1)
  fig.tight_layout(pad=pad)
  fig.subplots_adjust(hspace=hspace, top=top,bottom=0.15)
  fig.show()

  if saveplot:
    fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_pres_back_smoothdist.tiff", dpi=300,bbox_inches='tight')
    print("Saved feature distribution plot\n")
  return fig

def plot_zonal_stats_bar(zs_df,region_names, columns_to_plot, zone_var = 'County', sort_bars=False):
    """
    Plots separate bar charts for each specified column of selected regions
    ('NAME_1' or 'County') from a zonal statistics DataFrame.

    Args:
        zs_df (pd.DataFrame): DataFrame containing zonal statistics, with 'NAME_1' or 'County' as a column.
        region_names (list): A list of 'NAME_1' or 'County' values (region names) to select.
        columns_to_plot (list): A list of column names from zs_df to plot for the selected regions.
        sort_bars (bool): If True, sort regions by y-value in each plot.
    """
    selected_regions_df = zs_df[zs_df[zone_var].isin(region_names)].copy()

    if selected_regions_df.empty:
        print(f"No regions found for the provided names: {region_names}")
        return

    selected_regions_df = selected_regions_df.set_index(zone_var)

    n_plots = len(columns_to_plot)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(10, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    for ax, column in zip(axes, columns_to_plot):
        data_to_plot = selected_regions_df[column]

        if sort_bars:
            data_to_plot = data_to_plot.sort_values(ascending=False)

        data_to_plot.plot(
            kind='bar',
            ax=ax,
            title=column.replace('_', ' ').capitalize()
        )
        ax.set_ylabel(column)
        ax.set_xlabel("Region")
        ax.set_xticklabels(data_to_plot.index, rotation=90)

    plt.tight_layout()
    plt.show()
    return fig

def zonal_outmap(zs, name, kmd_zone, output_model_dir,saveplot = False):
  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]
  titleplot = f'(Presence: {namepresence}\n Background: {namebg} under {nameraster})'

  # create output directory for all out files from this run
  # output_model_dir = f'{outputpath}/ModelOut_{namepresence}_{namebg}_{nameraster}'
  os.makedirs(output_model_dir, exist_ok=True) # Create the directory if it doesn't exist


  figpredsplit, ax = plt.subplots(figsize=(8,8))
  try:
    zs.plot(ax=ax, column='output_raster_split_mean', legend=False, cmap='GnBu', vmin=0, vmax=1) # Use column and cmap instead of color
  except:
    zs.plot(ax=ax, column='output_raster_split_f_mean', legend=False, cmap='GnBu', vmin=0, vmax=1)
    print('This is for future climate')
  # create colorbar with label
  sm = mpl.cm.ScalarMappable(cmap='GnBu', norm=mpl.colors.Normalize(vmin=0, vmax=1))
  sm._A = []  # dummy array for ScalarMappable
  cbar = figpredsplit.colorbar(sm, ax=ax)
  cbar.set_label("cloglog output")
  kmd_zone.plot(ax=ax,  edgecolor='black', facecolor='none')
  ax.set_title(f"{titleplot}\noutput_raster_split_mean")
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  if saveplot:
    figpredsplit.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_zonal_predsplit.png", dpi=300,bbox_inches='tight')
    print("Saved zonal predsplit image\n")
  return figpredsplit
  
  
def maxent_core_training(annotated, grid_size, output_model_dir, name,
                         beta_multiplier = 1.5, feature_types = 'lhqp', late_df_gdf = late_df_gdf):
  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]
  # split the x/y data for model training with no train/test split
  x = annotated.drop(columns=['class', 'geometry'])
  y = annotated['class']

  # train/test split with checkerboard pattern (ela.checkerboard_split)
  grid_size = grid_size
  train, test = ela.checkerboard_split(annotated, grid_size=grid_size)
  # re-merge them for plotting purposes
  train['split'] = 'train'
  test['split'] = 'test'
  checker = ela.stack_geodataframes(train, test)
  # plot test and train annotated merged samples in different colour
  ax = checker.plot(column='split', markersize=0.75, legend=True,figsize=(8,8))
  kmd_zone.plot(ax=ax, color='none', edgecolor='black')
  ax.set_title(f'Checkerboard split all samples (grid size = {grid_size})\n{namepresence}_{namebg}')
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  plt.savefig(f'{output_model_dir}/{namepresence}_{namebg}_checker.png', dpi=300, bbox_inches='tight')
  checker.to_csv(f'{output_model_dir}/{namepresence}_{namebg}_checker.csv')
  print("Out2.1 - Saved checkerboard image and csv\n")


  # set up model fitting with train/test split
  xtrain = train.drop(columns=['class', 'split'])
  ytrain = train['class']
  xtest = test.drop(columns=['class', 'split'])
  ytest = test['class']
  # initialize model and fit (train the model)
  modelsplit = ela.MaxentModel(transform='cloglog', beta_multiplier = beta_multiplier, feature_types = feature_types)
  modelsplit.fit(xtrain, ytrain)
  pprint(modelsplit.get_params())
  # save the fitted models to disk
  # ela.save_object(model, f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_model.ela")
  ela.save_object(modelsplit, f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit.ela")
  # print("Out3/8 - Saved both models\n")
  print("Out2.2 - finish training checkerboard-split Maxent\n")

  print(f"EVALUATION\n")
  # calculate AUC to evaluate training performance
  y_test_pred = modelsplit.predict(xtest)
  auc_checker = metrics.roc_auc_score(ytest, y_test_pred)
  print(f"checkerboard AUC score: {auc_checker:0.3f}\n")

  # Calculate confusion matrix using the binary predictions
  y_test_prob = modelsplit.predict_proba(xtest)
  y_test_pred_binary = (y_test_prob[:, 1] >= 0.5).astype(int)
  tn, fp, fn, tp = metrics.confusion_matrix(ytest, y_test_pred_binary).ravel()
  sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
  specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
  print(f"checkerboard Sensitivity: {sensitivity:0.3f}\n")
  print(f"checkerboard Specificity: {specificity:0.3f}\n")

  # extrinsic_presence + sampled background
  print("... Annotating extrinsic_presence and sampled background ...")
  extrinsic_presence = ela.annotate(late_df_gdf.copy(), rasters, labels=labels, drop_na=True, quiet=False)
  extrinsic_presence['class'] = 1
  background_2021 = test[test['class'] == 0].sample(n=len(extrinsic_presence), random_state=42)
  extrinsic_all = ela.stack_geodataframes(extrinsic_presence, background_2021)
  display(extrinsic_all.head(3))
  display(extrinsic_all.tail(3))

  # calculate AUC to evaluate training performance with extrinsic data
  x_ext = extrinsic_all.drop(columns=['class', 'geometry'])
  y_ext = extrinsic_all['class']

  # predict with the previously trained checkerboard model
  y_ext_pred = modelsplit.predict(x_ext)
  auc_ext = metrics.roc_auc_score(y_ext, y_ext_pred)
  print(f"extrinsic 2021 AUC: {auc_ext:0.3f}")
  print("Out3 - finish extrinsic validation\n")

  # ytest.to_csv(f'{output_model_dir}/ytest.csv')
  # yall = pd.concat([
  #   pd.DataFrame(y_test_prob, columns=["proba0","proba1"]),
  #   pd.DataFrame(y_test_pred_binary, columns=["pred"])], axis=1)
  # yall.to_csv(f'{output_model_dir}/ypred.csv')
  int2017_20_eval = pd.concat([xtest, ytest, pd.DataFrame(y_test_prob, columns=["proba0","proba1"])], axis=1)
  print("int2017_20_eval")
  print(int2017_20_eval.info())
  int2017_20_eval.to_csv(f'{output_model_dir}/int2017_20_eval.csv')

  ext2021_eval = pd.concat([x_ext, y_ext, pd.DataFrame(y_ext_pred, columns=["proba1"])], axis=1)
  print("ext2021_eval")
  print(ext2021_eval.info())
  ext2021_eval.to_csv(f'{output_model_dir}/ext2021_eval.csv')

  performance = pd.DataFrame([[sensitivity,specificity , auc_checker, auc_ext]],columns=["sensitivity", "specificity","auc_checker", "auc_ext"])
  print("performance")
  display(performance)
  performance.to_csv(f'{output_model_dir}/auc_checker_auc_ext.csv')


  # save model parameters and auc score for both models to disk
  with open(f"{output_model_dir}/{namepresence}_{namebg}__{nameraster}_ModelParms.txt", "w") as f:
    print(modelsplit.get_params(), file = f)
    print(f"checkerboard AUC score: {auc_checker:0.3f}", file = f)
    print(f"checkerboard Sensitivity: {sensitivity:0.3f}", file = f)
    print(f"checkerboard Specificity: {specificity:0.3f}", file = f)
    print(f"ExtrinsicAUC: {auc_ext:0.3f}", file = f)
    print(vars(modelsplit), file = f)
  print("Out4 - Saved model' parms\n")

  # return modelsplit, xtrain, xtest, ytrain, ytest, y_test_prob,y_test_pred_binary, auc_checker, x_ext, y_ext, y_ext_pred, auc_ext
  return {
    "model_split": modelsplit,
    "x_train": xtrain,
    "y_train": ytrain,
    "int2017_20_eval": int2017_20_eval,
    "auc_checker": auc_checker,
    # external dataset fields (put placeholders if not computed yet)
    "x_ext": x_ext if 'x_ext' in locals() else None,
    "y_ext": y_ext if 'y_ext' in locals() else None,
    "y_ext_pred": y_ext_pred if 'y_ext_pred' in locals() else None,
    "auc_ext": auc_ext if 'auc_ext' in locals() else None,
    "ext2021_eval": ext2021_eval if 'ext2021_eval' in locals() else None,
}


def maxent_core_projecting(modelsplit, rasters, titleplot, output_model_dir, name):
  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]


  # checkerboard model
  output_raster_split = f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_map.tif"
  ela.apply_model_to_rasters(modelsplit, rasters, output_raster_split, quiet=False)
  with rasterio.open(output_raster_split, 'r') as src:
      pred_split = src.read(1, masked=True)
  # 5.2 plot the suitability predictions
  fig_modelsplitmap, ax = plt.subplots(1, 1, figsize=(8, 8))
  plot = ax.imshow(pred_split, vmin=0, vmax=1, cmap='GnBu')
  ax.set_title(f'FAW suitability\n{titleplot}\nmodelsplit')
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  cbar = plt.colorbar(plot, ax=ax, label="cloglog output", pad=0.04)
  plt.tight_layout()
  plt.show()
  # fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_map.png", dpi=300)
  # print("Out5.2/8 - Saved habitat suitability map for checkerboard model\n")

  return fig_modelsplitmap , output_raster_split




def maxent_single(presence,background,rasters, labels, name,
                  bound,kmd_zone, zone_var = 'County',
                  outputpath = 'maxent_output',grid_size = 0.5,
                  beta_multiplier = 1.5, bar = False, future_rasters = None,
                  late_df_gdf = late_df_gdf):

  """
  run maxent for a combination of presence,background,raster data

  Args:
      presence: FAW observation geopandas - col = species, geometry
      background: Background generated sample from different null hypothesis - col = gemetry
      rasters: climate rasters in tifs
      labels: labels of climate rasters
      namepresence: [str] name of presence data
      namebg: [str] name of background data
      nameraster: [str] name of raster data
      grid_size: for Checkerboard splitting - defauklt 0.5 for this data set
      bound: Kenya landscape boundary

  Returns:
      zs, model, modelsplit, x, xtrain, xtest, pred, pred_split
  """
  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]

  # create output directory for all out files from this run
  output_model_dir = f'{outputpath}/ModelOut_{namepresence}_{namebg}_{nameraster}'
  os.makedirs(output_model_dir, exist_ok=True) # Create the directory if it doesn't exist
  print(f"Producing model for ModelOut_{namepresence}_{namebg}_{nameraster}")

  # 1.1 plot presence and background point
  fig, ax = plt.subplots(figsize=(8,8))
  kmd_zone.plot(ax=ax, color='none', edgecolor='black')
  background.plot(ax=ax, marker='o', color='#61be03', markersize=1.2, label=f'{namebg}')
  presence.plot(ax=ax, legend=True, markersize=1.2, color='#b61458', label=f'{namepresence}')
  ax.set_title(f"Presence: {namepresence}\nBackground: {namebg}")
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.legend()
  plt.savefig(f'{output_model_dir}/{namepresence}_{namebg}_merged_samples.tiff', dpi=300, bbox_inches='tight')
  print("Out1.1 - Saved plot of samples\n")

  # 1.2 merge datasets and annotate with the covariates (climate factors) at each point location
  merged = ela.stack_geodataframes(presence, background, add_class_label=True)
  annotated = ela.annotate(merged, rasters, labels=labels, drop_na=True, quiet=False)
  annotated.to_csv(f'{output_model_dir}/{namepresence}_{namebg}_annotated.csv')
  print("Out1.2 - Saved annotated csv\n")

  # 2. core training, 3. exporting , 4. evaluating model
  # model, modelsplit, x, xtrain, xtest, y, ytrain, ytest,y_test_prob,y_test_pred_binary,auc, x_ext, y_ext, y_ext_pred, auc_ext = maxent_core_training(annotated, grid_size, output_model_dir, name, beta_multiplier = beta_multiplier, feature_types = 'lhpqt')
  training_result = maxent_core_training(annotated, grid_size, output_model_dir, name,
                                         beta_multiplier = beta_multiplier, feature_types = 'lpqh',
                                         late_df_gdf = late_df_gdf)
  modelsplit = training_result["model_split"]
  xtrain = training_result["x_train"]
  ytrain = training_result["y_train"]
  auc_checker = training_result["auc_checker"]
  auc_ext = training_result["auc_ext"]




  # 5. write the model predictions to disk, using historical rasters
  print(f"Using {nameraster} for visualising predictions on map, contain the bands:\n {labels}")
  titleplot = f'(Presence: {namepresence}\n Background: {namebg} under {nameraster})'
  fig_modelsplitmap , output_raster_split = maxent_core_projecting(modelsplit, rasters, titleplot, output_model_dir, name)
  fig_modelsplitmap.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_map.tiff", dpi=300)
  print("Out5 - Saved habitat suitability map for checkerboard model\n")

  # 6.1 plot the permutation_importance_plot
  fig, ax = modelsplit.permutation_importance_plot(xtrain, ytrain)
  ax.set_title(f'Permutation importance\n{titleplot}\nmodelsplit', fontsize=12, y=1.0)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_permutation.tiff", dpi=300,bbox_inches='tight')
  print("Out6.1 - Saved feature permutation importance plot\n")

  # 6.2 plot the partial_dependence_plot
  fig, ax = modelsplit.partial_dependence_plot(xtrain, labels=labels, constrained_layout=True, figsize=(15, 10))
  for a, label in zip(ax.flat, labels):
      a.set_title(label, fontsize=10)
  fig.suptitle(f'Response curve\n{titleplot}\nmodelsplit', fontsize=12, y=1.15)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_response.tiff", dpi=300,bbox_inches='tight')
  print("Out6.2 - Saved feature response curve plot\n")

  # 7. plot presence vs background presence
  presencetoplot = annotated[annotated['class'] == 1]
  backgroundtoplot = annotated[annotated['class'] == 0]
  # bar = False
  fig = pres_back_dis(pres_back_df = [presencetoplot,backgroundtoplot], name = name, labels = labels,
                  outputpath = outputpath ,saveplot = False, bar = bar, pad = 0.2, hspace=0.4, top=0.90)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_pres_back_smoothdist.png", dpi=400,bbox_inches='tight')
  print("Out7 - Saved presence vs background distribution image\n")

  # 8.1 calc zonal_stats
  zs = ela.zonal_stats(
    kmd_zone,
    rasters + [output_raster_split],
    labels + ['output_raster_split'],
    mean = True,
    percentiles = [25, 75],
    quiet=False)
  zs.to_csv(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_zonal_stats.csv")
  print("Out8.1 - Saved zonal stat csv\n")

  # 8.2 plot zonal_stats 1d
  selected_regions = zs[zone_var]
  columns_to_plot_example = [col for col in zs.columns if col.endswith('_mean')]
  fig = plot_zonal_stats_bar(zs, selected_regions, columns_to_plot_example, sort_bars=False)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_zonal_stats.png", dpi=400,bbox_inches='tight')
  print("Out8.2 - Saved zonal stat bar graph\n")

  # 8.3 plot zonal_stats 2d
  figpredsplit = zonal_outmap(zs ,name, kmd_zone, output_model_dir ,saveplot = False)
  figpredsplit.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_zonal_predsplit.tiff", dpi=300,bbox_inches='tight')
  print("Out8.3 - Saved zonal pred and predsplit images\n")

  print(f"✅ See all 8 results in {output_model_dir}")

  if future_rasters is not None:
    # use future rasters for projections (repeat maxent_core_projecting)
    # also calculate the zonal stats and plot
    print(f"Using future_rastersfor visualising predictions on map, must contain the bands:\n {labels}")
    # titleplot_f = f'(Presence: {namepresence}\n Background: {namebg} under {future_rasters})'
    # fig_modelmap_f, fig_modelsplitmap , output_raster, output_raster_split
    map, zonal = maxent_projecting_future(model, modelsplit, future_rasters, output_model_dir, name)
    fig_modelmap_f, fig_modelsplitmap_f, output_raster_f, output_raster_split_f = map
    zs_f, fig_zonal_bar_f, figpred_f, figpredsplit_f = zonal
    fig_modelmap_f.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_future_model_map.tiff",  dpi=300)
    fig_modelsplitmap_f.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_future_modelsplit_map.tiff", dpi=300)
    print("Out_future - Saved future habitat suitability map for no-split model\n")
    print("Out_future - Saved future habitat suitability map for checkerboard model\n")
    zs_f.to_csv(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_future_zonal_stats.csv")
    print("Out_future - Saved zonal stat csv\n")
    fig_zonal_bar_f.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_future_zonal_stats.tiff", dpi=300,bbox_inches='tight')
    print("Out_future - Saved zonal stat bar graph\n")
    figpred_f.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_future_zonal_pred.png", dpi=300,bbox_inches='tight')
    figpredsplit_f.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_future_zonal_predsplit.tiff", dpi=300,bbox_inches='tight')
    print("Out_future - Saved zonal pred and predsplit images\n")
  else:
    zs_f = None
    output_raster_f = None
    output_raster_split_f = None

  print(f"✅ Save zonal stat, modelsplit, output_raster_split, x, xtrain, xtest to this assigned variable for further processing")

  return [[zs,modelsplit,  output_raster_split], [presencetoplot,backgroundtoplot]], [zs_f,[output_raster_f,output_raster_split_f]]


def maxent_core_training_timeaware(annotated, grid_size, output_model_dir, name,
                         beta_multiplier = 1.5, feature_types = 'lhqp', anno_late_df_gdf = anno_late_df_gdf):
  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]
  # split the x/y data for model training with no train/test split
  x = annotated.drop(columns=['class', 'geometry'])
  y = annotated['class']

  # train/test split with checkerboard pattern (ela.checkerboard_split)
  grid_size = grid_size
  train, test = ela.checkerboard_split(annotated, grid_size=grid_size)
  # re-merge them for plotting purposes
  train['split'] = 'train'
  test['split'] = 'test'
  checker = ela.stack_geodataframes(train, test)
  # plot test and train annotated merged samples in different colour
  ax = checker.plot(column='split', markersize=0.75, legend=True,figsize=(8,8))
  kmd_zone.plot(ax=ax, color='none', edgecolor='black')
  ax.set_title(f'Checkerboard split all samples (grid size = {grid_size})\n{namepresence}_{namebg}')
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  plt.savefig(f'{output_model_dir}/{namepresence}_{namebg}_checker.png', dpi=300, bbox_inches='tight')
  checker.to_csv(f'{output_model_dir}/{namepresence}_{namebg}_checker.csv')
  print("Out2.1 - Saved checkerboard image and csv\n")


  # set up model fitting with train/test split
  xtrain = train.drop(columns=['class', 'split'])
  ytrain = train['class']
  xtest = test.drop(columns=['class', 'split'])
  ytest = test['class']
  # initialize model and fit (train the model)
  modelsplit = ela.MaxentModel(transform='cloglog', beta_multiplier = beta_multiplier, feature_types = feature_types)
  modelsplit.fit(xtrain, ytrain)
  pprint(modelsplit.get_params())
  # save the fitted models to disk
  # ela.save_object(model, f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_model.ela")
  ela.save_object(modelsplit, f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit.ela")
  # print("Out3/8 - Saved both models\n")
  print("Out2.2 - finish training checkerboard-split Maxent\n")

  print(f"EVALUATION\n")
  # calculate AUC to evaluate training performance
  y_test_pred = modelsplit.predict(xtest)
  auc_checker = metrics.roc_auc_score(ytest, y_test_pred)
  print(f"checkerboard AUC score: {auc_checker:0.3f}\n")

  # Calculate confusion matrix using the binary predictions
  y_test_prob = modelsplit.predict_proba(xtest)
  y_test_pred_binary = (y_test_prob[:, 1] >= 0.5).astype(int)
  tn, fp, fn, tp = metrics.confusion_matrix(ytest, y_test_pred_binary).ravel()
  sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
  specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
  print(f"checkerboard Sensitivity: {sensitivity:0.3f}\n")
  print(f"checkerboard Specificity: {specificity:0.3f}\n")

  # extrinsic_presence + sampled background
  print("... already have timeware annotated extrinsic_presence, just add sampled background ...")
  # extrinsic_presence = ela.annotate(late_df_gdf.copy(), rasters, labels=labels, drop_na=True, quiet=False)
  extrinsic_presence = anno_late_df_gdf
  extrinsic_presence['class'] = 1
  background_2021 = test[test['class'] == 0].sample(n=len(extrinsic_presence), random_state=42)
  extrinsic_all = ela.stack_geodataframes(extrinsic_presence, background_2021)
  display(extrinsic_all.head(3))
  display(extrinsic_all.tail(3))

  # calculate AUC to evaluate training performance with extrinsic data
  x_ext = extrinsic_all.drop(columns=['class', 'geometry'])
  y_ext = extrinsic_all['class']

  # predict with the previously trained checkerboard model
  y_ext_pred = modelsplit.predict(x_ext)
  auc_ext = metrics.roc_auc_score(y_ext, y_ext_pred)
  print(f"extrinsic 2021 AUC: {auc_ext:0.3f}")
  print("Out3 - finish extrinsic validation\n")

  # ytest.to_csv(f'{output_model_dir}/ytest.csv')
  # yall = pd.concat([
  #   pd.DataFrame(y_test_prob, columns=["proba0","proba1"]),
  #   pd.DataFrame(y_test_pred_binary, columns=["pred"])], axis=1)
  # yall.to_csv(f'{output_model_dir}/ypred.csv')
  int2017_20_eval = pd.concat([xtest, ytest, pd.DataFrame(y_test_prob, columns=["proba0","proba1"])], axis=1)
  print("int2017_20_eval")
  print(int2017_20_eval.info())
  int2017_20_eval.to_csv(f'{output_model_dir}/int2017_20_eval.csv')

  ext2021_eval = pd.concat([x_ext, y_ext, pd.DataFrame(y_ext_pred, columns=["proba1"])], axis=1)
  print("ext2021_eval")
  print(ext2021_eval.info())
  ext2021_eval.to_csv(f'{output_model_dir}/ext2021_eval.csv')

  performance = pd.DataFrame([[sensitivity,specificity , auc_checker, auc_ext]],columns=["sensitivity", "specificity","auc_checker", "auc_ext"])
  print("performance")
  display(performance)
  performance.to_csv(f'{output_model_dir}/auc_checker_auc_ext.csv')


  # save model parameters and auc score for both models to disk
  with open(f"{output_model_dir}/{namepresence}_{namebg}__{nameraster}_ModelParms.txt", "w") as f:
    print(modelsplit.get_params(), file = f)
    print(f"checkerboard AUC score: {auc_checker:0.3f}", file = f)
    print(f"checkerboard Sensitivity: {sensitivity:0.3f}", file = f)
    print(f"checkerboard Specificity: {specificity:0.3f}", file = f)
    print(f"ExtrinsicAUC: {auc_ext:0.3f}", file = f)
    print(vars(modelsplit), file = f)
  print("Out4 - Saved model' parms\n")

  # return modelsplit, xtrain, xtest, ytrain, ytest, y_test_prob,y_test_pred_binary, auc_checker, x_ext, y_ext, y_ext_pred, auc_ext
  return {
    "model_split": modelsplit,
    "x_train": xtrain,
    "y_train": ytrain,
    "int2017_20_eval": int2017_20_eval,
    "auc_checker": auc_checker,
    # external dataset fields (put placeholders if not computed yet)
    "x_ext": x_ext if 'x_ext' in locals() else None,
    "y_ext": y_ext if 'y_ext' in locals() else None,
    "y_ext_pred": y_ext_pred if 'y_ext_pred' in locals() else None,
    "auc_ext": auc_ext if 'auc_ext' in locals() else None,
    "ext2021_eval": ext2021_eval if 'ext2021_eval' in locals() else None,
}


def maxent_single_timeaware(annotated,rasters, labels,name, bound,kmd_zone,
                            zone_var = 'County', outputpath = 'maxent_output',
                            grid_size = 0.5, beta_multiplier = 1.5, bar = False,
                            future_rasters = None, feature_types = 'lpqh', anno_late_df_gdf = anno_late_df_gdf):
  presence = annotated[annotated['class'] ==1]
  background = annotated[annotated['class'] ==0]

  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]
  labels_anno = annotated.columns[2:]
  print('labels from given annotated df', labels_anno)
  print('keys of climate raster used', labels)
  if sorted(labels_anno) != sorted(labels):
    print('! Conflicts between labels from given annotated df and keys of climate raster used')
    raise ValueError("Label mismatch: stop execution.")
  # create output directory for all out files from this run
  output_model_dir = f'{outputpath}/ModelOut_{namepresence}_{namebg}_{nameraster}'
  os.makedirs(output_model_dir, exist_ok=True) # Create the directory if it doesn't exist
  print(f"Producing model for ModelOut_{namepresence}_{namebg}_{nameraster}")

  # 1 plot presence and background point
  fig, ax = plt.subplots(figsize=(6,6))
  kmd_zone.plot(ax=ax, color='none', edgecolor='black')
  background.plot(ax=ax, marker='o', color='#61be03', markersize=1.2, label=f'{namebg}')
  presence.plot(ax=ax, legend=True, markersize=1.2, color='#b61458', label=f'{namepresence}')
  ax.set_title(f"Presence: {namepresence}\nBackground: {namebg}")
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.legend()
  plt.savefig(f'{output_model_dir}/{namepresence}_{namebg}_merged_samples.tiff', dpi=300, bbox_inches='tight')
  print("Out1 - Saved plot of samples\n")

  # 2. core training, 3. exporting , 4. evaluating model
  # model, modelsplit, x, xtrain, xtest, y, ytrain, ytest = maxent_core_training(annotated, grid_size, output_model_dir, name, beta_multiplier = beta_multiplier, feature_types = 'lhpqt')
  training_result = maxent_core_training_timeaware(annotated, grid_size, output_model_dir, name,
                                         beta_multiplier = beta_multiplier, feature_types = feature_types,
                                         anno_late_df_gdf = anno_late_df_gdf)
  modelsplit = training_result["model_split"]
  xtrain = training_result["x_train"]
  ytrain = training_result["y_train"]
  auc_checker = training_result["auc_checker"]
  auc_ext = training_result["auc_ext"]

  # 5. write the model predictions to disk, using historical rasters
  print(f"Using {nameraster} for visualising predictions on map, contain the bands:\n {labels}")
  titleplot = f'(Presence: {namepresence}\n Background: {namebg} under {nameraster})'
  fig_modelsplitmap,output_raster_split = maxent_core_projecting(modelsplit, rasters, titleplot, output_model_dir, name)
  fig_modelsplitmap.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_map.tiff", dpi=300)
  print("Out5 - Saved habitat suitability map for checkerboard model\n")

  # 6.1 plot the permutation_importance_plot
  fig, ax = modelsplit.permutation_importance_plot(xtrain, ytrain)
  ax.set_title(f'Permutation importance\n{titleplot}\nmodelsplit', fontsize=12, y=1.0)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_permutation.tiff", dpi=300,bbox_inches='tight')
  print("Out6.1 - Saved feature permutation importance plot\n")

  # 6.2 plot the partial_dependence_plot
  fig, ax = modelsplit.partial_dependence_plot(xtrain, labels=labels, constrained_layout=True, figsize=(15, 10))
  for a, label in zip(ax.flat, labels):
      a.set_title(label, fontsize=10)
  fig.suptitle(f'Response curve\n{titleplot}\nmodelsplit', fontsize=12, y=1.15)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_modelsplit_response.tiff", dpi=300,bbox_inches='tight')
  print("Out6.2 - Saved feature response curve plot\n")

  # 7. plot presence vs background presence
  presencetoplot = annotated[annotated['class'] == 1]
  backgroundtoplot = annotated[annotated['class'] == 0]
  # bar = False
  fig = pres_back_dis(pres_back_df = [presencetoplot,backgroundtoplot], name = name, labels = labels,
                  outputpath = outputpath ,saveplot = False, bar = bar, pad = 0.2, hspace=0.4, top=0.90)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_pres_back_smoothdist.png", dpi=400,bbox_inches='tight')
  print("Out7/8 - Saved presence vs background image\n")

  # 8.1 calc zonal_stats
  zs = ela.zonal_stats(
    kmd_zone,
    rasters + [output_raster_split],
    list(labels)  + ['output_raster_split'],
    mean = True,
    percentiles = [25, 75],
    quiet=False
  )
  zs.to_csv(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_zonal_stats.csv")
  print("Out8.1 - Saved zonal stat csv\n")

  # 8.2 plot zonal_stats 1d
  selected_regions = zs[zone_var]
  columns_to_plot_example = [col for col in zs.columns if col.endswith('_mean')]
  fig = plot_zonal_stats_bar(zs, selected_regions, columns_to_plot_example)
  fig.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_zonal_stats.png", dpi=400,bbox_inches='tight')
  print("Out8.2/8 - Saved zonal stat bar graph\n")

  # 8.3 plot zonal_stats 2d
  figpredsplit = zonal_outmap(zs, name, kmd_zone, output_model_dir ,saveplot = False)
  figpredsplit.savefig(f"{output_model_dir}/{namepresence}_{namebg}_{nameraster}_zonal_predsplit.tiff", dpi=300,bbox_inches='tight')
  print("Out8.3/8 - Saved zonal pred and predsplit images\n")


  print(f"See all 8 results in {output_model_dir}")
  print(f"Save zonal stat, model, modelsplit, x, xtrain, xtest to this assigned variable for further processing")

  return [[zs,modelsplit, output_raster_split], [presencetoplot,backgroundtoplot]]
  
  
def maxent_projecting_future(modelsplit, future_rasters, labels, output_model_dir, name,kmd_zone, zone_var = 'County'):
  # use future rasters for projections (repeat maxent_core_projecting)
  # also calculate the zonal stats and plot
  namepresence = name[0]
  namebg = name[1]
  nameraster = name[2]

  titleplot = f'(Presence: {namepresence}\n Background: {namebg} under future {nameraster})'
  # maxent_core_projecting
  fig_modelsplitmap_f, output_raster_split_f = maxent_core_projecting(modelsplit, future_rasters, titleplot, output_model_dir, name)
  # cal zonal_stats
  zs_f = ela.zonal_stats(
    kmd_zone,
    future_rasters + [output_raster_split_f],
    labels + ['output_raster_split_f'],
    mean = True,
    percentiles = [25, 75],
    quiet=False)

  # plot zonal_stats 1d
  selected_regions = zs_f[zone_var]
  columns_to_plot_example = [col for col in zs_f.columns if col.endswith('_mean')]
  fig_zonal_bar_f = plot_zonal_stats_bar(zs_f, selected_regions, columns_to_plot_example)
  fig_zonal_bar_f_sorted = plot_zonal_stats_bar(zs_f, selected_regions, columns_to_plot_example,sort_bars=True)

  # plot zonal_stats 2d
  figpredsplit_f = zonal_outmap(zs_f, name, kmd_zone, output_model_dir ,saveplot = False)

  return [fig_modelsplitmap_f, output_raster_split_f] , [zs_f, fig_zonal_bar_f, fig_zonal_bar_f_sorted, figpredsplit_f]
