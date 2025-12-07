# supplementary_material_FAWKenyaMaxent
Supplementary material for the research "Fall Armyworm infestation and future likelihood of spread due to environmental change in maize-cropping regions of Kenya" 

## 1_data: 
- climate_data: folders of raw and post-processed climatic datasets
	- raw_historical: only shown raw cliamtic data called using Google Earth Engine (for era5land, era5yearmonthly). Raw data for the other three can be accessed from the website [worldclim.org](https://www.worldclim.org/data/worldclim21.html) 
	- raw_future: future cliamtes under 2 pathways in two subfolders, each contain five 19-band rasters from five global climate models (GCMs).
	- processed_historical: five processed historical climatic datasets in five subfolders named with the same climate data set names used in the paper
	- processed_future: future cliamtes under 2 pathways 5 GCMs in ten subfolders, each containing 19 single-band rasters for 19 bioclimatic factors.
- faw_data: 
	- raw: Kenya FAW observations from FAO (trap ans scout, which can be assessed online for global span from [scout_FAO](https://data.apps.fao.org/catalog//dataset/fall-armyworm-scout-famews-global-latlon) or [trap_FAO](https://data.apps.fao.org/catalog//dataset/fall-armyworm-traps-famews-global-latlon)). Raw data from CABI is confidential.
	- processed: early-year (2017-2020) FAW presence observations thinned with 3-, 5-, 9-km minimum distance (in .csv and .geojson) and unthinned late-year data (2021)
- spatial_data: 
	- leaf_area_index_low_vegetation_raster_tif_clipped: LAI index raster from GEE (for generating background 2) 
	- pnv_vecea_excludeWDDa_dissolved_extend: Kenya boundary removing deserts and water bodies (for maxent run)
	- augment_cleaned_counties_gdf: overlaid between valid boundary MaxEnt boundary and county boundaries (for zonal aggregation))
	- gadm41_KEN_1 and gadm41_KEN_2: raw boundary of Kenya at county (AMD1) and subcounty (AMD2) level
- augment_data: 
	- 3-year planting areas for maize (mergedstat_geo_df) calculated from the other three raw data files in the folder
- bg_generated_data: generated background samples of three types: bg1uniform, bg2lai, bg3meanA3yr
  
## 2_scripts: 
contains functions used by notebooks
- proprocess_fawdata
- visualisation_fawdata
- preprocess_climatedata
- visualisation_climatedata
- augment_elapid
- visualisation_elapid

## 3_notebooks: 
contains methodology workflow, from data preprocessing, maxent run, and output analysis. Note that 1_ and 2_ notebooks are for fall armyworm data preprocessing and visualisation which contain confidential data, so we leave them out.
Each notebook was run in Google Colab environment.
- 3_climate_data_ importing_visualisation.ipynb
- 4_prep_for_maxent.ipynb
- 5_maxent_run.ipynb
- 6_result_analysis.ipynb

## 4_maxent_output: 
contains outputs of maxent run and analysis from notebook5_maxentrun_analysis.ipynb
- grid0.5: 30 output folders for 30 MaxEnt training runs with five historical climatic datasets and three background samples for thinned5km and thinned9km presence data (the rest 105 models are not given to save storage and uploading time)
- bestmodels: output folders for M22, M23, M25 runs, along with their performance metrics and the combined prediction map (in raster .tif and vector .csv for calculating percentages of pixel value bins)
- stats: list_score_grid030405.csv contais performance metrics for all 135 models, and TableA1 shows correlation quantiles of among groups of models trained with the same climate datasets
- future: M23 run with with 10 climate scenerios (two pathways x 5 GCMs)

## 5_additional_visualisation
contain interactive downloadable .html to explore
early thinned data alone or with late data, and with three background types.


# Repository author
Pannaree Boonyuen
Email address: pannaree.boonyuen.2024@live.rhul.ac.uk 
ORCID: https://orcid.org/0009-0007-1206-7375 
Postal address: Department of Biological Sciences, Bourne Building, 
Royal Holloway University of London, Egham Hill, Egham TW20 0EX, United Kingdom 
