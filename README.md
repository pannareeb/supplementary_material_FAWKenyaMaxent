# supplementary_material_FAWKenyaMaxent
Supplementary material for the research "Fall Armyworm infestation and future risk from environmental change in maize-producing regions of Kenya 

## 1_data: 
- climate_data: folders of post-processed five historical climatic datasets and two future climatic datasets
- faw_data: raw data from FAO and final thinned FAW presence observation in Kenya (raw data from CABI is confidential)
  - Interactive_plot 01, showing thinned presence data: https://pannareeb.github.io/FAW_rhul_2025/PlotPublication/spatial_faw_presence_vs_thinned_map_interactive01.html 
- spatial_data: LAI index (for generating background 2), Kenya boundary removing deserts and water bodies (for maxent run), and boundary of Kenya at county (AMD1) and subcounty (AMD2) level (for zonal aggregation)
- augment_data: 3-year planting areas for maize (mergedstat_geo_df) calculated from the other two raw data files in the folder
- bg_generated_data: generated background samples of three types

  
## 2_scripts: 
contains functions used by notebooks

## 3_notebooks: 
contains methodology workflow, from data preprocessing, maxent run, and output analysis. Note that 1_ and 2_ notebooks are for fall armyworm data preprocessing and visualisation which contain confidential data, so we leave them out.
- 3_climate_data_ importing_visualisation.ipynb
- 4_prep_for_maxent.ipynb
- 5_maxentrun_analysis.ipynb

## 4_maxent_output: 
contains outputs of maxent run and analysis from notebook5_maxentrun_analysis.ipynb
- historical_run: 15 runs with five historical climatic datasets and three background samples   
- bestmodels_combinedprediction: M22, M23, M25 prediction map
- historical_varied_beta_run: M23 with varying regularisation multiplier (beta) 
- future_run: M23 run with future climate data
