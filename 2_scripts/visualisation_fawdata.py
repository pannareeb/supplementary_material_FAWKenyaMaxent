
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go

def timing_define(df):
  df = df.copy()
  df['Date'] = pd.to_datetime(df['Date'])
  df['Year'] = df['Date'].dt.year
  df['Month_of_year'] = df['Date'].dt.month
  df['YearMonth'] = df['Date'].dt.to_period('M')
  df['Week_of_year'] = df['Date'].dt.isocalendar().week
  return df

def timing_dist(listdf_sep, timelevel):
  '''
    listdf = [ky_scout_final_vis,ky_trap_final_vis,ky_cabi_final_vis] or [ky_scout_presence,ky_trap_presence,ky_cabi_presence]
    timelevel: 'Year', 'YearMonth','Month_of_year', 'Week_of_year'
    nametypedf: ['all', 'presence']
  '''

  ky_scout = listdf_sep[0]
  ky_trap = listdf_sep[1]
  ky_cabi = listdf_sep[2]

  ky_scout = timing_define(ky_scout)
  ky_trap = timing_define(ky_trap)
  ky_cabi = timing_define(ky_cabi)

  # Count observations per timelevel for each df
  scout_counts = ky_scout[timelevel].value_counts().sort_index()
  trap_counts = ky_trap[timelevel].value_counts().sort_index()
  cabi_counts = ky_cabi[timelevel].value_counts().sort_index()
  # Add observations per timelevel for fao (scout+trap) and combined (fao+cabi)
  fao_counts = scout_counts.add(trap_counts, fill_value=0).sort_index()
  combined_counts = fao_counts.add(cabi_counts, fill_value=0).sort_index()
  combined_counts_df = pd.DataFrame({'combined_counts': combined_counts})
  cabi_counts_df = pd.DataFrame({'cabi_counts': cabi_counts})
  trap_counts_df = pd.DataFrame({'trap_counts': trap_counts})
  scout_counts_df = pd.DataFrame({'scout_counts': scout_counts})
  merged_df = pd.concat([combined_counts_df, cabi_counts_df, trap_counts_df, scout_counts_df], axis=1)
  merged_df.index = merged_df.index.astype(str)
  return merged_df

def plot_timingdist(listdf_sep, timelevel, nametypedf, rangey = range(0, 1500, 200)):
  '''
    listdf = [ky_scout_final_vis,ky_trap_final_vis,ky_cabi_final_vis] or [ky_scout_presence,ky_trap_presence,ky_cabi_presence]
    timelevel: 'Year', 'YearMonth','Month_of_year', 'Week_of_year'
    nametypedf: ['all', 'presence']
  '''

  ky_scout = listdf_sep[0]
  ky_trap = listdf_sep[1]
  ky_cabi = listdf_sep[2]

  ky_scout = timing_define(ky_scout)
  ky_trap = timing_define(ky_trap)
  ky_cabi = timing_define(ky_cabi)

  # Count observations per timelevel for each df
  scout_counts = ky_scout[timelevel].value_counts().sort_index()
  trap_counts = ky_trap[timelevel].value_counts().sort_index()
  cabi_counts = ky_cabi[timelevel].value_counts().sort_index()
  # Add observations per timelevel for fao (scout+trap) and combined (fao+cabi)
  fao_counts = scout_counts.add(trap_counts, fill_value=0).sort_index()
  combined_counts = fao_counts.add(cabi_counts, fill_value=0).sort_index()
  
  # Convert the PeriodIndex to strings for plotting
  fao_counts.index = fao_counts.index.astype(str)
  cabi_counts.index = cabi_counts.index.astype(str)
  combined_counts.index = combined_counts.index.astype(str)

  # Plotting
  fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
  # Combined plot
  axes[2].bar(combined_counts.index, combined_counts.values, color='purple')
  axes[2].set_title(f'Combined FAO and CABI {nametypedf} Observations per {timelevel}')
  axes[2].set_ylabel('Number of Observations')
  axes[2].tick_params(axis='x', rotation=90)
  axes[2].set_yticks(rangey)
  axes[2].set_xlabel(f'{timelevel}')
  # FAO plot
  axes[0].bar(fao_counts.index, fao_counts.values, color='blue')
  axes[0].set_title(f'FAO {nametypedf} Observations per {timelevel}')
  axes[0].set_ylabel('Number of Observations')
  axes[0].tick_params(axis='x', rotation=90)
  axes[0].set_yticks(rangey)
  # CABI plot
  axes[1].bar(cabi_counts.index, cabi_counts.values, color='red')
  axes[1].set_title(f'CABI {nametypedf} Observations per {timelevel}')
  axes[1].set_ylabel('Number of Observations')
  axes[1].tick_params(axis='x', rotation=90)
  axes[1].set_yticks(rangey)


  plt.tight_layout()
  plt.show()
  return fig

def timewindow_aggregate(df,var, timelevel, amdlevel = 1, method = 'mean'):
  df = df.copy()
  df = timing_define(df)
  if timelevel in df.columns:
    print(f'{timelevel} column exist, start aggregate with {method}')
  else: 
    print(f"{timelevel} column does not exist, change timelevel to 'Year', 'YearMonth','Month_of_year', 'Week_of_year', 'Date'")
  
  if amdlevel == 0:
    print(f'existing amdlevel used, {method} aggregate at AMD level {amdlevel}')
    if method == 'mean':
      agg_data = df.groupby(timelevel)[[var]].mean().reset_index()
    elif method == 'sum':
      agg_data = df.groupby(timelevel)[[var]].sum().reset_index()
    else:
      print("method should be either 'mean' or 'sum'")
  elif amdlevel == 1:
    print(f'existing amdlevel used, {method} aggregate at AMD level {amdlevel}')
    if method == 'mean':
      agg_data = df.groupby([timelevel, 'search_state'])[[var]].mean().reset_index()
    elif method == 'sum':
      agg_data = df.groupby([timelevel, 'search_state'])[[var]].sum().reset_index()
    else:
      print("method should be either 'mean' or 'sum'")
  else:
    print("amdlevel should be either 0 or 1")
  
  return agg_data


def plot_timewindow_aggregate(df,var, timelevel, namedf, amdlevel = 1, method = 'mean'):
  fig = go.Figure()

  if amdlevel == 0:
    agg_data = timewindow_aggregate(df,var,timelevel,amdlevel = 0,method = method) 
    agg_data[timelevel] = agg_data[timelevel].astype(str)

    fig.add_trace(go.Scatter(
        x=agg_data[timelevel],
        y=agg_data[var])
    )

  elif amdlevel == 1:
    agg_data = timewindow_aggregate(df,var,timelevel,amdlevel = 1,method = method)
    agg_data[timelevel] = agg_data[timelevel].astype(str)
    agg_data['search_state'] = agg_data['search_state'].astype(str)
    locations = sorted(agg_data['search_state'].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(locations)))
    
    for i, location in enumerate(locations):
      location_data = agg_data[agg_data['search_state'] == location]
      fig.add_trace(go.Scatter(
          x=location_data[timelevel],
          y=location_data[var],
          mode='lines+markers',
          name=location,
          line=dict(dash='solid',color='rgba({},{},{},1)'.format(
              int(colors[i][0] * 255),
              int(colors[i][1] * 255),
              int(colors[i][2] * 255)
          ))
      ))
  else:
    print("amdlevel should be either 0 or 1")

  fig.update_layout(
      title=f'{timelevel} {method} aggregated {var} over time, at AMD level {amdlevel}, from {namedf}',
      xaxis_title=f'{timelevel}',
      yaxis_title=f'{var}',
      width=1500,
      height=500,
      template='gridon'
  )

  fig.update_xaxes(
      dtick="M1",
      tickformat="%m\n%Y",
      showgrid=True,
      ticks="outside",
      showline=True,
      tickangle=0
  )


  # fig.show()
  return fig
