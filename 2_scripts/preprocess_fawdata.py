import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def check():
  bound = gpd.read_file("data/spatial_data/gadm41_KEN_1.json")
  return bound

def check_inbound_latlon(df, boundary= 'kenya'):
  # Convert to GeoDataFrame
  geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
  gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

  # Load Kenya boundary
  if boundary != 'kenya':
    bound = gpd.read_file(boundary)
  else:
    bound = gpd.read_file("data/spatial_data/gadm41_KEN_1.json")  # adjust filename as needed

  # Ensure same CRS
  bound_same_crs = bound.to_crs(gdf_points.crs)

  # Spatial join: keep only points within Kenya
  gdf_points_in = gdf_points[gdf_points.within(bound_same_crs.union_all())]
  gdf_points_notin = gdf_points[~gdf_points.within(bound_same_crs.union_all())]
  # Reset index if needed
  gdf_points_in = gdf_points_in.reset_index(drop=True)
  gdf_points_notin = gdf_points_notin.reset_index(drop=True)
  print('There are', gdf_points_notin.shape[0], 'points not in the boundary, which being removed')
  print('There are', gdf_points_in.shape[0], 'points in the boundary, which are returned')

  return gdf_points_in.reset_index(drop=True)

def search_county_state(df, merged = False):
  from geopy.geocoders import Nominatim
  from geopy.extra.rate_limiter import RateLimiter
  import pandas as pd

  # Set up geocoder
  geolocator = Nominatim(user_agent="geoapi",  timeout=5)
  reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

  # Loop through each row in the KenyaTrap DataFrame
  address_all = []
  states_all = []
  counties_all = []

  for i, row in df.iterrows():
      lat, lon = row['lat'], row['lon']
      try:
          location = reverse((lat, lon), language='en')
          if location:
              address = location.raw.get('address', {})
              state = address.get('state')
              county = address.get('county')
          else:
              address, state, county = None, None, None
      except Exception as e:
          address, state, county = None, None, None
          print(f"Error at index {i}: {e}")
      address_all.append(address)
      states_all.append(state)
      counties_all.append(county)
      if i % 10 == 0:  # print every 10 rows
          print(f"Processed {i} rows out of {len(df)}")
  new_df = pd.DataFrame({
    'search_address': address_all,
    'search_state': states_all,
    'search_county': counties_all})
  if merged:
    new_df = pd.concat([df, new_df], axis=1)

  return new_df

import re

def clean_county_name(county_name):
    # Remove " County" and any leading/trailing spaces
    cleaned_name = county_name.replace(' County', '').strip()
    # Remove any remaining symbols (e.g., hyphens, apostrophes)
    cleaned_name = re.sub(r"[-'_,.;:!?(){}\[\]\s+]", '', cleaned_name)
    return cleaned_name
