import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import re

def check():
  bound = gpd.read_file("data/spatial_data/gadm41_KEN_1.json")
  return bound

def check_inbound_latlon(df, boundary= 'kenya'):
  # convert points in 'lon' and 'lat' cols to GeoDataFrame
  geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
  gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

  # load Kenya boundary
  if boundary != 'kenya':
    bound = gpd.read_file(boundary)
  else:
    bound = gpd.read_file("data/spatial_data/gadm41_KEN_1.json")  # adjust filename as needed

  # ensure same CRS
  bound_same_crs = bound.to_crs(gdf_points.crs)

  # Spatial join: keep only points within Kenya
  gdf_points_in = gdf_points[gdf_points.within(bound_same_crs.union_all())]
  gdf_points_notin = gdf_points[~gdf_points.within(bound_same_crs.union_all())]
  # Reset index if needed
  gdf_points_in = gdf_points_in.reset_index(drop=True)
  gdf_points_notin = gdf_points_notin.reset_index(drop=True)

  print('There are', gdf_points_notin.shape[0], 'points not in the boundary, which have been removed')
  print('There are', gdf_points_in.shape[0], 'points in the boundary, which are returned')

  return gdf_points_in

def search_county_state(df, merged = False):

  # set up geocoder
  geolocator = Nominatim(user_agent="geoapi",  timeout=5)
  reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

  # loop through each row in a given df
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
      # print every 10 rows
      if i % 10 == 0:
          print(f"Processed {i} rows out of {len(df)}")

  # create dataframe of all addresses
  new_df = pd.DataFrame({
    'search_address': address_all,
    'search_state': states_all,
    'search_county': counties_all})

  if merged:
    new_df = pd.concat([df, new_df], axis=1)

  return new_df


def clean_county_name(county_name):
    # remove " County" and any leading/trailing spaces
    cleaned_name = county_name.replace(' County', '').strip()
    # remove any remaining symbols (e.g., hyphens, apostrophes)
    cleaned_name = re.sub(r"[-'_,.;:!?(){}\[\]\s+]", '', cleaned_name)
    return cleaned_name


def spatial_partition_kmeans(df, lon_col='lon', lat_col='lat', n_partitions=10,
                              n_clusters=18, random_state=None):
    """
    Partition points into n sets with spatial balance using K-means clustering.
    Adapts to data density - creates more clusters where data is dense.

    Parameters:
    - df: DataFrame with lat/lon columns
    - lon_col: name of longitude column
    - lat_col: name of latitude column
    - n_partitions: number of partitions to create (default: 10)
    - n_clusters: number of spatial clusters to create (default: 18, should be >= n_partitions)
    - random_state: random seed for reproducibility

    Returns:
    - DataFrame with added 'partition' column (values 0 to n_partitions-1)
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    np.random.seed(random_state)

    df = df.copy()

    # prepare coordinates for clustering
    coords = df[[lon_col, lat_col]].values

    # cluster points based on spatial proximity
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)

    # initialize partition assignment
    df['partition'] = -1

    # For each cluster, distribute points across partitions
    for cluster_id in range(n_clusters):
        cluster_mask = df['cluster'] == cluster_id
        cluster_indices = df[cluster_mask].index.tolist()

        # shuffle points within cluster
        np.random.shuffle(cluster_indices)

        # assign to partitions in round-robin fashion
        for i, idx in enumerate(cluster_indices):
            df.loc[idx, 'partition'] = i % n_partitions

    # drop temporary cluster column
    # df = df.drop(columns=['cluster'])

    # Print summary
    print(f"Created {n_partitions} spatially balanced partitions using {n_clusters} K-means clusters")
    print(f"\nPartition sizes:")
    partition_counts = df['partition'].value_counts().sort_index()
    for p, count in partition_counts.items():
        print(f"  Partition {p}: {count} points")

    return df


def thin_by_degree_distance(df, lon_col='lon', lat_col='lat', min_dist=4.5):
    """
    Thin points by minimum degree distance (not accurate for real distance, but simpler).

    Parameters:
    - df: DataFrame with lat/lon columns
    - min_deg: minimum allowed separation in degrees

    Returns:
    - Thinned DataFrame
    """
    from shapely.geometry import Point
    import geopandas as gpd
    from sklearn.neighbors import BallTree
    import numpy as np

    # convert to GeoDataFrame with EPSG:4326
    gdf = gpd.GeoDataFrame(df.copy(),
                           geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                           crs="EPSG:4326")


    # convert degree (Lat, Lon) to radius for Haversine distance calc
    coords = np.deg2rad(np.c_[gdf[lat_col], gdf[lon_col]])  #  Now Lat, Lon in radians

    # creates a Ball Tree spatial index for efficient nearest-neighbor searches using the haversine distance metric.
    tree = BallTree(coords, metric='haversine')
	  # Haversine distance is the angular distance between two points on the surface of a sphere. Accounts for Earth's curvature
    # return points in a tree for fast spatial queries, output is in radians

    min_rad = min_dist/6371  # convert distance to radians

    # count neighbors for each point within min_dist
    neighbor_counts = np.array([
        len(tree.query_radius([coords[i]], r=min_rad)[0]) - 1  # subtract 1 to exclude self
        for i in range(len(coords))
    ])

    # sort indices by neighbor count (ascending - least dense first)
    sorted_indices = np.argsort(neighbor_counts)

    # initialise - set default to true for all points
    keep = np.full(len(gdf), True)

    for i in sorted_indices:
      if keep[i]:
        # find all points within that radius from a single point i and we unpacks it
        ind = tree.query_radius([coords[i]], r=min_rad)[0]
        nd = ind[ind != i] # remove the point itself
        # then set the rest to FALSE (for removal)
        keep[ind] = False
        keep[i] = True

    thinned = gdf[keep].reset_index(drop=True)

    print(f"Thinned from {len(gdf)} => {len(thinned)} points (≥ ~{min_dist} km or {min_rad:.2e} radians apart)")
    # thinned = thinned.drop(columns=['lat', 'lon'])
    return thinned
