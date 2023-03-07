"""
This Submodule aims to infer the actual hydrogeological properties by using either the boussinesq or maillet approach

Three parts:
    1) First we load the basin boundaries and map each of them with the DGM
    2)
    3)
"""

#%% import libs
import geopandas as gpd
import rasterio
from shapely.geometry import Point,Polygon,MultiPolygon,MultiPoint
import os
import pandas as pd
from shapely.ops import nearest_points
import numpy as np
from time import time
def extract_coords(pt):
    return pt.x, pt.y, pt.z
#%% The first function
#load the basins
basin_path = 'pegeleinzugsgebiete.gpkg'
basins = gpd.read_file(basin_path).iloc[:2,:]

#load the gw_map
gw_surface_path=os.path.join('gw_heads.tif')
gw_surface=rasterio.open(gw_surface_path)

#load the river network
river_network_path = 'river_network_z.gpkg'
river_network = gpd.read_file(river_network_path)
t_s=time()

def get_drainage_topography(basin,gw_surface=gw_surface, river_network=river_network):
    basin=gpd.GeoDataFrame(data={'basin':basin.T.basin,
                                 'area':basin.T.area,
                                 'L_represent':basin.T['L_represent']},index=[int(basin.T.value)],geometry=[basin.T.geometry],crs=river_network.crs)
    print('Check basin ',basin['basin'])
    # get the boundary of the basin
    boundary = basin.iloc[0].geometry.boundary
    
    # get the height of the boundary points from the gw map
    coords = list(boundary.coords)  
    heights=[float(x) if x!=gw_surface.nodata else None for x in gw_surface.sample(coords)]

    #generating a GeoDataFrame consisting of shapely points
    # create a GeoDataFrame of the boundary points
    gdf_basin_pts = gpd.GeoDataFrame(data={
        'x': [c[0] for c in coords],
        'y': [c[1] for c in coords],
        'z': heights
    }, geometry=[Point(x, y) for x, y in coords], crs=river_network.crs)
    #we clip the river network by the basin
    basin_network=gpd.clip(river_network,basin).explode()
    #get data as points

    gdf_network_pnts = gpd.GeoDataFrame(geometry=[Point(pt) for line in basin_network['geometry']
                                                  for pt in line.coords],
                                        crs=basin_network.crs)
    print(time()-t_s, 'seconds required for simulation')
    #get the closest point of the network for each point of the basin
    
    closest_network_pnts=[gdf_network_pnts.geometry[gdf_network_pnts.distance(basin_pnt).idxmin()] for basin_pnt in gdf_basin_pts['geometry']]
    """
    test whether this is faster
    # calculate the distance between each point in gdf_basin_pts and each point in gdf_network_pnts
    dists = np.sqrt((gdf_basin_pts['x'].values.reshape(-1, 1) - gdf_network_pnts['geometry'].x.values.reshape(1, -1)) ** 2 + 
                    (gdf_basin_pts['y'].values.reshape(-1, 1) - gdf_network_pnts['geometry'].y.values.reshape(1, -1)) ** 2)
    
    # get the index of the closest point in gdf_network_pnts for each point in gdf_basin_pts
    closest_idxs = np.argmin(dists, axis=1)
    """
    
    
    print(time()-t_s, 'seconds required for simulation')
    #get their x y and z data
    gdf_basin_pts[['x_rn','y_rn','z_rn']]=list(map(extract_coords,closest_network_pnts))
    
    # Compute the mean distance to the river and mean height above stream
    gdf_basin_pts['dist_to_stream'] = np.sqrt((gdf_basin_pts['x'] - gdf_basin_pts['x_rn']) ** 2 + (gdf_basin_pts['y'] - gdf_basin_pts['y_rn']) ** 2)
    h_m = (gdf_basin_pts['z'] - gdf_basin_pts['z_rn']).mean()
    L_mean = gdf_basin_pts['dist_to_stream'].mean()
    
    # Add the results to the basin DataFrame
    basin['h_m'] = h_m
    basin['L'] = L_mean
    
    return basin.iloc[0]

basins_out=basins.apply(lambda row: get_drainage_topography(row, gw_surface=gw_surface, river_network=river_network), axis=1)

print(time()-t_s, 'seconds required for simulation')
    
    

  



