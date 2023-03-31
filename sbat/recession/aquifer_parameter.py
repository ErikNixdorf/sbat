"""
This Submodule aims to infer the actual hydrogeological properties by using either the boussinesq or maillet approach

Three parts:
    1) First we load the basin boundaries and map each of them with the DGM
    2)
    3)
"""

from pathlib import Path
import logging
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.io import DatasetReader
from shapely.geometry import Point
from pandas.core.series import Series

aquifer_logger = logging.getLogger('sbat.aquifer_parameter')
def extract_coords(pt: Point) -> Tuple[float, float,float]:
    """
    Extracts the x and y and z coordinates from a Shapely Point object.

    Args:
        pt: A Point object representing a point in 3D space.

    Returns:
        A tuple containing the x and y and z coordinates of the point.
    """
    return pt.x, pt.y, pt.z


# %% Get drainage parameters
def get_drainage_topographic_parameters(basin: gpd.GeoDataFrame, 
                                        basin_id_col: str = 'basin',
                                        gw_surface: DatasetReader = None,
                                        river_network: gpd.GeoDataFrame = None) -> Series:
    """
    Computes topographic parameters of a given drainage basin based on a groundwater surface and a river network.
    
    Args:
    - basin (GeoDataFrame): A GeoDataFrame containing a single drainage basin to compute topographic parameters for.
    - basin_id_col (str): Column name for basin ID.
    - gw_surface (Raster): A raster file containing groundwater surface data.
    - river_network (GeoDataFrame): A GeoDataFrame containing river network data for the region.
    
    Returns:
    - basin_params (Series): A pandas Series containing the computed topographic parameters for the given drainage basin.
                            The parameters include: 
                            - basin_id: the ID of the drainage basin
                            - area: the area of the drainage basin
                            - L_represent: the representative length of the drainage basin
                            - h_m: the mean height above the river network for the boundary points of the drainage basin
                            - dist_m: the mean distance of the boundary points to the river network
                            - network_length: the total length of the river network within the drainage basin
    """
    basin = gpd.GeoDataFrame(data={'basin_id': basin.T[basin_id_col],
                                   'area': basin.T.area,
                                   'L_represent': basin.T['L_represent']}, index=[int(basin.T.value)],
                             geometry=[basin.T.geometry], crs=river_network.crs)
    basin_name = basin['basin_id'].iloc[0]
    aquifer_logger.info(f'Check basin {basin_name}')
    # get the boundary of the basin
    boundary = basin.iloc[0].geometry.boundary

    # get the height of the boundary points from the gw map
    coords = list(boundary.coords)
    heights = [float(x) if x != gw_surface.nodata else None for x in gw_surface.sample(coords)]

    # generating a GeoDataFrame consisting of shapely points
    # create a GeoDataFrame of the boundary points
    gdf_basin_pts = gpd.GeoDataFrame(data={
        'x': [c[0] for c in coords],
        'y': [c[1] for c in coords],
        'z': heights
    }, geometry=[Point(x, y) for x, y in coords], crs=river_network.crs)
    # we clip the river network by the basin
    basin_network = gpd.clip(river_network, basin).explode(index_parts=True)

    if basin_network.empty:
        aquifer_logger.info(f'No river sections within basin {basin_name}')
        basin['h_m'] = np.nan
        basin['dist_m'] = np.nan
        basin['network_length'] = np.nan
        return basin.iloc[0]

    # compute the length of the network
    network_length = basin_network.geometry.length.sum()

    # get data as points

    gdf_network_pnts = gpd.GeoDataFrame(geometry=[Point(pt) for line in basin_network['geometry']
                                                  for pt in line.coords],
                                        crs=basin_network.crs)

    # get the closest point of the network for each point of the basin

    # closest_network_pnts=[gdf_network_pnts.geometry[gdf_network_pnts.distance(basin_pnt).idxmin()] for basin_pnt in gdf_basin_pts['geometry']]

    # calculate the distance between each point in gdf_basin_pts and each point in gdf_network_pnts
    # https://stackabuse.com/guide-to-numpy-matrix-subtraction/
    dists = np.sqrt(
        (gdf_basin_pts['x'].values.reshape(-1, 1) - gdf_network_pnts['geometry'].x.values.reshape(1, -1)) ** 2 +
        (gdf_basin_pts['y'].values.reshape(-1, 1) - gdf_network_pnts['geometry'].y.values.reshape(1, -1)) ** 2)

    # get the index of the closest point in gdf_network_pnts for each point in gdf_basin_pts

    closest_idxs = np.argmin(dists, axis=1)

    closest_network_pnts = gdf_network_pnts.geometry[closest_idxs]

    # get their x y and z data
    gdf_basin_pts[['x_rn', 'y_rn', 'z_rn']] = list(map(extract_coords, closest_network_pnts))

    # Compute the mean distance to the river and mean height above stream
    gdf_basin_pts['dist_to_stream'] = np.sqrt(
        (gdf_basin_pts['x'] - gdf_basin_pts['x_rn']) ** 2 + (gdf_basin_pts['y'] - gdf_basin_pts['y_rn']) ** 2)
    h_m = (gdf_basin_pts['z'] - gdf_basin_pts['z_rn']).mean()
    L_mean = gdf_basin_pts['dist_to_stream'].mean()

    # Add the results to the basin DataFrame
    basin['h_m'] = h_m
    basin['dist_m'] = L_mean
    basin['network_length'] = network_length
    return basin.iloc[0]


# map it on the gauge_data
def map_topo_parameters(row: pd.Series, df2: pd.DataFrame, 
                        parameters: List[str] = ['h_m', 'dist_m', 'network_length']):
    """
    Maps topographic parameters from a separate DataFrame `df2` to each row of a pandas Series `row` based on a matching gauge
    ID in the two DataFrames.
    Parameters:
    -----------
    row: pandas Series
        A row of a DataFrame that contains a gauge ID column and possibly other data columns.
    df2: pandas DataFrame
        A DataFrame containing topographic parameters for basins, with a column of basin IDs matching the gauge ID column in `row`.
    parameters: list of str
        A list of topographic parameter names to map from `df2` to `row`.
    
    Returns:
    --------
    pandas Series
        A modified copy of `row` with additional columns for the mapped topographic parameters from `df2`.
    """
    if isinstance(row.name, Tuple):
        gauge_name = row.name[0]
    else:
        gauge_name = row.name

    # check whether the gauge is actually in the basin_list
    if gauge_name in df2['basin_id'].values:
        topo_params = df2[df2['basin_id'] == gauge_name][parameters].iloc[0]
    else:
        aquifer_logger.info(f'No basin parameters for gauge {gauge_name}')
        topo_params = df2.iloc[0, :][parameters]
        # replace all by nan
        topo_params[~topo_params.isna()] = np.nan
    # Change row_name if it is a tuple to a string to avoid future warning
    if len(row.name)>1:
        row.name=row.name[0]
    return pd.concat([row, topo_params])


# %% Next we map it on the recession parameters and calculate the aquifer parameters
def infer_hydrogeo_parameters(basin_data: pd.DataFrame,
                              conceptual_model: str,
                              **kwargs) -> pd.DataFrame:
    """
    Infer hydrogeological parameters from basin data using a given conceptual model.
    @ChatGPT
    Parameters
    ----------
    basin_data : pandas.DataFrame
        A DataFrame containing the basin data. It must contain the columns specified in the kwargs.
    conceptual_model : str
        The name of the conceptual model to use. Currently supported models are 'maillet' and 'boussinesq'.
    **kwargs : dict
        Additional keyword arguments required by the specified conceptual model. The required arguments
        depend on the model used. For the 'maillet' model, the following arguments are required:
        - Q0 : str
            The name of the column in basin_data containing the baseflow recession rate (m/s).
        - alpha : str
            The name of the column in basin_data containing the recession coefficient (1/s).
        - dist_m : str
            The name of the column in basin_data containing the distance to the outlet (m).
        - network_length : str
            The name of the column in basin_data containing the total length of the stream network (m).
        - h_m : str
            The name of the column in basin_data containing the average height of the basin (m) above river.
        For the 'boussinesq' and 'maillet' model, the following arguments are required:
        - Q0 : str
            The name of the column in basin_data containing the baseflow recession rate (m/s).
        - alpha : str
            The name of the column in basin_data containing the recession coefficient (1/s).
        - dist_m : str
            The name of the column in basin_data containing the distance to the outlet (m).
        - network_length : str
            The name of the column in basin_data containing the total length of the stream network (m).
        - h_m : str
            The name of the column in basin_data containing the average height of the basin (m).
    
    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns containing the inferred hydrogeological parameters.
        The names of the added columns depend on the conceptual model used.
    Raises
    ------
    KeyError
        If any of the required columns specified in the kwargs are not present in basin_data.
    ValueError
        If the specified conceptual model is not supported.
    """

    # check whether the columns we need are in the input data
    required_columns = [kwargs['Q0'], kwargs['alpha'], kwargs['dist_m'], kwargs['network_length'], kwargs['h_m']]
    for col in required_columns:
        if col not in basin_data.columns:
            raise KeyError(f"Required column '{col}' not found in basin_data")

    if conceptual_model == 'maillet':
        # based on doi:10.1016/S0022-1694(02)00418-3
        porosity_col = f'porosity_{conceptual_model}'
        transmissivity_col = f'transmissivity_{conceptual_model}'
        # first calculate the porosity, merged both equations from the source
        basin_data[porosity_col] = (basin_data[kwargs['Q0']] * np.pi) / (
                2 * basin_data[kwargs['alpha']] * basin_data[kwargs['network_length']] *
                basin_data[kwargs['h_m']] * basin_data[kwargs['dist_m']])
        # then the transmissivity
        basin_data[transmissivity_col] = (2 * basin_data[kwargs['Q0']] * basin_data[kwargs['dist_m']]) / (
                np.pi * basin_data[kwargs['network_length']] * basin_data[kwargs['dist_m']])


    elif conceptual_model == 'boussinesq':
        # based on doi:10.1016/S0022-1694(02)00418-3
        kf_value_col = f'kf_value_{conceptual_model}'
        porosity_col = f'porosity_{conceptual_model}'
        # kf value
        basin_data[kf_value_col] = (basin_data[kwargs['Q0']] * basin_data[kwargs['dist_m']]) / (
                1.724 * basin_data[kwargs['h_m']] ** 2 * basin_data[kwargs['network_length']])
        # porosity
        basin_data[porosity_col] = (1.115 * basin_data[kf_value_col] * basin_data[kwargs['h_m']]) / (
                basin_data[kwargs['alpha']] * basin_data[kwargs['dist_m']] ** 2)
    else:
        raise ValueError(f"Invalid conceptual model '{conceptual_model}'")

    return basin_data


# %% Wrapper Function to call all functions
def get_hydrogeo_properties(
    gauge_data: pd.DataFrame = pd.DataFrame(),
    basins: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    gw_surface: str = 'rasterio',
    network: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    conceptual_model: str = 'maillet',
    basin_id_col: str = 'basin',
    ) -> pd.DataFrame:
    """
    Calculate hydrogeological properties for a set of gauging stations based on given inputs.

    Args:
        gauge_data (pd.DataFrame, optional): Dataframe containing information on gauging stations. Defaults to pd.DataFrame().
        basins (gpd.GeoDataFrame, optional): Geodataframe containing information on drainage basins. Defaults to gpd.GeoDataFrame().
        gw_surface (str, optional): Type of surface representation for groundwater flow calculations. Defaults to 'rasterio'.
        network (gpd.GeoDataFrame, optional): Geodataframe containing information on river networks. Defaults to gpd.GeoDataFrame().
        conceptual_model (str, optional): Type of conceptual hydrological model used to calculate properties. Defaults to 'maillet'.
        basin_id_col (str, optional): Name of the column containing basin IDs in the basins dataframe. Defaults to 'basin'.

    Returns:
        pd.DataFrame: Dataframe containing hydrogeological properties for the input gauging stations.
    """    
    # %% run the steps
    # get drainage parameters
    basins_out = basins.apply(
        lambda row: get_drainage_topographic_parameters(row, gw_surface=gw_surface, basin_id_col=basin_id_col,
                                                        river_network=network), axis=1)
    # map to gauge data
    gauge_data = gauge_data.apply(lambda x: map_topo_parameters(x, basins_out), axis=1)
    # get hydrogeological parameters
    gauge_data = infer_hydrogeo_parameters(basin_data=gauge_data,
                                           conceptual_model=conceptual_model,
                                           Q0='Q0_rec',
                                           alpha='n0_rec',
                                           dist_m='dist_m',
                                           network_length='network_length',
                                           h_m='h_m')
    return gauge_data

