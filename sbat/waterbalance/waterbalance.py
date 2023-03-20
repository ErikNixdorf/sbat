"""
This script analysis the water balance between river sections
A section is either the branch between two gauges or betweeen a gauge and river_mouth and vise versa

#next tasks
confidence_interval
hash_

"""

from copy import deepcopy
import logging
import secrets
from typing import Dict, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, MultiPolygon
from shapely.ops import nearest_points, unary_union


def multilinestring_to_singlelinestring(
        multilinestring,
        start_point,
        search_radius: float = 2000,
        min_distance: float = 200,
):
    """
    Converts a multilinestring to a single linestring starting from a given point.

    Parameters
    ----------
    multilinestring : shapely.geometry.MultiLineString
        The vertices of the points of which the multilinestring consists.
    start_point : shapely.geometry.Point
        The starting point from where to check for contiguous closest distance, e.g a river outlet.
    search_radius : float, optional
        The maximum distance to search for the next closest point, by default 2000.
    min_distance : float, optional
        The minimum distance required to accept a point as the next closest point, by default 200.

    Returns
    -------
    List[shapely.geometry.Point]
        The sorted vertices of the single linestring.
    """
    # start logging
    logger = logging.getLogger(__name__)
    # one annoything is that the Neisse River is not a nicely sorted linestring, so we need to resort all vertices, from downstream to upstream
    closest_point = deepcopy(start_point)
    sorted_point_bag = [start_point]
    # first remove the start point from the bag
    unsorted_point_bag = multilinestring.difference(closest_point)
    # start to loop
    for iteration in range(0, len(unsorted_point_bag)):
        # get the newest closest point
        closest_point_updated, _ = nearest_points(unsorted_point_bag, closest_point)
        # calculate the distance between the two points
        pp_distance = closest_point.distance(closest_point_updated)

        # if distance is too close remove point and continue
        if pp_distance < min_distance:
            logger.info(f"Points too close ({pp_distance:.2f}m), removing point")
            unsorted_point_bag = unsorted_point_bag.difference(closest_point_updated)
            continue

        # remove the closest point from unsorted point bag and add to sorted point bag
        if pp_distance < search_radius:
            unsorted_point_bag = unsorted_point_bag.difference(closest_point_updated)
            sorted_point_bag.append(closest_point_updated)
            closest_point = deepcopy(closest_point_updated)
        else:
            logger.info(f"Closest point too far ({pp_distance:.2f}m) from existing line, probably a branch")
            logger.debug(f"Loop stopped at iteration {iteration}, try to increase resolution to avoid")
            break
    return sorted_point_bag

#%% We loop trough the gauges in order to define the geometry
def generate_upstream_network(
    gauge_meta=pd.DataFrame(), network_connections=pd.DataFrame()
):

    gauges_connection_dict = dict()

    #%% The Idea is we loop trough the gauges, find the upstream stream and calculate the section water balance
    # first we split the network_connections to tributaries and distributarys because they need to be treated differenttly
    tributary_connections = network_connections[
        network_connections["type"] == "tributary"
    ]
    distributary_connections = network_connections[
        network_connections["type"] == "distributary"
    ]

    for i, gauge in gauge_meta.iterrows():
        # first we create a hexadecimal string
        gauge_connection_dict = dict()

        # first we create a hex string which helps us
        gauge_connection_dict["id"] = secrets.token_hex(4)

        gauge_connection_dict["reach_name"] = gauge["stream"]

        # check whether there is an upstream gauge in the system
        stream_gauges = gauge_meta[gauge_meta["stream"] == gauge["stream"]]
        stream_gauges["upstream_distance"] = (
            stream_gauges["distance_to_mouth"] - gauge["distance_to_mouth"]
        )
        stream_gauges = stream_gauges[stream_gauges["upstream_distance"] > 0]
        upstream_gauge = stream_gauges.copy()
        # tributaries upstream
        tributaries = tributary_connections.loc[
            tributary_connections["main_stream"] == gauge["stream"], :
        ]
        tributaries["upstream_distance"] = (
            tributaries["distance_junction_from_receiving_water_mouth"]
            - gauge["distance_to_mouth"]
        )
        tributaries = tributaries[tributaries["upstream_distance"] > 0]

        # similar we look for distributaries upstream
        distributaries = distributary_connections.loc[
            distributary_connections["main_stream"] == gauge["stream"], :
        ]
        distributaries["upstream_distance"] = (
            distributaries["distance_junction_from_receiving_water_mouth"]
            - gauge["distance_to_mouth"]
        )
        distributaries = distributaries[distributaries["upstream_distance"] > 0]

        # if we have a stream gauge upstream we will reduce the tributaries and distributaries to the ones between both gauges
        if len(stream_gauges) > 0:

            upstream_gauge = stream_gauges.loc[
                [stream_gauges.upstream_distance.idxmin()]
            ]
            tributaries = tributaries.loc[
                (
                    tributaries["upstream_distance"]
                    - upstream_gauge["upstream_distance"].values
                )
                < 0,
                :,
            ]
            distributaries = distributaries.loc[
                (
                    distributaries["upstream_distance"]
                    - upstream_gauge["upstream_distance"].values
                )
                < 0,
                :,
            ]

        # connect
        gauge_connection_dict["gauge_up"] = pd.DataFrame(upstream_gauge)
        gauge_connection_dict["tributaries_up"] = tributaries.copy()
        gauge_connection_dict["distributaries_up"] = distributaries.copy()

        # we check for subdistributaries/subtributaries with gauges on a second level
        for branch_type in ["tributaries_up", "distributaries_up"]:
            for _, branch in gauge_connection_dict[branch_type].iterrows():
                # first we check whether there is a subtributary between closest gauge and the river mouth
                subtributaries = tributary_connections.loc[
                    tributary_connections["main_stream"] == branch["stream"], :
                ]
                subdistributaries = distributary_connections.loc[
                    distributary_connections["main_stream"] == branch["stream"], :
                ]

                # calculate the most downstream gauge of the tributary
                branch_gauges = gauge_meta[gauge_meta["stream"] == branch["stream"]]
                if len(branch_gauges) == 0:
                    print("No Gauge at tributary ", branch["stream"])
                    continue
                # take the one closest to the river mouth
                if branch_type == "tributaries_up":
                    branch_gauge = branch_gauges.loc[
                        branch_gauges["distance_to_mouth"].idxmin(), :
                    ]
                elif branch_type == "distributaries_up":
                    branch_gauge = branch_gauges.loc[
                        branch_gauges["distance_to_mouth"].idxmax(), :
                    ]

                # calculate whether there is an inflow inbetween:
                if branch_type == "tributaries_up":
                    subtributaries = subtributaries[
                        (
                            subtributaries[
                                "distance_junction_from_receiving_water_mouth"
                            ]
                            - branch_gauge["distance_to_mouth"]
                        )
                        < 0
                    ]
                    subdistributaries = subdistributaries[
                        (
                            subdistributaries[
                                "distance_junction_from_receiving_water_mouth"
                            ]
                            - branch_gauge["distance_to_mouth"]
                        )
                        < 0
                    ]
                elif branch_type == "distributaries_up":
                    subtributaries = subtributaries[
                        (
                            subtributaries[
                                "distance_junction_from_receiving_water_mouth"
                            ]
                            - branch_gauge["distance_to_mouth"]
                        )
                        > 0
                    ]
                    subdistributaries = subdistributaries[
                        (
                            subdistributaries[
                                "distance_junction_from_receiving_water_mouth"
                            ]
                            - branch_gauge["distance_to_mouth"]
                        )
                        > 0
                    ]

                # append to data
                gauge_connection_dict[branch_type] = pd.concat(
                    [gauge_connection_dict[branch_type], subtributaries]
                )
                gauge_connection_dict[branch_type] = pd.concat(
                    [gauge_connection_dict[branch_type], subdistributaries]
                )

        # define the name of the upstream points
        if len(gauge_connection_dict["gauge_up"]) == 0:
            # we first define upstream and downstream_points
            if (
                len(gauge_connection_dict["tributaries_up"]) == 0
                and len(gauge_connection_dict["distributaries_up"]) == 0
                and gauge["stream"] in distributary_connections["stream"].to_list()
            ):
                gauge_connection_dict["upstream_point"] = "river_junction"
            else:
                gauge_connection_dict['upstream_point'] = 'river_spring'
        else:
            gauge_connection_dict['upstream_point'] = gauge_connection_dict['gauge_up'].index.values[0]

        # downstream is always the same
        gauge_connection_dict['downstream_point'] = gauge.name

        # check for tributary or distributary gauges between the spring and the gauge
        # generate a copy of gauge_meta
        gauge_meta_reset = gauge_meta.copy().reset_index()
        for trib_type in ['tributaries', 'distributaries']:

            # if there are tributary_gauges
            if len(gauge_connection_dict[trib_type + '_up']) > 0:

                # first we check whether the tributaries have gauges

                tribs_with_gauges = gauge_connection_dict[trib_type + '_up'
                        ][gauge_connection_dict[trib_type + '_up']['stream'
                          ].isin(gauge_meta['stream'])].set_index('stream')

                if not tribs_with_gauges.empty:

                    # get_the_tributary_gauges which is the most downstream_gauge

                    trib_gauges = gauge_meta_reset.loc[gauge_meta_reset['stream'
                            ].isin(tribs_with_gauges.index), :]

                    # we only select the ones which are most downstream

                    if trib_type == 'tributaries':
                        trib_gauges = trib_gauges.loc[trib_gauges.groupby('stream'
                                )['distance_to_mouth'].idxmin()].set_index('stream')
                    elif 'distributaries':
                        trib_gauges = trib_gauges.loc[trib_gauges.groupby('stream'
                                )['distance_to_mouth'].idxmax()].set_index('stream')
                    tribs_with_gauges['upstream_point'] = trib_gauges.gauge
                    tribs_with_gauges['downstream_point'] = 'river_mouth'
                    tribs_with_gauges = tribs_with_gauges.reset_index()

                # we also add the tribs with no gauges

                tribs_without_gauges = gauge_connection_dict[trib_type + '_up'
                        ][~gauge_connection_dict[trib_type + '_up']['stream'
                          ].isin(gauge_meta_reset['stream'])]
                if len(tribs_without_gauges) > 0:
                    tribs_without_gauges['upstream_point'] = 'river_spring'
                    tribs_without_gauges['downstream_point'] = 'river_mouth'

                # merge

                tribs_merged = pd.concat([tribs_with_gauges, tribs_without_gauges],
                                         ignore_index=True)

                gauge_connection_dict[trib_type + '_up'] = tribs_merged

            gauges_connection_dict.update({gauge.name: gauge_connection_dict})


    return gauges_connection_dict


# %% Next function how to calculate the balance
def calculate_network_balance(ts_data=pd.DataFrame(),
                              network_dict=dict(),
                              confidence_acceptance_level=0.05):
    """
    Calculates the water balance for a network of gauges using time series data and 
    a dictionary representing the network topology.
    
    Parameters:
    -----------
    ts_data : pd.DataFrame(), optional
        A DataFrame containing time series data for each gauge. Default is an empty DataFrame.
        
    network_dict : dict, optional
        A dictionary representing the topology of the network, with each key representing a gauge and the corresponding value being another dictionary containing information about that gauge's upstream and downstream connections. Default is an empty dictionary.
        
    confidence_acceptance_level : float, optional
        A float representing the confidence interval below which water balance values are set to NaN. Default is 0.05.
    
    Returns:
    --------
    sections_meta : pd.DataFrame()
        A DataFrame containing information about each gauge, including the water balance for each time step.
        
    q_diff : pd.DataFrame()
        A pivot table showing the water balance for each gauge at each time step.
    """

    sections_meta = list()
    nr_ts = ts_data.shape[0]

    # Create list of gauge keys
    gauge_keys = network_dict.keys()
    # loop over gauges
    for gauge in gauge_keys:

        # we write some empty dataframes for the tributaries
        ts_distributaries = pd.Series(np.zeros((nr_ts)), index=ts_data.index)
        ts_tributaries = ts_distributaries.copy()
        ts_data_gauge_up = ts_distributaries.copy()

        print('add water balance to gauge ', gauge)

        # generate an empty dataframe to fill the balance data for each time step
        gauge_boundary_names = {k: network_dict[gauge][k] for k in ['upstream_point', 'downstream_point'] if
                                k in network_dict[gauge]}
        df_row = pd.DataFrame(gauge_boundary_names, index=[0])
        # create a section dataframe for each time step
        df_section = df_row.reindex(index=pd.RangeIndex(nr_ts), method='ffill')

        # get the topology parameters as variables
        upstream_point = network_dict[gauge]['upstream_point']
        distributaries_up = network_dict[gauge].get('distributaries_up', None)
        tributaries_up = network_dict[gauge].get('tributaries_up', None)
        gauge_up = network_dict[gauge].get('gauge_up', None)
        # we get parts of water balance from the network

        # first case is easy if upstream is a distributary, we cant give any balance, it is actually covered in another scenario
        if upstream_point == 'river_junction':

            # in this case balance is nan            
            df_section['balance'] = np.nan
        else:
            # in all other cases we just compute them from distributary and tributary gauges
            if not gauge_up.empty:
                ts_data_gauge_up = ts_data[gauge_up.index.tolist()].sum(axis=1)

                # get discharge of all distributary gauges
            if not distributaries_up.empty:
                distri_gauges = distributaries_up[distributaries_up.upstream_point != 'river_spring']
                if not distri_gauges.empty:
                    ts_distributaries = ts_data[distri_gauges.upstream_point.tolist()].sum(axis=1)
                else:
                    ts_distributaries = 0

            # get discharge of all tributary gauges
            if not tributaries_up.empty:
                tri_gauges = tributaries_up[tributaries_up.upstream_point != 'river_spring']
                if len(tri_gauges) > 0:
                    ts_tributaries = ts_data[tri_gauges.upstream_point.tolist()].sum(axis=1)
                else:
                    ts_tributaries = 0

            # calculate the water balance
            balance = ts_data[gauge] - ts_data_gauge_up - ts_tributaries + ts_distributaries
            df_section['balance'] = balance.values
            df_section['balance_confidence'] = np.divide(balance.values, (ts_data[gauge] + ts_data_gauge_up).values)

        # add index data
        df_section['Date'] = ts_data.index
        print(f'Water balance added to gauge {gauge}')

        sections_meta.append(df_section)

    # Finally we get out the data

    sections_meta = pd.concat(sections_meta)
    # remove all balances below confidence interval
    low_confidence_mask = abs(sections_meta['balance_confidence']) < confidence_acceptance_level
    sections_meta.loc[low_confidence_mask, 'balance'] = np.nan
    q_diff = sections_meta.pivot(index='Date', columns='downstream_point', values='balance')

    return sections_meta, q_diff


def map_network_sections(
        network_dict: Dict,
        gauge_meta: pd.DataFrame,
        network: gpd.GeoDataFrame,
):
    """
    Parameters
    ----------
    network_dict : dict
        A dictionary containing the gauge sections.
    gauge_meta : pd.DataFrame
        A DataFrame containing the metadata for the gauges.
    network : gpd.GeoDataFrame
        A GeoDataFrame containing the river network.

    Returns
    -------
    gdf_balances : gpd.GeoDataFrame
        A GeoDataFrame containing the sections of the river network.
    """

    # %% we write a function to get the basin_data

    gdf_balances = gpd.GeoDataFrame()

    # %% We will loop through the gauge sections and extract the relevant stream reaches and clip them
    for _, gauge in network_dict.items():
        # first we get the line within the main reach
        if gauge['reach_name'] not in network.reach_name.tolist():
            print(gauge['reach_name'], 'not in network data, check correct names')
            continue
        river_line = network[network.reach_name == gauge['reach_name']].geometry.iloc[0]
        pnt_gauge = gauge_meta.loc[gauge_meta.index == gauge['downstream_point'], 'geometry'].iloc[0]

        # get all river points as geodataframe
        river_pnts = gpd.GeoDataFrame(geometry=[Point(pt) for pt in river_line.coords])
        # reduce the points to all upstream the river gauge
        pnt_id_downstream = river_pnts.iloc[river_pnts.distance(pnt_gauge).idxmin():, :].iloc[0].name

        # now we make some conditions depending on the upstream point
        if gauge['upstream_point'] == 'river_spring' or gauge['upstream_point'] == 'river_junction' or gauge[
            'upstream_point'] == 'river_mouth':
            pnt_id_upstream = river_pnts.iloc[-1, :].name
        else:
            pnt_gauge_up = gauge_meta.loc[gauge_meta.index == gauge['upstream_point'], 'geometry'].iloc[0]
            pnt_id_upstream = river_pnts.iloc[river_pnts.distance(pnt_gauge_up).idxmin():, :].iloc[0].name

        # depending which is larger we select from the reach, because some times there are problems with the stream points
        if pnt_id_upstream > pnt_id_downstream:
            river_pnts = river_pnts.iloc[pnt_id_downstream:pnt_id_upstream + 1]
        elif pnt_id_upstream < pnt_id_downstream:
            river_pnts = river_pnts.iloc[pnt_id_upstream:pnt_id_downstream + 1]
        else:
            print('Stream Line needs at least two points, we move by one point, but you should check geometry')
            river_pnts = river_pnts.iloc[pnt_id_upstream:pnt_id_downstream + 2]


        section_line = LineString(river_pnts.geometry)
        #get the lines of the tributaries
        trib_lines = list()
        
        for branch_name in ['tributaries_up', 'distributaries_up']:
        
            # we loop trough the dataset        
            for (_, branch) in gauge[branch_name].iterrows():        
                # extract the river line if available        
                if branch['stream'] not in network.reach_name.tolist():
                    print (branch['stream'],
                           'not in network data, check correct names')
                    continue
                river_line = network[network.reach_name == branch['stream'
                                     ]].geometry.iloc[0]
        
                # first we check whether there is really data in the dataset        
                if len(gauge[branch_name]) > 0:        
                    # we get the river line        
                    if branch['stream'] not in network.reach_name.tolist():        
                        print (branch['stream'],
                               'not in network data, check correct names')
                        continue
        
                    # extract the river line        
                    river_line = network[network.reach_name == branch['stream'
                                         ]].geometry.iloc[0]
        
                    # get all river points as geodataframe        
                    river_pnts = gpd.GeoDataFrame(geometry=[Point(pt) for pt in
                            river_line.coords])
        
                    # next we will find upstream and downstream
                    # downstream is always the points        
                    if branch_name == 'tributaries_up':
                        pnt_id_downstream = 0

                        if branch.upstream_point == 'river_spring':
                            pnt_id_upstream = len(river_pnts) - 1

                    elif branch_name == 'distributaries_up':
                        pnt_id_downstream = len(river_pnts) - 1
                        if branch.upstream_point == 'river_spring':
                            pnt_id_upstream = 0
                    # upstream depends whether there is a gauge or not
                    if branch.upstream_point != 'river_spring':
                        pnt_branch_gauge = \
                            gauge_meta.loc[gauge_meta.index == branch['upstream_point'], 'geometry'].iloc[0]
                        pnt_id_upstream = river_pnts.distance(pnt_branch_gauge).idxmin()

                    # we rearrange if it does make sense with the flow direction
                    if pnt_id_upstream > pnt_id_downstream:
                        river_pnts_extracted = river_pnts.iloc[pnt_id_downstream:pnt_id_upstream + 1]
                    elif pnt_id_upstream < pnt_id_downstream:
                        river_pnts_extracted = river_pnts.iloc[pnt_id_upstream:pnt_id_downstream + 1]
                    else:
                        print(
                            'Stream Line needs at least two points, we move by one point, but you should check geometry')
                        river_pnts_extracted = river_pnts.iloc[pnt_id_upstream:pnt_id_downstream + 2]

                    # we append the trib geometries
                    trib_lines.append(LineString(river_pnts_extracted.geometry))
        trib_lines.append(section_line)
        section_lines = MultiLineString(lines=trib_lines)

        # if we have tributaries we also take their geometrical information
        # for

        # we make a dataframe
        df_columns_dict = dict(
            (k, gauge[k]) for k in ['id', 'reach_name', 'upstream_point', 'downstream_point'] if k in gauge.keys())
        gdf_balance = gpd.GeoDataFrame(pd.DataFrame.from_dict({0: df_columns_dict}).T, geometry=[section_lines],
                                       crs=network.crs)

        # append
        gdf_balances = gdf_balances.append(gdf_balance, ignore_index=True)

    return gdf_balances.set_crs(network.crs)


# add a function for time series manipulation
def aggregate_time_series(data_ts, analyse_option='overall_mean'):
    if analyse_option is None:
        print('No data aggregation option select, continue with original time series')
        return data_ts

    # just for testing we take the mean
    if analyse_option == 'overall_mean':
        print(analyse_option, 'takes entire time series')
        ts_stats = data_ts.mean()
        stats_name = 'mean_discharge_m_s'
        ts_stats = ts_stats.rename(stats_name).to_frame().T

    elif analyse_option == 'annual_mean':
        ts_stats = data_ts.resample('Y').mean()

    elif analyse_option == 'summer_mean':
        print('Calculating summer mean (June to September)')
        ts_stats = data_ts.loc[data_ts.index.month.isin([6, 7, 8, 9])].resample('Y').mean()

    # daily calculations
    elif analyse_option == 'daily':
        print('Calculating daily statistics')
        ts_stats = data_ts.copy()
    else:
        print('Invalid aggregation option selected, continuing with original time series')
        return data_ts

    ts_stats.index = ts_stats.index.strftime("%Y-%m-%d")

    return ts_stats


# get the section basins
def get_section_basins(basins: gpd.GeoDataFrame,
                       network_dict: Dict[str, Any],
                       basin_id_col: str = 'basin'
                       ):
    """
    Computes the section basins for each gauge in a river network, given a GeoDataFrame of basin polygons and a dictionary
    describing the network topology. The section basin is the downstream basin of a gauge, clipped by the upstream basins of
    any tributary or distributary gauges.

    Parameters:
    -----------
    basins : gpd.GeoDataFrame
        A GeoDataFrame containing the basin polygons for the river network.
    network_dict : dict
        A dictionary describing the topology of the river network, with gauge IDs as keys and topology parameters as values.
    basin_id_col : str
        The name of the column in the basins GeoDataFrame containing the basin IDs.

    Returns:
    --------
    section_basins : gpd.GeoDataFrame
        A GeoDataFrame containing the section basins for each gauge in the river network.
    """

    # loop over gauges
    section_basins = list()
    gauge_keys = network_dict.keys()
    for gauge in gauge_keys:

        # get the topology parameters as variables
        upstream_point = network_dict[gauge]['upstream_point']
        distributaries_up = network_dict[gauge].get('distributaries_up')
        tributaries_up = network_dict[gauge].get('tributaries_up')
        gauge_up = network_dict[gauge].get('gauge_up')

        # We start by defining the section basin as the downstream gauge basin
        section_basin = basins.loc[basins['basin'] == gauge]

        # only if there is no river junction upstream we can compute the subbasin
        if upstream_point != 'river_junction':

            # Clip the gauge_basin with the upstream basin
            if not gauge_up.empty:
                gauge_up_basin = basins[basins['basin'] == gauge_up.index[0]]
                # get difference with upstream basin
                section_basin = gpd.overlay(section_basin, gauge_up_basin, how='difference')

            # get all tributary gauges
            if not tributaries_up.empty:
                tri_gauges = tributaries_up[tributaries_up.upstream_point != 'river_spring']
                # get difference with tributary gauges
                for tri_gauge in tri_gauges.upstream_point:
                    section_basin = gpd.overlay(section_basin, basins.loc[basins['basin'] == tri_gauge],
                                                how='difference')

            # if Multipolygon is result of clipping we reduce to largest polygon

            if not section_basin.empty and isinstance(section_basin.geometry.iloc[0], MultiPolygon):
                section_basin.geometry = [max(section_basin.iloc[0].geometry, key=lambda a: a.area)]

            # get the next distributary gauge and add their basin
            if not distributaries_up.empty:
                distri_gauges = distributaries_up[distributaries_up.upstream_point != 'river_spring']
                for distri_gauge in distri_gauges.upstream_point:
                    section_basin.iloc[0].geometry = unary_union(
                        [section_basin.iloc[0].geometry, basins.loc[basins['basin'] == distri_gauge].iloc[0].geometry])

                    # we dissolve the geometries
                    section_basin = section_basin.dissolve()

            # append
            section_basins.append(section_basin)

    # merge
    section_basins = pd.concat(section_basins)
    # overwrite some layers
    # assuming a representative Circle, we assume the radius of this circle is the mean length towards the stream
    section_basins['area'] = section_basins.geometry.area
    section_basins['L_represent'] = np.sqrt(section_basins['area'] / np.pi)

    return section_basins


# %% A function which connects all functions
def get_section_water_balance(gauge_data: pd.DataFrame = pd.DataFrame(),
                              data_ts: pd.DataFrame = pd.DataFrame(),
                              network: gpd.GeoDataFrame = gpd.GeoDataFrame(),
                              basins: gpd.GeoDataFrame = gpd.GeoDataFrame(),
                              network_connections: pd.DataFrame = pd.DataFrame(),
                              confidence_acceptance_level: float = 0.05,
                              time_series_analysis_option: str = 'overall_mean',
                              basin_id_col: str = 'basin',
                              ):

    """
    Calculates the water balance for a network of stream gauges and their upstream
    and downstream connections, based on time series data and metadata.

    Parameters
    ----------
    gauge_data : pandas.DataFrame, optional
        The metadata for the stream gauges, with columns for 'ostwert' (longitude),
        'nordwert' (latitude), and any other relevant information. The index of the
        DataFrame should match the columns of the `data_ts` DataFrame. The default is
        an empty DataFrame.
    data_ts : pandas.DataFrame, optional
        The time series data for the stream gauges, with timestamps as index and
        gauge IDs as columns. The default is an empty DataFrame.
    network : geopandas.GeoDataFrame, optional
        A network representation of the streams, with columns for 'source' (ID of the
        upstream stream), 'target' (ID of the downstream stream), and 'geometry'
        (shapely LineString representing the stream segment). The default is an empty
        GeoDataFrame.
    basins : geopandas.GeoDataFrame, optional
        Hydrological catchments belonging to each gauge in the gauge data
    tributary_connections : pandas.DataFrame, optional
        A table with columns for 'source' (ID of the upstream stream) and 'target'
        (ID of the downstream tributary stream). The default is an empty DataFrame.

    confidence_acceptance_level : float, optional
        The significance level (alpha) used for the confidence interval of the water
        balance calculations. The default is 0.05.
    time_series_analysis_option : str, optional
        The method used to aggregate the time series data. The default is 'overall_mean'.
    basin_id_col : str, optional
        The name of the column in `basins` DataFrame that contains basin IDs. The default is 'basin'.

    Returns
    -------
    sections_meta : pandas.DataFrame
        The metadata for each section of the network, with columns for 'section_id',
        'upstream_ids', 'downstream_ids', 'length', 'centroid', and 'geometry'.
    q_diff : numpy.ndarray
        The water balance for each section of the network, as the difference between
        the upstream and downstream flow rates (in m^3/s). Has shape `(N, 1)`, where
        `N` is the number of sections in the network.
    gdf_network_map : geopandas.GeoDataFrame
        A GeoDataFrame representing the network with a single line for each section,
        colored according to the sign of the water balance.
    section_basins : geopandas.GeoDataFrame
    The Hydrological subbasin belonging to each section for which the water balance was been computed
    """

    # %% We do some small data manipulations
    ts_stats = aggregate_time_series(data_ts=data_ts, analyse_option=time_series_analysis_option)

    # synchrnonize our datasets
    gauge_data = gauge_data.loc[gauge_data.index.isin(ts_stats.columns), :]
    # reduce the datasets to all which have metadata
    ts_stats = ts_stats[gauge_data.index.to_list()]
    print(ts_stats.shape[1], 'gauges with valid meta data')

    # our gauge data has to converted to geodataframe
    # make a geodataframe out of the data
    geometry = [Point(xy) for xy in zip(gauge_data['easting'], gauge_data['northing'])] 

    gauge_data = gpd.GeoDataFrame(gauge_data, crs=network.crs, geometry=geometry)


    gauge_data = gauge_data[~gauge_data.geometry.is_empty]

    # %% run the main functions

    gauges_connection_dict = generate_upstream_network(gauge_meta=gauge_data,
                                                       network_connections=network_connections)

    sections_meta, q_diff = calculate_network_balance(ts_data=ts_stats,
                                                      network_dict=gauges_connection_dict,
                                                      confidence_acceptance_level=confidence_acceptance_level)

    gdf_network_map = map_network_sections(network_dict=gauges_connection_dict,
                                           gauge_meta=gauge_data,
                                           network=network)

    # we want a function to calculate the network subbasin

    section_basins = get_section_basins(basins=basins,
                                        network_dict=gauges_connection_dict,
                                        basin_id_col=basin_id_col)

    # return the results
    return sections_meta, q_diff, gdf_network_map, section_basins
