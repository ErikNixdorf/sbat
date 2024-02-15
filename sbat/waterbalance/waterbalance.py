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
from typing import Dict, Any, Union, Tuple, Optional, List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString, MultiPolygon
from shapely.ops import nearest_points, unary_union

waterbalance_logger = logging.getLogger('sbat.waterbalance')


class Bayesian_Updating:
    def __init__(self, 
                 gauges_meta:pd.DataFrame(),
                 ts_data: pd.DataFrame(),
                 bayes_options: dict, 
                 output: bool = True,
                 ):
        """
        Initialize the Bayesian Updating class.

        Parameters
        ----------
        gauges_meta : pd.DataFrame
            DataFrame containing metadata for gauges.
        ts_data : pd.DataFrame
            DataFrame containing time series data.
        bayes_options : dict
            Dictionary containing Bayesian updating options.
        output : bool, optional
            Flag to indicate whether to output the results. Default is True.

        Raises
        ------
        ValueError
            If the prior Gaussian parameters type is not recognized.

        Returns
        -------
        None
        """
        # add the input data to class attributes
        self.gauges_meta = gauges_meta
        self.ts_data = ts_data
        self.bayes_options = bayes_options
        
        # add the specific prior information to the gauge data
        prior_type = self.bayes_options['prior_gaussian_parameters']['type'].lower()
        if prior_type == 'constant':
            #add constant information to each gauge
            self.gauges_meta['prior_q_diff_mean[m2/s]'] = self.bayes_options['prior_gaussian_parameters']['mean']
            self.gauges_meta['prior_q_diff_std[m2/s]'] = self.bayes_options['prior_gaussian_parameters']['standard_deviation']
        elif prior_type == 'gauge_dependent':
                   print('Predefined data from the Gauge Metadataset is used')
        else:
            raise ValueError('Invalid prior Gaussian parameters type. Should be either constant or gauge_dependent')
    
    def add_uncertainty(self):
        """
        Add uncertainty to the time series data based on specified gauge uncertainty options.
        
        This function computes and adds uncertainty to the time series data based on the specified gauge uncertainty options
        in the `bayes_options` dictionary.
        
        Raises:
            ValueError: If the gauge uncertainty type is not either constant or discharge_dependent.
        """
        # Copy the time series data and melt it
        q_uncertain_ts = self.ts_data.copy().reset_index().melt(id_vars=['date'])
    
        #add the measurement uncertainty
        q_uncertain_ts['measurement_uncertainty'] = self.bayes_options['gauge_uncertainty']['measurement_uncertainty']
        
        # Add rating curve uncertainty based on the specified type
        gauge_uncertainty_type = self.bayes_options['gauge_uncertainty']['type'].lower()
    
        if gauge_uncertainty_type == 'constant':

            q_uncertain_ts['rating_curve_uncertainty'] = self.bayes_options['gauge_uncertainty']['rating_curve_uncertainty']
        elif gauge_uncertainty_type == 'discharge_dependent':
            # Merge mean discharge (MQ) to the data
            q_uncertain_ts=pd.merge(q_uncertain_ts, self.gauges_meta[['gauge', 'MQ']], on='gauge', how='left')
            
            #depending on the difference between mean discharge and actual discharge, the uncertainty is provided
            #method see LAWA Pegelhandbuch 2018, page B-60, we assume that std = np.sqrt(relative_maximum_deviation)
            q_uncertain_ts.loc[q_uncertain_ts['value']<=0.5 * q_uncertain_ts['MQ'],'rating_curve_uncertainty'] = np.sqrt(0.2)
            q_uncertain_ts.loc[q_uncertain_ts['value']>=2 * q_uncertain_ts['MQ'],'rating_curve_uncertainty'] = np.sqrt(0.1)
            #for the values in between which are still nan
            q_uncertain_ts.loc[q_uncertain_ts['rating_curve_uncertainty'].isna(),'rating_curve_uncertainty'] = np.sqrt(0.05)
            
        else:
            raise ValueError('Gauge uncertainty type can be either constant or discharge_dependent')
        
        #add global uncertainty by simply merging --> reference is www.hydrol-earth-syst-sci.net/13/913/2009/ 
        q_uncertain_ts['Q_error_std'] = q_uncertain_ts['measurement_uncertainty'] + q_uncertain_ts['rating_curve_uncertainty']
        
        # clean data
        self.q_uncertain_ts = q_uncertain_ts[['date', 'gauge', 'value', 'Q_error_std']]
        self.q_uncertain_ts.rename(columns={'value':'Q'},inplace=True)
        
        
    def generate_uncertainty_samples(self):
        """
        Function generates samples based on gaussian distribution


        Returns
        -------
        None.

        """
        no_of_samples= self.bayes_options['number_of_samples']
        
        # Duplicate the uncertain time series to create multiple samples
        self.q_ts_samples = pd.concat([self.q_uncertain_ts] * no_of_samples, ignore_index=True)
    
        # Generate sample IDs
        self.q_ts_samples['sample_id'] = [i for i in range(no_of_samples) for _ in range(len(self.q_uncertain_ts))]
        
        # Draw random numbers from the distribution based on the standard deviation
        self.q_ts_samples['sample_error']  = np.random.normal(0, self.q_ts_samples['Q_error_std'])
        
        # Compute the discharge with error
        self.q_ts_samples['Q*'] = self.q_ts_samples['Q'] + self.q_ts_samples['sample_error'] * self.q_ts_samples['Q']
        
        #finally we map the data on the prior information on the dataset
        self.q_ts_samples=pd.merge(self.q_ts_samples, self.gauges_meta[[ 'prior_q_diff_mean[m2/s]','prior_q_diff_std[m2/s]']], on='gauge', how='left')
        
        return self.q_ts_samples

def generate_upstream_networks(
    gauge_meta: pd.DataFrame, 
    network_connections: pd.DataFrame
    ) -> dict:
    """
    Generate the upstream network of gauges, tributaries, and distributaries.

    Args:
        gauge_meta (pd.DataFrame): A DataFrame containing metadata for each gauge
            including the stream name, distance to mouth, etc.
        network_connections (pd.DataFrame): A DataFrame containing the network
            connections between streams, including the type of connection (tributary
            or distributary), the distance from the junction to the receiving water mouth,
            and the main stream name.

    Returns:
        dict: A dictionary containing gauge metadata and a list of tributaries and
            distributaries for each gauge, where the key is the gauge id (a hexadecimal
            string).
    """
    
    #get the functions within this function
    def generate_gauge_up_network(
        gauge: pd.Series, gauge_meta: pd.DataFrame, 
        network_connections: pd.DataFrame,
        gauge_column: str = 'index',
        ) -> dict:
        """
        Generate the upstream network of a gauge.

        Args:
            gauge_meta (pd.DataFrame): A DataFrame containing metadata for each gauge
                including the stream name, distance to mouth, etc.
            network_connections (pd.DataFrame): A DataFrame containing the network
                connections between streams, including the type of connection (tributary
                or distributary), the distance from the junction to the receiving water mouth,
                and the main stream name.

        Returns:
            dict: A dictionary containing gauge metadata and a list of tributaries and
                distributaries for each gauge, where the key is the gauge id (a hexadecimal
                string).
        """

        section=pd.DataFrame(index=[0])
        #we define our dictionary
        gauge_connection_dict = dict()
        gauge_connection_dict["id"] = secrets.token_hex(4)
        gauge_connection_dict["reach_name"] = gauge["stream"]
        
        #first we try to find the correct stream section of the stream where the gauge is
        section.loc[0,'stream']=gauge.stream
        section.loc[0,'distance_to_mouth_dp']=gauge['distance_to_mouth']
        #default 
        section.loc[0,'upstream_gauge'] = np.nan
        #analyse each section
        #while ~section.empty:
            #check for upstream part        
        stream_gauges = gauge_meta[gauge_meta["stream"] == gauge["stream"]].copy()
        stream_gauges["upstream_distance"] = (
                stream_gauges["distance_to_mouth"] - gauge["distance_to_mouth"]
        )
        stream_gauges = stream_gauges[stream_gauges["upstream_distance"] > 0]
        if not stream_gauges.empty:
            upstream_gauge = pd.DataFrame(stream_gauges.loc[
                [stream_gauges.upstream_distance.idxmin()]
            ]).set_index(gauge_column)
            section['upstream_gauge'] = stream_gauges.loc[stream_gauges.upstream_distance.idxmin(),gauge_column]
            section['distance_to_mouth_up']=stream_gauges.loc[stream_gauges.upstream_distance.idxmin(),'distance_to_mouth']
        else:
            section['distance_to_mouth_up']=np.inf
            upstream_gauge = pd.DataFrame()
        
        #get all branches within the range
        branch_network=get_branches(network_connections,
                                      section,
                                      dist_col_network = 'distance_junction_from_receiving_water_mouth',
                                      dist_col_gauge_up = 'distance_to_mouth_up',
                                      dist_col_gauge_dp = 'distance_to_mouth_dp',
                                      )
        #we go trough the entire branch network and check for gauges    

        ls_subbranches_network = extract_subbranches_network(branch_network,gauge_meta,network_connections)
        

        #label the upstream gauge 
        if not isinstance(section.loc[:,'upstream_gauge'].values[0],str):
            if 'distributary' in branch_network['type'].tolist():
                section['upstream_gauge'] ='river_junction'
            else:
                section['upstream_gauge'] = 'river_spring'
        
        # we finally label the missing information of the dictionary
        gauge_connection_dict["upstream_point"] = section['upstream_gauge'].iloc[0]
        gauge_connection_dict['downstream_point'] = gauge.name
        gauge_connection_dict["gauge_up"] = upstream_gauge
        
        #the network
        if len(ls_subbranches_network)> 0:
            df_subbranches_network=pd.concat(ls_subbranches_network,axis=0)
            df_subbranches_network=df_subbranches_network[['stream','main_stream', 'type',
                   'distance_junction_from_receiving_water_mouth', 'upstream_point',
                   'downstream_point']]
            
            gauge_connection_dict["tributaries_up"] = df_subbranches_network[df_subbranches_network['type']=='tributary']
            gauge_connection_dict["distributaries_up"] = df_subbranches_network[df_subbranches_network['type']=='distributary']
        else:
            gauge_connection_dict["tributaries_up"] = pd.DataFrame()
            gauge_connection_dict["distributaries_up"] = pd.DataFrame()
        
        return gauge_connection_dict
    
    
    def get_branches(network_connections: pd.DataFrame, gauge_meta: pd.DataFrame,
                       dist_col_network: str = 'distance_junction_from_receiving_water_mouth',
                       dist_col_gauge_up: str = 'distance_to_mouth_up',
                       dist_col_gauge_dp: str = 'distance_to_mouth_dp') -> pd.DataFrame:
        """
        Extracts sub-branches from a river network based on connections and gauge information.
        
        Args:
            network_connections (pd.DataFrame): A dataframe containing information about the connections between branches in the network.
            gauge_meta (pd.DataFrame): A dataframe containing information about gauges located in the network.
            dist_col_network (str): The name of the column in `network_connections` that specifies the distance of a branch from the receiving water mouth.
            dist_col_gauge_up (str): The name of the column in `gauge_meta` that specifies the distance of a gauge from the mouth of the upstream branch.
            dist_col_gauge_dp (str): The name of the column in `gauge_meta` that specifies the distance of a gauge from the mouth of the downstream branch.
        
        Returns:
            pd.DataFrame: A dataframe containing information about the sub-branches of the network that meet the distance criteria, based on the provided gauge information.
        """
        branch_network = network_connections[network_connections['main_stream'].isin(gauge_meta['stream'].tolist())]
        branch_network = branch_network[branch_network[dist_col_network]<gauge_meta[dist_col_gauge_up].values[0]]
        branch_network = branch_network[branch_network[dist_col_network]>gauge_meta[dist_col_gauge_dp].values[0]]

        return branch_network
    

    def extract_subbranches_network(branch_network: pd.DataFrame,
        gauge_meta: pd.DataFrame,
        network_connections: Dict,
        ) -> List[pd.DataFrame]:
        
        """
        Extracts subbranches from a river network and attaches gauge information to each branch.

        Args:
            branch_network (pd.DataFrame): A pandas dataframe containing information about the branches in the network. The dataframe should have the following columns:
                - 'name': Name of the branch (str)
                - 'stream': ID of the stream that the branch belongs to (str)
                - 'type': Type of the branch, either 'tributary' or 'distributary' (str)
                - 'upstream_junction': ID of the upstream junction (str)
                - 'downstream_junction': ID of the downstream junction (str)
                - 'distance_junction_from_receiving_water_mouth': Distance between the downstream junction and the receiving water mouth (float)
            gauge_meta (pd.DataFrame): A pandas dataframe containing information about gauges located in the network. The dataframe should have the following columns:
                - 'gauge': ID of the gauge (str)
                - 'stream': ID of the stream that the gauge is located on (str)
                - 'distance_to_mouth': Distance from the gauge to the receiving water mouth (float)
            network_connections (Dict): A dictionary containing information about the connections between branches in the network. The dictionary should have the following keys:
                - 'upstream': A pandas dataframe containing information about the upstream connections of each branch. The dataframe should have the following columns:
                    - 'name': Name of the branch (str)
                    - 'stream': ID of the stream that the branch belongs to (str)
                    - 'downstream_junction': ID of the downstream junction (str)
                - 'downstream': A pandas dataframe containing information about the downstream connections of each branch. The dataframe should have the following columns:
                    - 'name': Name of the branch (str)
                    - 'stream': ID of the stream that the branch belongs to (str)
                    - 'upstream_junction': ID of the upstream junction (str)

        Returns:
            List[pd.DataFrame]: A list of dataframes containing information about the subbranches of the network, with gauge information attached. Each dataframe in the list should have the following columns:
                - 'name': Name of the subbranch (str)
                - 'stream': ID of the stream that the subbranch belongs to (str)
                - 'type': Type of the subbranch, either 'tributary' or 'distributary' (str)
                - 'upstream_point': ID of the upstream point (either a gauge ID or a junction ID) (str)
                - 'downstream_point': ID of the downstream point (either a gauge ID or a junction ID) (str)
                - 'distance_to_mouth_up': Distance from the upstream point to the receiving water mouth (float)
                - 'distance_to_mouth_dp': Distance from the downstream point to the receiving water mouth (float)
        """
        #definitions
        gauge_branch_list=list()
        sorting_order={'tributary':True,
                     'distributary':False}
        
        branch_type_info = {'tributary': ('river_spring', 'river_junction', np.inf, 0),
                            'distributary': ('river_junction', 'river_mouth', 0, np.inf)}
            
        while not branch_network.empty:
            
            branches =branch_network.copy()
            for i,branch in branches.iterrows():
                #check whether there is a gauge
                branch_gauges = gauge_meta[gauge_meta["stream"] == branch["stream"]]
                branch_gauges= branch_gauges.sort_values('distance_to_mouth',
                                                        ascending = sorting_order[branch['type']
                                                                                  ])            
                # set upstream and downstream point labels and distances based on branch type
                upstream_point, downstream_point, distance_to_mouth_up, distance_to_mouth_dp = branch_type_info[branch['type']]
                
                if not branch_gauges.empty:
                    gauge_info = branch_gauges.iloc[0]
                    if branch['type'] == 'tributary':
                        upstream_point, distance_to_mouth_up = gauge_info['gauge'], gauge_info['distance_to_mouth']
                    else:
                        downstream_point, distance_to_mouth_dp = gauge_info['gauge'], gauge_info['distance_to_mouth']
                
                # set values in branch dataframe
                branch.at['upstream_point'] = upstream_point
                branch.at['downstream_point'] = downstream_point
                branch.at['distance_to_mouth_up'] = distance_to_mouth_up
                branch.at['distance_to_mouth_dp'] = distance_to_mouth_dp
                #append branch to list
                gauge_branch_list.append(branch.to_frame().T)        
                #we remove the branch from the iterations
                branch_network=branch_network.drop(index=branch.name)
                #get new subbranches and add them to the network

                subbranches=get_branches(network_connections,
                                              branch.to_frame().T,
                                              dist_col_network = 'distance_junction_from_receiving_water_mouth',
                                              dist_col_gauge_up = 'distance_to_mouth_up',
                                              dist_col_gauge_dp = 'distance_to_mouth_dp',
                                              )
                if not subbranches.empty:
                    #we add them to the network_data
                    branch_network = pd.concat([branch_network, subbranches])
                        
        return gauge_branch_list
    
    
    #%% start main function
    gauges_connection_dict = dict()
    for i, gauge in gauge_meta.iterrows():
        gauge_connection_dict = generate_gauge_up_network(gauge,gauge_meta.reset_index(),network_connections,
                                                          gauge_column = gauge_meta.index.name)
        gauges_connection_dict.update({gauge.name: gauge_connection_dict})
        
    return gauges_connection_dict
    
    


def map_time_dependent_cols_to_gdf(
        geodf: gpd.GeoDataFrame,
        time_dep_df: pd.DataFrame,
        geodf_index_col: str = 'downstream_point',
        time_dep_df_index_col: str = 'gauge',
        time_dep_df_time_col: str = 'decade',
        nan_value: Union[int, float] = -99999,
        remove_nan_rows: bool = True,
        ) -> gpd.GeoDataFrame:
    """
    Map time-dependent columns from a dataframe to a geodataframe based on shared index columns.

    Parameters
    ----------
    geodf : geopandas.GeoDataFrame
        The geodataframe to be expanded with time-dependent columns.
    time_dep_df : pandas.DataFrame
        The dataframe with time-dependent columns to be mapped onto the geodataframe.
    geodf_index_col : str, optional
        The name of the index column in the geodataframe. Default is 'downstream_point'.
    time_dep_df_index_col : str, optional
        The name of the index column in the time-dependent dataframe. Default is 'gauge'.
    time_dep_df_time_col : str, optional
        The name of the time-dependent column in the time-dependent dataframe. Default is 'decade'.
    nan_value : Union[int, float], optional
            The value to replace NaN values in the output geodataframe. Default is -99999.


    Returns
    -------
    geopandas.GeoDataFrame
        The expanded geodataframe with time-dependent columns mapped onto it.
    """
    
    #very first we check whether there is no geometry for a gauge
    no_geo_gauges = list(set(time_dep_df.reset_index()[time_dep_df_index_col].tolist())-set(geodf[geodf_index_col]))
    for no_geo_gauge in no_geo_gauges:
        waterbalance_logger.warning(f'No geometry for gauge {no_geo_gauge} provided')
    
    #after warning remove from dataset
    time_dep_df=time_dep_df.loc[~time_dep_df.reset_index()[time_dep_df_index_col].isin(no_geo_gauges).values,:]
    
    
    
    #check which row of geodataframe has to be expanded how often and for which decades
    copy_dict=dict()
    for gauge,df_gauge in time_dep_df.groupby(time_dep_df_index_col):
        copy_dict.update({gauge:list(df_gauge.reset_index()[time_dep_df_time_col].unique())})
        
    #copy_dict=time_dep_df.groupby(time_dep_df_index_col).size().to_dict()
    #set_index of the geodataframe
    geodf=geodf.set_index(geodf_index_col)
    

    # create expanded gdf 
    list_expanded_df=list()
    #loop trough original gdf in order to expand the rows
    for _,row in geodf.iterrows():
        #extend the row to a dataframe
        df_row_extend = pd.DataFrame([row]*len(copy_dict[row.name]))
        #add the decaded
        df_row_extend[time_dep_df_time_col] = copy_dict[row.name]
        
        list_expanded_df.extend([df_row_extend])
        
    
    #merge the expanded geometrical data
    expanded_df = pd.concat(list_expanded_df)
    expanded_df.index.name=time_dep_df_index_col
    #we make a multi_index in order to merge it properly with the time series
    expanded_df=expanded_df.reset_index().set_index([time_dep_df_index_col,time_dep_df_time_col])
    
    #merge with the time dependend statistics
    df_merged = pd.concat([expanded_df,
                           time_dep_df,
                           ],
                          axis=1
                          )
    
    #reset the index in order to save it as a geodataframe
    df_merged = df_merged.reset_index()
    
    # we clean the data
    df_merged = df_merged.loc[~df_merged[time_dep_df_time_col].isna(),:]
    
    #replace NaN Values
    df_merged=df_merged.replace(np.nan,nan_value)
    
    #generate the output geodataframe
    geodf_out = gpd.GeoDataFrame(data=df_merged,
                               geometry=df_merged['geometry'],
                               crs=geodf.crs)
    if remove_nan_rows:
        #remove geodata with no proper water balance
        waterbalance_logger.info(f'remove values where balance is{nan_value}')
        geodf_out = geodf_out[geodf_out['balance[m³/s]']!=nan_value]
    return geodf_out

def multilinestring_to_singlelinestring(
        multilinestring: MultiLineString,
        start_point: Point,
        search_radius: float = 2000,
        min_distance: float = 200,
) :
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

    # one annoything is that the Neisse River is not a nicely sorted linestring, so we need to resort all vertices,
    # from downstream to upstream
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
            waterbalance_logger.info(f"Points too close ({pp_distance:.2f}m), removing point")
            unsorted_point_bag = unsorted_point_bag.difference(closest_point_updated)
            continue

        # remove the closest point from unsorted point bag and add to sorted point bag
        if pp_distance < search_radius:
            unsorted_point_bag = unsorted_point_bag.difference(closest_point_updated)
            sorted_point_bag.append(closest_point_updated)
            closest_point = deepcopy(closest_point_updated)
        else:
            waterbalance_logger.info(f"Closest point too far ({pp_distance:.2f}m) from existing line, probably a branch")
            waterbalance_logger.debug(f"Loop stopped at iteration {iteration}, try to increase resolution to avoid")
            break
    return sorted_point_bag


# %% Next function how to calculate the balance
def calculate_network_balance(
        ts_data: pd.DataFrame = pd.DataFrame(),
        network_dict: Dict[str, Dict[str, Tuple[str, str]]] = dict(),
        get_decadal_stats: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the water balance for a network of gauges using time series data and 
    a dictionary representing the network topology.
    
    Parameters:
    -----------
    ts_data : pd.DataFrame(), optional
        A DataFrame containing time series data for each gauge. Default is an empty DataFrame.
        
    network_dict : dict, optional
        A dictionary representing the topology of the network, with each key representing a gauge and the corresponding value being another dictionary containing information about that gauge's upstream and downstream connections. Default is an empty dictionary.
  
    get_decadal_stats: boolean, optional
        decides whether decadal stats will be calculated or not
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
        ts_distributaries = pd.Series(np.zeros(nr_ts), index=ts_data.index)
        ts_tributaries = ts_distributaries.copy()
        ts_data_gauge_up = ts_distributaries.copy()

        logging.info(f'add water balance to gauge {gauge}')

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
                distri_gauges = distributaries_up[distributaries_up.downstream_point != 'river_mouth']
                if not distri_gauges.empty:
                    ts_distributaries = ts_data[distri_gauges.downstream_point.tolist()].sum(axis=1)
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

            with np.errstate(invalid='ignore'):
                df_section['balance_confidence'] = np.divide(balance.values, (ts_data[gauge] + ts_data_gauge_up).values)

        # add index data
        df_section['Date'] = ts_data.index
        logging.info(f'Water balance added to gauge {gauge}')

        sections_meta.append(df_section)

    # Finally we get out the data

    sections_meta = pd.concat(sections_meta)

    q_diff = sections_meta.drop_duplicates().pivot(index='Date', columns='downstream_point', values='balance')

    if get_decadal_stats:
        sections_meta['decade']=sections_meta['Date'].apply(lambda x: x[:3]+'5')
    else:
        sections_meta['decade']=-9999

    return sections_meta, q_diff


def map_network_sections(
        network_dict: Dict,
        gauge_meta: pd.DataFrame,
        stream_network: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    network_dict : dict
        A dictionary containing the gauge sections.
    gauge_meta : pd.DataFrame
        A DataFrame containing the metadata for the gauges.
    network : gpd.GeoDataFrame
        A GeoDataFrame containing the river network.
        The order of vertex is assumed that it is counted with lowest numbers being downstream

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
        if gauge['reach_name'] not in stream_network.reach_name.tolist():
            logging.info(f'{gauge["reach_name"]} not in network data, check correct names')
            continue
        river_line = stream_network[stream_network.reach_name == gauge['reach_name']].geometry.iloc[0]
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
            logging.info('Stream Line needs at least two points, we move by one point, but you should check geometry')
            river_pnts = river_pnts.iloc[pnt_id_upstream:pnt_id_downstream + 2]


        section_line = LineString(river_pnts.geometry)
                
        #get the lines of the tributaries
        trib_lines = list()

        for branch_name in ['tributaries_up', 'distributaries_up']:

            # we loop trough the dataset        
            for (_, branch) in gauge[branch_name].iterrows():
                # extract the river line if available        
                if branch['stream'] not in stream_network.reach_name.tolist():
                    logging.info(f'{branch["stream"]} not in network data, check correct names')
                    continue
                river_line = stream_network[stream_network.reach_name == branch['stream'
                ]].geometry.iloc[0]

                # first we check whether there is really data in the dataset        
                if len(gauge[branch_name]) > 0:
                    # we get the river line        
                    if branch['stream'] not in stream_network.reach_name.tolist():
                        logging.info(f'{branch["stream"]} not in network data, check correct names')
                        continue

                    # extract the river line        
                    river_line = stream_network[stream_network.reach_name == branch['stream'
                    ]].geometry.iloc[0]

                    # get all river points as geodataframe        
                    river_pnts = gpd.GeoDataFrame(geometry=[Point(pt) for pt in
                                                            river_line.coords])
                    
                    #%% identify the point_ids of the branches
                    if branch_name == 'tributaries_up':
                        #downstream is always 0, because it is the river_mouth
                        pnt_id_downstream = 0                        
                        #upstream depends whether there is a gauge or not
                        if branch.upstream_point != 'river_spring':
                            pnt_branch_gauge = \
                                gauge_meta.loc[gauge_meta.index == branch['upstream_point'], 'geometry'].iloc[0]
                            pnt_id_upstream = river_pnts.distance(pnt_branch_gauge).idxmin()
                        else:
                            pnt_id_upstream = len(river_pnts) - 1
                    
                    elif branch_name == 'distributaries_up':
                        #upstream is always the last point in the branch
                        pnt_id_upstream = len(river_pnts) - 1
                        
                        #the downstream point depends whether there is a gauge or not
                        if branch.downstream_point != 'river_mouth':
                            pnt_branch_gauge = \
                                gauge_meta.loc[gauge_meta.index == branch['downstream_point'], 'geometry'].iloc[0]
                            pnt_id_downstream = river_pnts.distance(pnt_branch_gauge).idxmin()
                        else:
                            #the first point
                            pnt_id_downstream = 0

                    # we rearrange if it does make sense with the flow direction
                    if pnt_id_upstream > pnt_id_downstream:
                        river_pnts_extracted = river_pnts.iloc[pnt_id_downstream:pnt_id_upstream + 1]
                    elif pnt_id_upstream < pnt_id_downstream:
                        river_pnts_extracted = river_pnts.iloc[pnt_id_upstream:pnt_id_downstream + 1]
                    else:
                        logging.info(
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
                                       crs=stream_network.crs)

        gdf_balances = pd.concat([gdf_balances, gdf_balance])
    
    # finally we calculate the length of the section and add the information to the gauges as well
    gdf_balances['waterway_length'] = gdf_balances.geometry.length
    
    gauge_meta['waterway_length'] = gdf_balances.set_index('downstream_point')['waterway_length']

    return gdf_balances.set_crs(stream_network.crs), gauge_meta


# add a function for time series manipulation
def aggregate_time_series(data_ts: pd.DataFrame, 
                          analyse_option: Optional[str] = 'overall_mean',
                          ) -> pd.DataFrame:
    """
    Aggregates time series data based on the selected option
    
    Parameters:
    -----------
    data_ts : pd.DataFrame()
        A DataFrame containing time series data for a given gauge
        
    analyse_option : str, optional
        A string representing the desired data aggregation option. Default is 'overall_mean'
        Available options:
            - 'overall_mean': Takes entire time series
            - 'annual_mean': Resamples data to annual mean
            - 'summer_mean': Resamples data to summer mean (June to September)
            - 'daily': Uses daily statistics
    
    Returns:
    --------
    ts_stats : pd.DataFrame()
        A DataFrame containing the aggregated time series data
    """    
    if analyse_option is None:
        logging.info('No data aggregation option select, continue with original time series')
        return data_ts

    # just for testing we take the mean
    if analyse_option == 'overall_mean':
        logging.info(f'{analyse_option} takes entire time series')
        ts_stats = data_ts.mean()
        stats_name = 'mean_discharge_m_s'
        ts_stats = ts_stats.rename(stats_name).to_frame().T

    elif analyse_option == 'annual_mean':
        ts_stats = data_ts.resample('Y').mean()

    elif analyse_option == 'summer_mean':
        logging.info('Calculating summer mean (June to September)')
        ts_stats = data_ts.loc[data_ts.index.month.isin([6, 7, 8, 9])].resample('Y').mean()

    # daily calculations
    elif analyse_option == 'daily':
        logging.info('Calculating daily statistics')
        ts_stats = data_ts.copy()
    else:
        logging.info('Invalid aggregation option selected, continuing with original time series')
        return data_ts
    
    if analyse_option != 'overall_mean':
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
                section_basin.geometry = [max(section_basin.iloc[0].geometry.geoms, key=lambda a: a.area)]

            # get the next distributary gauge and add their basin
            if not distributaries_up.empty:
                distri_gauges = distributaries_up[distributaries_up.downstream_point != 'river_mouth']
                for distri_gauge in distri_gauges.downstream_point:
                    section_basin.loc[section_basin.index[0], 'geometry'] = unary_union(
                        [section_basin.iloc[0].geometry, basins.loc[basins['basin'] == distri_gauge].iloc[0].geometry])

                    # we dissolve the geometries
                    section_basin = section_basin.dissolve()

            # append
            section_basins.append(section_basin)

    # merge
    section_basins = pd.concat(section_basins)
    # overwrite some layers
    # assuming a representative Circle, we assume the radius of this circle is the mean length towards the stream
    section_basins['basin_area'] = section_basins.geometry.area
    section_basins['L_represent'] = np.sqrt(section_basins['basin_area'] / np.pi)

    return section_basins


# %% A function which connects all functions
def get_section_waterbalance(gauge_data: pd.DataFrame = pd.DataFrame(),
                              data_ts: pd.DataFrame = pd.DataFrame(),
                              stream_network: gpd.GeoDataFrame = gpd.GeoDataFrame(),
                              basins: gpd.GeoDataFrame = gpd.GeoDataFrame(),
                              network_connections: pd.DataFrame = pd.DataFrame(),
                              confidence_acceptance_level: float = 0.05,
                              time_series_analysis_option: str = 'overall_mean',
                              basin_id_col: str = 'basin',
                              decadal_stats: bool =True,
                              bayesian_options= dict()
                              ) -> Tuple:
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
    stream_network : geopandas.GeoDataFrame, optional
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
        balance calculations. The default is 0.05. NOT IMPLEMENTED YET
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

    # first time is aggregated
    data_ts = aggregate_time_series(data_ts=data_ts, analyse_option=time_series_analysis_option)
    
    # synchrnonize our datasets
    gauge_data = gauge_data.loc[gauge_data.index.isin(data_ts.columns), :]
    # reduce the datasets to all which have metadata
    data_ts = data_ts[gauge_data.index.unique().to_list()]
    logging.info(f'{data_ts.shape[1]} gauges with valid meta data')
    
    #
    # make a geodataframe out of the data
    geometry = [Point(xy) for xy in zip(gauge_data['easting'], gauge_data['northing'])]

    gauge_data = gpd.GeoDataFrame(gauge_data, crs=stream_network.crs, geometry=geometry)

    gauge_data = gauge_data[~gauge_data.geometry.is_empty]
    
    # Generate upstream networks and map network sections
    gauges_connection_dict = generate_upstream_networks(gauge_meta=gauge_data,
                                                       network_connections=network_connections)
    gdf_network_map, gauge_data = map_network_sections(network_dict=gauges_connection_dict,
                                           gauge_meta=gauge_data,
                                           stream_network=stream_network)


    #%% Apply Bayesian Updating if requestest
    if bayesian_options['activate']:
        logging.info('Initialize Discharge Samples with Uncertainty') 
        if time_series_analysis_option.lower() in ['overall_mean','annual_mean','summer_mean']:
            logging.warning('Aggregated Time Series cant be used for Bayesian Updating')
        else:
            # Generate the Bayesian Updater class
            bayesian_Updater = Bayesian_Updating(gauge_data, data_ts, bayesian_options)
            bayesian_Updater.add_uncertainty()
            data_ts_samples =bayesian_Updater.generate_uncertainty_samples()
            
            # Pivot data for analysis
            discharge_samples = data_ts_samples.pivot(columns='gauge',index=['date','sample_id'],values='Q*')
            
            # Calculate network balance based on discharge samples
            sections_meta, q_diff = calculate_network_balance(ts_data=discharge_samples,
                                                              network_dict=gauges_connection_dict,
                                                              get_decadal_stats = decadal_stats)
            
            #repair the multiindex
            q_diff.index=discharge_samples.index
            
            #same for sections_meta
            sections_meta[['date', 'sample_id']] = pd.DataFrame(sections_meta['Date'].tolist(), index=sections_meta.index)
            sections_meta=sections_meta.drop(columns='Date')
            #unpivot q_diff
            q_diff = q_diff.melt(ignore_index=False).rename(columns={'value':'q_diff'}).reset_index()
            
            #merge with data from ts_samples, both need a unique index
            data_ts_samples = data_ts_samples.rename(columns={'gauge':'downstream_point'}).set_index(['date','sample_id','downstream_point'])
            q_diff=q_diff.set_index(['date','sample_id','downstream_point'])
            q_diff = pd.concat([q_diff,data_ts_samples],axis=1)
    else:
        # Run the water balance analysis without Bayesian Updating
        sections_meta, q_diff = calculate_network_balance(ts_data=data_ts,
                                                          network_dict=gauges_connection_dict,
                                                          get_decadal_stats = decadal_stats)
        
        # Prepare similar output to the Bayesian case for consistency
        q_diff=q_diff.melt(ignore_index=False).rename(columns={'value':'q_diff'}).reset_index()
        q_diff['sample_id'] = 0
        sections_meta=sections_meta.rename(columns={'Date':'date'})
        sections_meta['sample_id'] = 0



    # we want a function to calculate the network subbasin area
    if basins is None:
        section_basins=None
    else:
        section_basins = get_section_basins(basins=basins,
                                            network_dict=gauges_connection_dict,
                                            basin_id_col=basin_id_col)
    
    #in any case we calculate the balance per length of the section
    gauge_index=gauge_data.reset_index().rename(columns={'gauge':'downstream_point'})[['downstream_point','waterway_length']]
    #merge with section meta and q_diff to get the values per section length
    q_diff = pd.merge(q_diff.reset_index(), gauge_index, on='downstream_point', suffixes=('_df1', '_df2'))
    q_diff['q_diff[m2/s]']=q_diff['q_diff']/q_diff['waterway_length']
    q_diff=q_diff.rename(columns={'q_diff':'Q_diff[m³/s]'})
    
    #now for section meta
    sections_meta = pd.merge(sections_meta, gauge_index, on='downstream_point', suffixes=('_df1', '_df2'))
    sections_meta['balance[m2/s]']=sections_meta['balance']/sections_meta['waterway_length']
    sections_meta=sections_meta.rename(columns={'balance':'balance[m³/s]'})
    
    
    
    
    # return the results
    return sections_meta, q_diff, gdf_network_map, section_basins,data_ts
