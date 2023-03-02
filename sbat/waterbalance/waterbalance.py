"""
This script analysis the water balance between river sections
A section is either the branch between two gauges or betweeen a gauge and river_mouth and vise versa

#next tasks
confidence_interval
hash_

"""
import pandas as pd
import os
import numpy as np
import geopandas as gpd
from copy import deepcopy
from shapely.geometry import Point, MultiPoint, LineString,MultiLineString
from shapely.ops import nearest_points,cascaded_union
import secrets
import logging
from typing import Dict
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
    #start logging
    logger = logging.getLogger(__name__)
    #one annoything is that the Neisse River is not a nicely sorted linestring, so we need to resort all vertices, from downstream to upstream
    closest_point=deepcopy(start_point)
    sorted_point_bag=[start_point]
    #first remove the start point from the bag
    unsorted_point_bag=multilinestring.difference(closest_point)
    #start to loop
    for iteration in range(0,len(unsorted_point_bag)):
        #get the newest closest point
        closest_point_updated,_=nearest_points(unsorted_point_bag, closest_point)
        #calculate the distance between the two points
        pp_distance=closest_point.distance(closest_point_updated)
        
        #if distance is too close remove point and continue
        if pp_distance<min_distance:
            logger.info(f"Points too close ({pp_distance:.2f}m), removing point")           
            unsorted_point_bag=unsorted_point_bag.difference(closest_point_updated)
            continue
        
        
        #remove the closest point from unsorted point bag and add to sorted point bag
        if pp_distance<search_radius:            
            unsorted_point_bag=unsorted_point_bag.difference(closest_point_updated)
            sorted_point_bag.append(closest_point_updated)
            closest_point=deepcopy(closest_point_updated)
        else:
            logger.info(f"Closest point too far ({pp_distance:.2f}m) from existing line, probably a branch")
            logger.debug(f"Loop stopped at iteration {iteration}, try to increase resolution to avoid")
            break
    return sorted_point_bag         


#%% We loop trough the gauges in order to define the geometry
def generate_upstream_network(gauge_meta=pd.DataFrame(),tributary_connections=pd.DataFrame(),
                              distributary_connections=pd.DataFrame()):
    
    gauges_connection_dict=dict()
    
    
    #%% The Idea is we loop trough the gauges, find the upstream stream and calculate the section water balance
    
    
    for i,gauge in gauge_meta.iterrows():
        #first we create a hexadecimal string
        gauge_connection_dict=dict()
        
        #first we create a hex string which helps us 
        gauge_connection_dict['id']=secrets.token_hex(4)

        
        #we write the name of the gewaesser
        gauge_connection_dict['reach_name']=gauge.gewaesser
     
        #check whether there is an upstream gauge in the system
        stream_gauges=gauge_meta[gauge_meta.gewaesser==gauge.gewaesser]
        stream_gauges['upstream_distance']=stream_gauges.km_muendung_hauptfluss_model-gauge.km_muendung_hauptfluss_model
        stream_gauges=stream_gauges[stream_gauges['upstream_distance']>0]
        upstream_gauge=stream_gauges.copy()
        #tributaries upstream
        tributaries=tributary_connections.loc[tributary_connections.Vorfluter==gauge.gewaesser,:]
        tributaries['upstream_distance']=tributaries.km_zufluss_Vorfluter_ab_muendung_vorfluter-gauge.km_muendung_hauptfluss_model
        tributaries=tributaries[tributaries['upstream_distance']>0]
    
        #similar we look for distributaries upstream
        distributaries=distributary_connections.loc[distributary_connections.Hauptfluss==gauge.gewaesser,:]
        distributaries['upstream_distance']=distributaries.km_abfluss_hauptfluss_ab_muendung_hauptfluss-gauge.km_muendung_hauptfluss_model
        distributaries=distributaries[distributaries['upstream_distance']>0]        
        
        
        #if we have a stream gauge upstream we will reduce the tributaries and distributaries to the ones between both gauges
        if len(stream_gauges)>0:
            
            upstream_gauge=stream_gauges.loc[[stream_gauges.upstream_distance.idxmin()]]
            tributaries=tributaries.loc[(tributaries['upstream_distance']-upstream_gauge['upstream_distance'].values)<0,:]
            distributaries=distributaries.loc[(distributaries['upstream_distance']-upstream_gauge['upstream_distance'].values)<0,:]
            
            
        #connect    
        gauge_connection_dict['gauge_up']=pd.DataFrame(upstream_gauge)
        gauge_connection_dict['tributaries_up']=tributaries.copy()
        gauge_connection_dict['distributaries_up']=distributaries.copy()
        

        #we check for subdistributaries/subtributaries with gauges on a second level
        for branch_type in ['tributaries_up','distributaries_up']:
            for _,branch in gauge_connection_dict[branch_type].iterrows():
                #first we check whether there is a subtributary between closest gauge and the river mouth
                subtributaries=tributary_connections.loc[tributary_connections.Vorfluter==branch.Gewaesser,:]
                subdistributaries=distributary_connections.loc[distributary_connections.Hauptfluss==branch.Gewaesser,:]
                
    
                
                #calculate the most downstream gauge of the tributary
                branch_gauges=gauge_meta[gauge_meta.gewaesser==branch.Gewaesser]
                if len(branch_gauges) == 0:
                    print('No Gauge at tributary ',branch.Gewaesser)
                    continue
                #take the one closest to the river mouth
                if branch_type=='tributaries_up':
                    branch_gauge=branch_gauges.loc[branch_gauges.km_muendung_hauptfluss_model.idxmin(),:]
                elif branch_type=='distributaries_up':
                    branch_gauge=branch_gauges.loc[branch_gauges.km_muendung_hauptfluss_model.idxmax(),:]
                
                #calculate whether there is an inflow inbetween:
                if branch_type == 'tributaries_up':
                    subtributaries=subtributaries[(subtributaries.km_zufluss_Vorfluter_ab_muendung_vorfluter-branch_gauge.km_muendung_hauptfluss_model)<0]
                    subdistributaries=subdistributaries[(subdistributaries.km_abfluss_hauptfluss_ab_muendung_hauptfluss-branch_gauge.km_muendung_hauptfluss_model)<0]
                elif branch_type == 'distributaries_up':
                    subtributaries=subtributaries[(subtributaries.km_zufluss_Vorfluter_ab_muendung_vorfluter-branch_gauge.km_muendung_hauptfluss_model)>0]
                    subdistributaries=subdistributaries[(subdistributaries.km_abfluss_hauptfluss_ab_muendung_hauptfluss-branch_gauge.km_muendung_hauptfluss_model)>0]
     
                #append to data
                gauge_connection_dict[branch_type]=pd.concat([gauge_connection_dict[branch_type],subtributaries])
                gauge_connection_dict[branch_type]=pd.concat([gauge_connection_dict[branch_type],subdistributaries])
            
         
    
        
        #define the name of the upstream points
        if len(gauge_connection_dict['gauge_up'])==0:
            #we first define upstream and downstream_points
            if len(gauge_connection_dict['tributaries_up'])==0 and len(gauge_connection_dict['distributaries_up'])==0 and gauge.gewaesser in distributary_connections.Gewaesser.to_list():
                gauge_connection_dict['upstream_point']='river_junction'
            else:
                gauge_connection_dict['upstream_point']='river_spring'
        else:
            gauge_connection_dict['upstream_point']=gauge_connection_dict['gauge_up'].index.values[0]
            
        #downstream is always the same
        gauge_connection_dict['downstream_point']=gauge.name
        
    
    
            
        #check for tributary or distributary gauges between the spring and the gauge
        for trib_type in ['tributaries','distributaries']:
            #we generate a copy of gauge_meta
            gauge_meta_reset=gauge_meta.copy().reset_index()
            # if there are tributary_gauges
            if len(gauge_connection_dict[trib_type+'_up'])>0:
                
                #first we check whether the tributaries have gauges
                tribs_with_gauges=gauge_connection_dict[trib_type+'_up'][gauge_connection_dict[trib_type+'_up'].Gewaesser.isin(gauge_meta.gewaesser)].set_index('Gewaesser')
                
                if len(tribs_with_gauges)>0:
                    #get_the_tributary_gauges which is the most downstream_gauge
    
                    trib_gauges=gauge_meta_reset.loc[gauge_meta_reset.gewaesser.isin(tribs_with_gauges.index),:]
                    #we only select the ones which are most downstream
                    if trib_type=='tributaries':
                        trib_gauges=trib_gauges.loc[trib_gauges.groupby('gewaesser').km_muendung_hauptfluss_model.idxmin()].set_index('gewaesser')
                    elif 'distributaries':
                        trib_gauges=trib_gauges.loc[trib_gauges.groupby('gewaesser').km_muendung_hauptfluss_model.idxmax()].set_index('gewaesser')
                    tribs_with_gauges['upstream_point']=trib_gauges.gauge
                    tribs_with_gauges['downstream_point']='river_mouth'
                    tribs_with_gauges=tribs_with_gauges.reset_index()
                #we also add the tribs with no gauges 
                tribs_without_gauges=gauge_connection_dict[trib_type+'_up'][~gauge_connection_dict[trib_type+'_up'].Gewaesser.isin(gauge_meta_reset.gewaesser)]           
                if len(tribs_without_gauges)>0:
                    tribs_without_gauges['upstream_point']='river_spring'
                    tribs_without_gauges['downstream_point']='river_mouth'
                
                #merge
                tribs_merged=pd.concat([tribs_with_gauges,tribs_without_gauges],ignore_index=True)
    
                        
                gauge_connection_dict[trib_type+'_up']=tribs_merged
    
                    
        gauges_connection_dict.update({gauge.name:gauge_connection_dict})
    
    return gauges_connection_dict


#%% Next function how to calculate the balance
def calculate_network_balance(ts_data=pd.DataFrame(),
                              network_dict=dict(),
                              confidence_acceptance_level=0.05):   
    """
    

    Parameters
    ----------
    ts_data : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    network_dict : TYPE, optional
        DESCRIPTION. The default is dict().
    confidence_acceptance_level : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    None.

    """
    
    #%% now we loop through the time steps
    sections_meta=pd.DataFrame(columns=['upstream_point','downstream_point','gauge','balance','Date'])
    nr_ts=ts_data.shape[0]
    #if below threshold, balance is not significant
    confidence_acceptance_level=0.05
    
    
    for gauge in network_dict.keys():
        
        #we write some empty dataframes for the tributaries
        ts_distributaries=pd.concat([pd.Series(0)]*nr_ts)
        ts_distributaries.index=ts_data.index
        ts_tributaries=pd.concat([pd.Series(0)]*nr_ts)
        ts_tributaries.index=ts_data.index
        ts_data_gauge_up=pd.concat([pd.Series(0)]*nr_ts)
        ts_data_gauge_up.index=ts_data.index    
        
        print('add water balance to gauge ',gauge)
        df_section=pd.DataFrame(columns=['upstream_point','downstream_point','balance'])
        df_row=pd.Series(dict((k, network_dict[gauge][k]) for k in ['upstream_point','downstream_point'] if k in network_dict[gauge]),name=gauge).to_frame().T
        #name of gauge and id
        df_row['gauge']=gauge
        df_row['id']=network_dict[gauge]['id']
        df_section=pd.concat([df_row]*nr_ts, ignore_index=True)
        #first case is easy if upstream is a distributary, we cant give any balance, it is actually covered in another scenario
        if network_dict[gauge]['upstream_point']=='river_junction':
    
            # in this case balance is nan
            df_section=pd.concat([df_row]*nr_ts, ignore_index=True)
            df_section['balance']=np.nan
            df_section['Date']=ts_data.index
        else:
            #in all other cases we just compute them from distributary and tributary gauges
            if len(network_dict[gauge]['gauge_up']) > 0:
                ts_data_gauge_up=ts_data[network_dict[gauge]['gauge_up'].index.tolist()].sum(axis=1)
    
    
            # in this case we have to compute the difference between gauge, tributaries and distributaries
            if 'distributaries_up' in network_dict[gauge].keys() and len(network_dict[gauge]['distributaries_up'])>0:
                distri_gauges=network_dict[gauge]['distributaries_up'][network_dict[gauge]['distributaries_up'].upstream_point!='river_spring']
                if len(distri_gauges)>0:
                    ts_distributaries=ts_data[distri_gauges.upstream_point.tolist()].sum(axis=1)
    
    
                
            if 'tributaries_up' in network_dict[gauge].keys() and len(network_dict[gauge]['tributaries_up'])>0:
                tri_gauges=network_dict[gauge]['tributaries_up'][network_dict[gauge]['tributaries_up'].upstream_point!='river_spring']
                if len(tri_gauges)>0:
                    ts_tributaries=ts_data[tri_gauges.upstream_point.tolist()].sum(axis=1)
    
    
                
            #calculate the water balance
    
            df_section['balance']=(ts_data[gauge]-ts_data_gauge_up-ts_tributaries+ts_distributaries).values
            df_section['Date']=ts_data.index
            df_section['balance_confidence']=df_section['balance'].divide((ts_data[gauge]+ts_data_gauge_up).values)
            
         
            
            print('add water balance to gauge ',gauge, 'done')
            
    
        sections_meta=sections_meta.append(df_section)
        
    
            
    #%% Finally we get out the data
    #spreadsheet with all data
    sections_meta.loc[abs(sections_meta['balance_confidence'])<confidence_acceptance_level,'balance']=np.nan
    q_diff=sections_meta.pivot(index='Date',columns='downstream_point',values='balance')
    
    return sections_meta,q_diff


def map_network_sections(
    network_dict: Dict, 
    gauge_meta: pd.DataFrame, 
    network: gpd.GeoDataFrame
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
    
    gdf_balances=gpd.GeoDataFrame()
    
    #%% We will loop through the gauge sections and extract the relevant stream reaches and clip them 
    for _,gauge in network_dict.items():
        #first we get the line within the main reach
        if gauge['reach_name'] not in network.reach_name.tolist():
    
            print(gauge['reach_name'], 'not in network data, check correct names')
            continue
        river_line=network[network.reach_name==gauge['reach_name']].geometry.iloc[0]
        pnt_gauge=gauge_meta.loc[gauge_meta.index==gauge['downstream_point'],'geometry'].iloc[0]
        
        #get all river points as geodataframe
        river_pnts=gpd.GeoDataFrame(geometry=[Point(pt) for pt in river_line.coords])
        #reduce the points to all upstream the river gauge
        pnt_id_downstream=river_pnts.iloc[river_pnts.distance(pnt_gauge).idxmin():,:].iloc[0].name
        
        #now we make some conditions depending on the upstream point
        if gauge['upstream_point'] == 'river_spring' or gauge['upstream_point'] == 'river_junction' or gauge['upstream_point'] == 'river_mouth':
            pnt_id_upstream=river_pnts.iloc[-1,:].name
        else:
            pnt_gauge_up=gauge_meta.loc[gauge_meta.index==gauge['upstream_point'],'geometry'].iloc[0]
            pnt_id_upstream=river_pnts.iloc[river_pnts.distance(pnt_gauge_up).idxmin():,:].iloc[0].name
        
        
        #depending which is larger we select from the reach, because some times there are problems with the stream points
        if pnt_id_upstream>pnt_id_downstream:
            river_pnts=river_pnts.iloc[pnt_id_downstream:pnt_id_upstream+1]
        elif pnt_id_upstream<pnt_id_downstream:
            river_pnts=river_pnts.iloc[pnt_id_upstream:pnt_id_downstream+1]
        else:
            print('Stream Line needs at least two points, we move by one point, but you should check geometry')
            river_pnts=river_pnts.iloc[pnt_id_upstream:pnt_id_downstream+2]
            
        section_line=LineString(MultiPoint(river_pnts.geometry.to_list()))
        
        #get the lines of the tributaries
        trib_lines=list()

        
        for branch_name in ['tributaries_up','distributaries_up']:
    
            #we loop trough the dataset
            for _,branch in gauge[branch_name].iterrows():
                #extract the river line if available
                if branch['Gewaesser'] not in network.reach_name.tolist():
                    print(branch['Gewaesser'], 'not in network data, check correct names')
                    continue
                river_line=network[network.reach_name==branch['Gewaesser']].geometry.iloc[0]
                
                    
                #first we check whether there is really data in the dataset
                if len(gauge[branch_name])>0:
                    
                    #we get the river line
                    if branch['Gewaesser'] not in network.reach_name.tolist():
        
                        print(branch['Gewaesser'], 'not in network data, check correct names')
                        continue
                    #extract the river line
                    river_line=network[network.reach_name==branch['Gewaesser']].geometry.iloc[0]
                    #get all river points as geodataframe
                    river_pnts=gpd.GeoDataFrame(geometry=[Point(pt) for pt in river_line.coords])
                    
                    #next we will find upstream and downstream
                    #downstream is always the points
                    if branch_name=='tributaries_up':
                        pnt_id_downstream=0
                        if branch.upstream_point == 'river_spring':
                            pnt_id_upstream=len(river_pnts)-1
        
                    elif branch_name=='distributaries_up':
                        pnt_id_downstream=len(river_pnts)-1
                        if branch.upstream_point == 'river_spring':
                            pnt_id_upstream=0
                    #upstream depends whether there is a gauge or not
                    if branch.upstream_point != 'river_spring':
                        
                        pnt_branch_gauge=gauge_meta.loc[gauge_meta.index==branch['upstream_point'],'geometry'].iloc[0]
                        pnt_id_upstream=river_pnts.distance(pnt_branch_gauge).idxmin()
                    
                    #we rearrange if it does make sense with the flow direction
                    if pnt_id_upstream>pnt_id_downstream:
                        river_pnts_extracted=river_pnts.iloc[pnt_id_downstream:pnt_id_upstream+1]
                    elif pnt_id_upstream<pnt_id_downstream:
                        river_pnts_extracted=river_pnts.iloc[pnt_id_upstream:pnt_id_downstream+1]
                    else:
                        print('Stream Line needs at least two points, we move by one point, but you should check geometry')
                        river_pnts_extracted=river_pnts.iloc[pnt_id_upstream:pnt_id_downstream+2]
                    
                    #we append the trib geometries
                    trib_lines.append(LineString(MultiPoint(river_pnts_extracted.geometry.to_list())))
        
        
        trib_lines.append(section_line)            
        section_lines=MultiLineString(lines=trib_lines)
        
               
    
        #if we have tributaries we also take their geometrical information
        #for 
        
        #we make a dataframe 
        df_columns_dict=dict((k, gauge[k]) for k in ['id', 'reach_name','upstream_point','downstream_point'] if k in gauge.keys())    
        gdf_balance=gpd.GeoDataFrame(pd.DataFrame.from_dict({0:df_columns_dict}).T,geometry=[section_lines],crs=network.crs)
        
        #append
        gdf_balances=gdf_balances.append(gdf_balance,ignore_index=True)
        
        
    return gdf_balances.set_crs(network.crs)


#add a function for time series manipulation
def aggregate_time_series(data_ts,analyse_option='overall_mean'):
    
    if analyse_option is None:
        print('No data aggregation option select, continue with original time series')
        return data_ts
    
    #just for testing we take the mean
    if analyse_option=='overall_mean':
        print(analyse_option, 'takes entire time series')
        ts_stats=data_ts.mean()
        stats_name='mean_discharge_m_s'
        ts_stats=ts_stats.rename(stats_name).to_frame().T

    elif analyse_option=='annual_mean':
        ts_stats=data_ts.resample('Y').mean()

    elif analyse_option=='summer_mean':
        print('Calculating summer mean (June to September)')
        ts_stats = data_ts.loc[data_ts.index.month.isin([6, 7, 8, 9])].resample('Y').mean()

    #daily calculations    
    elif analyse_option=='daily':
        print('Calculating daily statistics')
        ts_stats=data_ts.copy()
    else:
        print('Invalid aggregation option selected, continuing with original time series')
        return data_ts
    
    ts_stats.index=ts_stats.index.strftime("%Y-%m-%d")
    
    return ts_stats

    
#%% A function which connects all functions
def get_section_water_balance(gauge_data=pd.DataFrame(),
                          data_ts=pd.DataFrame(),
                          network=gpd.GeoDataFrame(),
                          tributary_connections=pd.DataFrame(),
                          distributary_connections=pd.DataFrame(),
                          confidence_acceptance_level=0.05,
                          time_series_analysis_option='overall_mean',
                              ):
    """
    

    Parameters
    ----------
    gauge_meta : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    ts_data : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    network : TYPE, optional
        DESCRIPTION. The default is gpd.GeoDataFrame().
    tributary_connections : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    distributary_connections : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    confidence_acceptance_level : TYPE, optional
        DESCRIPTION. The default is 0.05.
    ts_analyse_options : TYPE, optional
        DESCRIPTION. The default is 
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    

    #%% We do some small data manipulations
    ts_stats=aggregate_time_series(data_ts=data_ts,analyse_option=time_series_analysis_option)
    
    #synchrnonize our datasets
    gauge_data=gauge_data.loc[gauge_data.index.isin(ts_stats.columns),:]
    #reduce the datasets to all which have metadata
    ts_stats=ts_stats[gauge_data.index.to_list()]
    print(ts_stats.shape[1], 'gauges with valid meta data')
    
    # our gauge data has to converted to geodataframe
    #make a geodataframe out of the data
    geometry = [Point(xy) for xy in zip(gauge_data.ostwert, gauge_data.nordwert)]   
    
    gauge_data = gpd.GeoDataFrame(gauge_data, crs=network.crs, geometry=geometry)
    # clean it
    
    gauge_data=gauge_data[~gauge_data.geometry.is_empty]
    
    #%% run the main functions
    
    gauges_connection_dict=generate_upstream_network(gauge_meta=gauge_data,
                                   tributary_connections=tributary_connections,
                                   distributary_connections=distributary_connections)
    
    
    sections_meta,q_diff=calculate_network_balance(ts_data=ts_stats,
                                  network_dict=gauges_connection_dict,
                                  confidence_acceptance_level=confidence_acceptance_level)
    
    gdf_network_map=map_network_sections(network_dict=gauges_connection_dict,
                         gauge_meta=gauge_data,
                         network=network)
    
    # return the results
    return sections_meta,q_diff,gdf_network_map
    
    
    
    
    
    

def test_waterbalance():
#%% We run the model
#%% load the data of the gauges

    gauge_data=pd.read_csv(os.path.join(os.getcwd(),'input','pegel_uebersicht.csv'))
    
    
    
    
    zufluesse=pd.read_csv(os.path.join(os.getcwd(),'input','zufluesse.csv'))
    abfluesse=pd.read_csv(os.path.join(os.getcwd(),'input','abfluesse.csv'))
    
    #we load time series data
    data_ts=pd.read_csv(os.path.join(os.getcwd(),'input','pegel_ts.csv'))
    data_ts=data_ts.set_index(pd.to_datetime(data_ts['Datum']),drop=True).drop(columns=['Datum'])
    
    #we load network data
    network=gpd.read_file(os.path.join(os.getcwd(),'input','Network_z.shp'))
    
    
    
    #%% We do some small data manipulations
    ts_stats=aggregate_time_series(data_ts=data_ts,analyse_option='overall_mean',
                              start_date='2020-12-01',end_date='2020-12-15')
    
    #synchrnonize our datasets
    gauge_data=gauge_data.loc[gauge_data.pegelname.isin(ts_stats.columns),:]
    #reduce the datasets to all which have metadata
    ts_stats=ts_stats[gauge_data.pegelname.to_list()]
    print(ts_stats.shape[1], 'gauges with valid meta data')
    
    # our gauge data has to converted to geodataframe
    #make a geodataframe out of the data
    geometry = [Point(xy) for xy in zip(gauge_data.ostwert, gauge_data.nordwert)]   
    
    gauge_data = gpd.GeoDataFrame(gauge_data, crs=network.crs, geometry=geometry)
    # clean it
    
    gauge_data=gauge_data[~gauge_data.geometry.is_empty]
    
    
    #%% We run through the main functions
    
    
    gauges_connection_dict=generate_upstream_network(gauge_meta=gauge_data,
                                   tributary_connections=zufluesse,
                                   distributary_connections=abfluesse)
    
    
    sections_meta,q_diff=calculate_network_balance(ts_data=ts_stats,
                                  network_dict=gauges_connection_dict,
                                  confidence_acceptance_level=0.05)
    
    gdf_network_map=map_network_sections(network_dict=gauges_connection_dict,
                         gauge_meta=gauge_data,
                         network=network)
        
       
    #we finally map the mean on the data and provide the output
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),'output'),exist_ok=True)
    df_sections_mean=sections_meta.groupby('id').mean().rename(columns={'balance':'q_dif_mean_m3_s'})
    
    gdf_network_map=pd.concat([gdf_network_map.set_index('id'),df_sections_mean],axis=1).reset_index(drop=False)
    
    gdf_network_map.to_file(os.path.join(os.getcwd(),'output','stream_water_balance.gpkg'),driver='GPKG')

  



