"""
Call Function
"""
from datetime import datetime

from sbat.sbat import model
import pandas as pd
import os
import geopandas as gpd
#functiton to dateparse
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')

#%% load data 
gauge_ts=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','pegel_ts.csv'),
                     index_col=0,
                     parse_dates=['Datum'], 
                     date_parser=dateparse)

gauge_meta=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','pegel_uebersicht.csv'),index_col=0)

#%% generate the object
sbat=model(gauge_time_series=gauge_ts,
           gauge_network=gpd.GeoDataFrame(),
           gauge_metadata=gauge_meta,
           valid_datapairs_only=True,
           decadal_stats=True,
           dropna_axis=0)

#get discharge data
sbat.get_discharge_stats()

#get baseflow
sbat.get_baseflow(methods=['UKIH','Fixed'],
                  compute_bfi=True, update_metadata=True,plot=False)

#compute the master recession curve
sbat.get_recession_curve(curve_type='baseflow',plot=False)



sections_meta,q_diff,gdf_network_map = sbat.get_waterbalance(network_geometry=gpd.read_file(os.path.join(os.path.dirname(__file__),'input','Network_z.shp')),
                                                             tributary_connections=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','zufluesse.csv')),
                                                             distributary_connections=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','abfluesse.csv')),
                                                             flow_type='baseflow',
                                                             confidence_acceptance_level=0.05,
                                                             ts_analysis_option='daily')


