"""
Call Function
"""
from datetime import datetime

from sbat.sbat import Model
import pandas as pd
import os
import geopandas as gpd

#%% generate the object
sbat=Model()

#get discharge data
sbat.get_discharge_stats()

#get baseflow
sbat.get_baseflow()
#compute the master recession curve

sbat.get_recession_curve()

sections_meta,q_diff,gdf_network_map = sbat.get_water_balance()

"""
sections_meta,q_diff,gdf_network_map = sbat.get_water_balance(network_geometry=gpd.read_file(os.path.join(os.path.dirname(__file__),'input','Network_z.shp')),
                                                             tributary_connections=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','zufluesse.csv')),
                                                             distributary_connections=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','abfluesse.csv')),
                                                             flow_type='baseflow',
                                                             confidence_acceptance_level=0.05,
                                                             time_series_analysis_option='summer_mean')
"""
