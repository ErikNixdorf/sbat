"""
Call Function
"""
from datetime import datetime

from sbat.sbat import main
import pandas as pd
import os
import geopandas as gpd

#%% generate the object
sbat=main(output=True)
"""
#get discharge data
sbat.get_discharge_stats()

#get baseflow
sbat.get_baseflow()
#compute the master recession curve

sbat.get_recession_curve()

sections_meta,q_diff,gdf_network_map = sbat.get_water_balance()
"""

