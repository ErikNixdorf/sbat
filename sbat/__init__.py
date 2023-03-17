"""
sbat package to generate an automatized stream balance analysis
"""
__version__ = "0.1"

# System
import os

from os.path import dirname
from sys import path

path.insert( 0 , dirname( __file__ ) ) ;

# Date and Time
from datetime import datetime, timedelta

# Data
import numpy as np

# Utilities
import configparser

#from sbat.bflow.bflow import compute_baseflow,add_gauge_stats,plot_bf_results
#from sbat.recession.recession import analyse_recession_curves,plot_recession_results
#from sbat.waterbalance.waterbalance import get_section_water_balance
