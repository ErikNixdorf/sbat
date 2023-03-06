"""
sbat package to generate an automatized stream balance analysis
"""
__version__ = "0.1"

# System
import os
import sys

# Date and Time
from datetime import datetime, timedelta

# Data
import numpy as np

# Utilities
import configparser

from .baseflow.baseflow import compute_baseflow,add_gauge_stats,plot_bf_results
from .recession.recession import analyse_recession_curves,plot_recession_results
from .waterbalance.waterbalance import get_section_water_balance
