# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:40:59 2022
loads all our data and calculates the monthly baseflow
@author: Nixdorf.E
"""

import logging
from pathlib import Path

import baseflow as bf_package
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple, Union, List
from postprocess.plot import plot_along_streamlines
bflow_logger = logging.getLogger('sbat.bflow')

def clean_gauge_ts(Q: pd.Series) -> Optional[pd.Series]:
    """
    Cleans the gauge time series from NaNs.

    Parameters
    ----------
    Q : pd.Series
        The gauge time series.

    Returns
    -------
    Q : pd.Series or None
        The cleaned gauge time series. If there are only NaNs in the series,
        None is returned.

    """
    # remove starting and ending nan
    first_idx = Q.first_valid_index()
    last_idx = Q.last_valid_index()
    Q = Q.loc[first_idx:last_idx]

    # if there are only nan we do not need the data:
    if Q.isna().sum() == len(Q):
        bflow_logger.warning(f"compute baseflow for gauge {Q.name} not possible")
        return None
    return Q


def call_bf_package(Q: pd.Series, methods: str = 'all', 
                    basin_area: Optional[float] = None,
                    ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Function that converts pandas Series to numpy and calls the baseflow package

    Parameters
    ----------
    Q : pd.Series
        The time series of streamflow discharge observations
    methods : str, optional
        Method(s) used for baseflow separation. Default is 'all'.
        Available methods: 'BFI', 'BFI_N', 'Lyne_Hollick', 'Chapman', 'Eckhardt', 'FixedQ', 'ZRW', 'Cooper'
    area : float, optional
        Area of the catchment in square kilometers. If provided, it is converted to square meters for use in baseflow
        calculations. Default is None.

    Returns
    -------
    bf_daily : pd.DataFrame
        Time series of daily baseflow discharge
    KGEs : np.ndarray
        Kling-Gupta efficiency (KGE) values for each baseflow separation method used

    """
    # we write out the dates with NaNs
    nan_dates = Q.index[Q.isna()]

    # %% we prepare for the daily calculations
    # get the dates for the daily calculations
    date = np.rec.fromarrays([Q.index.year, Q.index.month, Q.index.day],
                             dtype=[('Y', 'i4'), ('M', 'i4'), ('D', 'i4')])
    # interpolate the Nans and convert to 0D Array
    Q_array = Q.interpolate().values.astype(float).flatten()

    if basin_area is not None:
        bflow_logger.info('Assume that area is in m2, recompute to km2')
        basin_area = basin_area / 1000 / 1000
    
    if isinstance(methods,List) and 'all' in methods:
        methods='all'

    b, KGEs = bf_package.separation(Q_array, date, area=basin_area, method=methods)

    # convert results back to DataFrame
    bf_daily = pd.DataFrame.from_records(b)
    bf_daily.index = Q.index

    # remove the dates where original data was nan
    bf_daily.loc[nan_dates.values, :] = np.nan

    return bf_daily, KGEs



# %%We calculate baseflows using the baseflow package
def compute_baseflow(Q: pd.Series, basin_area: float = None, methods: Union[str, List[str]] = 'all',
                     compute_bfi: bool = True) -> dict:
    """
       Compute baseflow for all gauges in the given data.
    
       Parameters
       ----------
       Q : pandas.Series
           The streamflow data for a single gauge.
       basin_area : float
           The area of the basin in m**2
       methods : str or list of str, optional
           The methods to use for baseflow separation. Can be one of 'all', or a single algorithm e.g. demuth
           for documentation check the external baseflow package
       compute_bfi : bool, optional
           Whether to compute baseflow index (BFI) as well. Defaults to True.
    
       Returns
       -------
        - 'bf_daily': a pandas DataFrame with daily baseflow values for a gauge.
        - 'bf_monthly': a pandas DataFrame with monthly baseflow values for a gauge.
        - 'bfi_monthly': a pandas DataFrame with monthly BFI values for a gauge, if compute_bfi is True.
        - 'performance_metrics': dictionary which contains the performance per method
        deviation of baseflow and BFI values, and Demuth curve type, if applicable.
       """
    # prepare the output data

    bf_monthly = pd.DataFrame()
    bfi_monthly = pd.DataFrame()
    bf_daily = pd.DataFrame()    
    performance_metrics = dict()
    #start the workflow
    gauge = Q.name
    bflow_logger.info(f'compute baseflow for gauge {gauge}')

    # clean the data prior to processing
    Q = clean_gauge_ts(Q)
    if Q is None:
        bflow_logger.warning(f'No calculation possible for gauge {gauge} due to lack of discharge data')
        return bf_daily, bf_monthly, bfi_monthly, performance_metrics      
    
    # call the baseflow module
    bf_daily_raw, KGEs = call_bf_package(Q, methods=methods, basin_area=basin_area)
    #convert
    bf_daily = bf_daily_raw.reset_index().melt(id_vars='date',
                                               var_name='bf_method',
                                               value_name='BF').set_index('date')
    
    #get output of performance
    performance_metrics=dict(zip(['kge_' + col for col in methods],KGEs))
    
    # get monthly values and bfi
    bf_monthly = bf_daily.groupby('bf_method').resample('m').mean(numeric_only=True)
    # bfi computation if requested
    if compute_bfi:
        # we compute the BFI
        Q_monthly = Q.resample('m').mean(numeric_only=True)
        bfi_monthly = bf_daily.groupby('bf_method').resample('m').mean(numeric_only=True).divide(Q_monthly,axis=0)
        bfi_monthly=bfi_monthly.rename(columns={'BF':'BFI'})

    bflow_logger.info(f'compute baseflow for gauge....{gauge}...done')

    return bf_daily, bf_monthly, bfi_monthly, performance_metrics

def add_gauge_stats(gauge_meta: pd.DataFrame, ts_data: pd.DataFrame, col_name: str = 'bf',
                        DECADAL_NAN_VALUE : int = -9999) -> pd.DataFrame:
    """
    Calculates the mean, standard deviation, and coefficient of variation (cv) of a time series subset and adds them to a copy
    of a DataFrame containing metadata for a gauge.

    Args:
        gauge_meta: A pandas DataFrame containing metadata for a gauge.
        ts_data: A pandas DataFrame containing the time series data for the gauges.
        col_name: A string representing the name of the column to be added to the gauge_meta DataFrame. Defaults to 'bf'.
        DECADAL_NAN_VALUE: An integer representing the value used to indicate a missing decade in gauge_meta. Defaults to -9999.

    Returns:
        A pandas DataFrame containing the modified gauge metadata with the added statistics.
    """
    #select the subset from time series
    if gauge_meta['decade'] == DECADAL_NAN_VALUE:
        time_series_subset= ts_data[gauge_meta.name]
    else:
        time_series_subset=ts_data.loc[str(int(gauge_meta['decade'])-5):str(int(gauge_meta['decade'])+4),gauge_meta.name]
    
    # Calculate the statistics
    mean_value = time_series_subset.mean()
    std_value = time_series_subset.std()
    cv_value = std_value / mean_value
    # Add mean, std, and cv to gauge_meta
    modified_gauge_meta = gauge_meta.copy()
    modified_gauge_meta[col_name + '_mean'] = mean_value
    modified_gauge_meta[col_name + '_std'] = std_value
    modified_gauge_meta[col_name + '_cv'] = cv_value
    
    return modified_gauge_meta




    
    
