"""
This module computes monthly baseflow using the separation method of Kille (1970),
extended and formalized by Demuth 1993
"""
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Optional
# functiton to dateparse
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')

demuth_logger = logging.getLogger('sbat.demuth')
# %% Functions for our tool
# get monthly minima
def get_monthly_nmq(Q: pd.DataFrame = pd.DataFrame(), 
                    min_days: int = 15, 
                    minimum_years: int = 10
                    ) -> pd.Series:
    """
    Calculates the monthly NMQ (non-exceedance flow) from a time series of streamflow values.

    Parameters
    ----------
    Q : pandas.Series
        A time series of streamflow values.
    min_days : int, optional
        The minimum number of days with non-missing data required for a month to be included in the NMQ calculation.
        The default is 15.
    minimum_years : int, optional
        The minimum number of complete years of data required for the function to return a result.
        The default is 10.

    Returns
    -------

     nmq: A pandas.Series containing the monthly NMQ values.
    Raises
    ------
    ValueError
        If the input time series has fewer than `minimum_years` complete years of data.
    """

    # calculate the data points per month
    data_per_month = Q.groupby(pd.Grouper(freq='M')).count()
    # calculate the nmq
    nmq = Q.groupby(pd.Grouper(freq='M')).min()
    # add dates per month
    nmq['dates_per_month'] = data_per_month
    # reduce all with less then minimum days
    nmq = nmq[nmq.dates_per_month >= min_days]
    nmq = nmq.drop(columns=['dates_per_month'])
    # check for minimum length of time series
    valid_years=sorted(nmq.index.year.unique())
    if len(valid_years) >= minimum_years:
        return nmq.squeeze().rename('nmq')
    else:
        demuth_logger.info(f"Time series has {len(valid_years)} complete years of data, which is less than the minimum of {minimum_years} years.")
        return None


# sort split dataset prior to
def sort_split_ts(nmq: pd.Series) -> tuple:
    """
    Sorts and splits the dataset into two halves based on the median value.

    Parameters
    ----------
    nmq_data : pd.Series
        The NMQ data as a pandas Series.

    Returns
    -------
    tuple
        A tuple containing the month indices, smaller half of data, and larger half of data.
    """
    
    if nmq.empty:
        raise ValueError("The input DataFrame is empty")
    # first we calculate the thresholds
    nmq_median = nmq.median()
    nmq_min = nmq.quantile(0.05)

    # we reset the index and sort the dataset
    nmq_sorted = nmq.sort_values().copy()
    nmq_sorted = nmq_sorted.reset_index()

    month_indices = dict(zip(nmq_sorted.index, nmq_sorted['date']))
    # change to series
    nmq_sorted = nmq_sorted['nmq']

    # get smaller half
    nmq_s_half = nmq_sorted[nmq_sorted <= nmq_median]

    # delete all below 5% threshold
    nmq_s_half = nmq_s_half[nmq_s_half >= nmq_min]

    # we get the second half of data
    nmq_l_half = nmq_sorted[nmq_sorted > nmq_median]

    return month_indices, nmq_s_half, nmq_l_half


# do the regression
def regress_baseflow(nmq_s_half: pd.Series, 
                     nmq_l_half: pd.Series, 
                     label_curve_type: bool = True) -> pd.DataFrame:
    """
    Calculate the baseflow of a stream according to the demuth methodusing the recursive digital filter method.

    Parameters
    ----------
    nmq_s_half : pd.Series
        A sorted pandas series containing the smaller half of the data.
    nmq_l_half : pd.Series
        A sorted pandas series containing the larger half of the data.
    label_curve_type : bool, optional
        If True, label the curve type as Type 1 or Type 2. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe containing the linearized nmq and its curve type (if label_curve_type=True).
    """
    # start with an initial linear regression
    reg_result_init = linregress(nmq_s_half.index, nmq_s_half.values)
    # get the initial r_value
    rvalue_init = reg_result_init.rvalue

    # we add larger values one by one until there will be no improvement of correlation
    counter = 1
    rvalue = rvalue_init
    while rvalue >= rvalue_init:
        nmq_extended = pd.concat([nmq_s_half, nmq_l_half[0:counter]])
        reg_result = linregress(nmq_extended.index, nmq_extended.values)
        rvalue = reg_result.rvalue
        counter += 1
        if counter > len(nmq_l_half):
            break

    # we compute the linearized nmq
    nmq_int = pd.Series(reg_result.intercept + reg_result.slope * np.array(range(0, nmq_l_half.index.max() + 1)),
                        name='baseflow').to_frame()

    if label_curve_type:
        # we check whether the critical point is in the upper 2/3 of the upper half of data or the intercept is negative
        if counter > len(nmq_l_half) / 3 or reg_result.intercept < 0:

            nmq_int['curve_type'] = 2
        else:
            nmq_int['curve_type'] = 1
    else:
        nmq_int['curve_type'] = np.nan

    return nmq_int


# %% the main function
def baseflow_demuth(Q: pd.Series = pd.Series(), 
                    gauge_name: str ='gauge',
                    reduce_excess_baseflow: bool = True,
                    ) -> pd.DataFrame:
    """
    Calculation of baseflow after Demuth(1993) and Kille(1970)

    Parameters
    ----------
    Q : pd.Series
        Streamflow data.
    
    gauge_name : str, optional
        Name of the gauge (default is 'gauge').

    reduce_excess_baseflow : bool, optional
        If True, baseflow results larger than the actual discharge will be set equal to discharge (default is True).
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing baseflow series, the name of the gauge, curve type, and date index.

    """

    # get monthy minimum
    nmq = get_monthly_nmq(Q)

    # if no data is included we drop it
    if nmq is None:
        demuth_logger.info(f'No Calculation possible for gauge {gauge_name} due to lack of data')
        # just write nans
        baseflow = pd.DataFrame({'baseflow': Q.resample('M').mean(), 'curve_type': 0})
        baseflow[gauge_name] = np.nan
        return baseflow

    # sort the data
    month_indices, nmq_s_half, nmq_l_half = sort_split_ts(nmq)

    # do the regression
    baseflow = regress_baseflow(nmq_s_half, nmq_l_half, label_curve_type=True)

    # use the months as index again
    baseflow['date'] = baseflow.index.map(month_indices)
    baseflow.set_index('date', drop=True, inplace=True)

    # we finally overwrite if baseflow is larger than the original flow by the nmq
    if reduce_excess_baseflow:
        # this way we also change the infinity ones
        baseflow.loc[~(baseflow.baseflow / nmq <= 1), 'baseflow'] = nmq[~(baseflow.baseflow / nmq <= 1)]

    baseflow.rename(columns={'baseflow': gauge_name}, inplace=True)
    demuth_logger.info(f'demuth baseflow was calculated for gauge {gauge_name}')
    return baseflow

