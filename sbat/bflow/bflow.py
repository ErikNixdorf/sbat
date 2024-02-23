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
    bf_daily = bf_daily_raw.reset_index().melt(id_vars='date').set_index('date')
    
    #get output of performance
    performance_metrics=dict(zip(['kge_' + col for col in methods],KGEs))
    
    # get monthly values and bfi
    bf_monthly = bf_daily.groupby('variable').resample('m').mean(numeric_only=True)
    # bfi computation if requested
    if compute_bfi:
        # we compute the BFI
        Q_monthly = Q.resample('m').mean(numeric_only=True)
        bfi_monthly = bf_daily.groupby('variable').resample('m').mean(numeric_only=True).divide(Q_monthly,axis=0)

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

def plot_along_streamlines(stream_gauges: pd.DataFrame,
                           stream_name: str = 'river',
                           sort_column: str = 'river_km',
                           para_column: str = 'q_daily',
                           gauge_ticklabels: List[str] = None,
                           plot_context='talk',
                           fig_width=10,
                           output_dir: Union[str, Path] = Path.cwd() / 'bf_analysis' / 'figures') -> Tuple:
    """
    Plot a line chart of a given parameter (e.g. daily discharge) along the streamlines of a river system,
    and a separate line chart for each decade of data available.

    Parameters:
    -----------
    stream_gauges: pd.DataFrame
        A pandas DataFrame containing the data to plot, with one row per gauge station and columns for the
        parameters of interest (e.g. 'q_daily' for daily discharge), the station name and location (e.g. 'station_id',
        'river_km') and the decade of observation (e.g. 'decade').
    stream_name: str, optional (default='river')
        The name of the river system to plot.
    sort_column: str, optional (default='river_km')
        The column of `stream_gauges` to use for sorting the data along the river system. By default, this is the
        river kilometre column.
    para_column: str, optional (default='q_daily')
        The name of the column in `stream_gauges` containing the parameter of interest to plot (e.g. 'q_daily' for daily
        discharge). This column should contain numeric values.
    gauge_ticklabels: list of str, optional (default=[])
        A list of labels to use for the x-axis ticks, one per gauge station. By default, no labels are shown.
    output_dir: str or Path-like, optional (default='bf_analysis/figures')
        The directory where to save the output plots. By default, the plots are saved in a 'bf_analysis/figures'
        subdirectory of the current working directory.

    Returns:
    --------
    None
    """    
    sns.set_context(plot_context)
    fig, ax = plt.subplots(figsize=(fig_width,0.70744 *fig_width))
    sns.lineplot(data=stream_gauges, x=sort_column, y=para_column,
                      marker='o', linewidth=2, markersize=10, color='dodgerblue')
    # we give an error band if available
    if 'mean' in para_column:
        std_col = f"{para_column.split('_mean')[0]}_std"
        ax.fill_between(stream_gauges[sort_column], 
                        stream_gauges[para_column] - stream_gauges[std_col],
                        stream_gauges[para_column] + stream_gauges[std_col], 
                        alpha=0.2, color='k')
    
    plt.title(f'{para_column} along {stream_name}')
    plt.ylabel(para_column)
    plt.xlabel(sort_column)
    ax.set_xticks(stream_gauges[sort_column].unique())
    plt.xticks(rotation=90)
    if gauge_ticklabels is not None:
        ax.set_xticklabels(gauge_ticklabels)
    plt.tight_layout()
    fig.savefig(Path(output_dir, f'{stream_name}_{para_column}_along_streamlines.png'), dpi=300)
    plt.close()

    #plot for each decade
    fig, ax = plt.subplots(figsize=(fig_width,0.70744 *fig_width))
    sns.lineplot(data=stream_gauges, x=sort_column, y=para_column, hue='decade',
                      marker='o', linewidth=2, markersize=10, palette='rocket',
                      hue_order=stream_gauges['decade'].sort_values())
    
    plt.title(f'{para_column} along {stream_name} and decade')
    plt.ylabel(para_column)
    plt.xlabel(sort_column)
    ax.set_xticks(stream_gauges[sort_column].unique())
    plt.xticks(rotation=90)
    if gauge_ticklabels is not None:
        ax.set_xticklabels(gauge_ticklabels)
    plt.tight_layout()
    fig.savefig(Path(output_dir, f'{stream_name}_{para_column}_decadal_along_streamlines.png'), dpi=300)
    plt.close()
    
    return None


def plot_bf_results(ts_data: pd.DataFrame = pd.DataFrame(),
                     meta_data: pd.DataFrame = pd.DataFrame(),
                     parameter_name: str = 'bf_monthly',
                     plot_along_streams: bool = True,
                     output_dir: Union[str, Path] = Path.cwd() / 'bf_analysis' / 'figures',
                     fig_size=10,
                     plot_context='poster',
                     ) -> None:
    """
    Plot the results of the baseflow calculation.

    Parameters
    ----------
    ts_data : pandas.DataFrame, optional
        A pandas DataFrame containing the timeseries data. The default is an empty DataFrame.
    meta_data : pandas.DataFrame, optional
        A pandas DataFrame containing metadata for the timeseries data. The default is an empty DataFrame.
    parameter_name : str, optional
        The name of the parameter to plot. The default is 'bf_monthly'.
    plot_along_streams : bool, optional
        Whether to plot the results along each stream in separate subplots. The default is True.
    output_dir : str or Path, optional
        The directory in which to save the output plots. The default is './bf_analysis/figures'.

    Returns
    -------
    None
    """

    # first we generate the output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    #default seaborn setting
    sns.set_context(plot_context)
    performance_cols= [col for col in meta_data.columns if col.startswith('kge_')]
    
    parameter_titles={'bf_daily':'daily baseflow',
            'bf_monthly':'monthly baseflow',
            'bfis_monthly':'baseflow index'}
    #%% first we plot some histogram and boxplots for each station
    for gauge_name,subset in ts_data.groupby('gauge'):
        subset= subset.reset_index()
        try:
            performance_dict = meta_data.loc[gauge_name,performance_cols].iloc[0].to_dict()
        except:
            performance_dict = meta_data.loc[gauge_name,performance_cols].to_dict()
            pass
        performance_str =r"$\bf{Performance}$"+'\n'+'\n'.join(f'{key}: {np.round(value,3)}' for key, value in performance_dict.items())
        # histogram over all data
        fig,ax = plt.subplots(figsize=(fig_size, fig_size)) 
        sns.histplot(data=subset, x='value', kde=True)
        plt.title(f'{parameter_titles[parameter_name]} at {gauge_name}')
        plt.tight_layout()
        fig.savefig(Path(output_dir,f'{gauge_name}_histplot_{parameter_name}.png'), dpi=300)
        plt.close()
        # for each method
        if 'variable' in subset.columns:
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            sns.histplot(data=subset, x='value', hue='variable', kde=True)
            plt.legend(title='bf_method', loc='upper right', labels=subset.variable.unique())
            ax.text(ax.get_xlim()[1]*0.5,
                    ax.get_ylim()[1]*0.6,
                    performance_str,
                    bbox=dict(facecolor='white', alpha=0.5),
                    )
            plt.title(f'{parameter_titles[parameter_name]} at {gauge_name}')
            plt.tight_layout()
            fig.savefig(Path(output_dir, f'{gauge_name}_method_histplot_{parameter_name}.png'), dpi=300)
            plt.close()        
            # we further make some boxplots
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            sns.boxplot(data=subset, x='variable', y='value')
            plt.xticks(rotation=90)
            plt.xlabel(gauge_name)
            plt.ylabel(parameter_name)
            plt.title(f'{parameter_titles[parameter_name]} at {gauge_name}')
            plt.tight_layout()
            fig.savefig(Path(output_dir, f'{gauge_name}_method_boxplot_{parameter_name}.png'), dpi=300)
            plt.close()
        #across all decades
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        subset['decade'] = [x[0:3] + '5' for x in subset['date'].dt.strftime('%Y')]
        sns.boxplot(data=subset, x='value',y='decade',hue='variable')
        plt.title(f'{parameter_titles[parameter_name]} at {gauge_name}')
        plt.tight_layout()
        fig.savefig(Path(output_dir, f'{gauge_name}_decade_boxplot_{parameter_name}.png'), dpi=300)
        plt.close()

    
    #%% if we have daily dataset our calculation ends here
    if 'daily' in parameter_name:
        return

    #%% next we plot the top 15 gauges with the largest deviations
    if len(meta_data) < 15:
        index_max = len(meta_data)
    else:
        index_max = 15
    # plot
    cv_col = f'{parameter_name}_cv'
    meta_data.index.name = 'gauge'
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.barplot(data=meta_data.reset_index().sort_values(cv_col, ascending=False)[0:index_max], x=cv_col,
                y='gauge').set(title=cv_col)
    fig.savefig(Path(output_dir, f'Gauges_with_largest_{cv_col}.png'), dpi=300, bbox_inches="tight")
    plt.close()
    

    #%% we make lineplots along all river systems
    para_cols = [cv_col]
    para_cols.append(f'{parameter_name}_mean')    
    #create a subset for each stream
    for stream,stream_gauges in meta_data.reset_index().groupby('stream'):        
        #get river km
        stream_gauges['river_km'] = stream_gauges['distance_to_mouth'].max() - stream_gauges[
            'distance_to_mouth']
        stream_gauges = stream_gauges.sort_values('river_km')
        gauge_ticklabels = stream_gauges['gauge'].unique()        
        #plot for each parameter
        for para_col in para_cols:
            plot_along_streamlines(stream_gauges = stream_gauges,
                                       stream_name = stream,
                                       sort_column = 'river_km',
                                       para_column = para_col,
                                       gauge_ticklabels = gauge_ticklabels,
                                       output_dir = output_dir)

    return None

    
    
