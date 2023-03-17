# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:40:59 2022
loads all our data and calculates the monthly baseflow
@author: Nixdorf.E
"""

from .demuth import baseflow_demuth

import os
import pandas as pd
from datetime import datetime
import numpy as np
import baseflow as bf_package

import seaborn as sns
from matplotlib import pyplot as plt


def clean_gauge_ts(Q):
    """
    Cleans the gauge time series from nans

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.

    Returns
    -------
    Q : TYPE
        DESCRIPTION.

    """
    # remove starting and ending nan
    first_idx = Q.first_valid_index()
    last_idx = Q.last_valid_index()
    Q = Q.loc[first_idx:last_idx]

    # if there are only nan we do not need the data:
    if Q.isna().sum() == len(Q):
        print('compute baseflow for gauge....', Q.name, ' not possible')
        return None
    return Q


def call_bf_package(Q, methods='all', area=None):
    """
    Function that converts pandas df to numpy and calls the baseflow package

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # we write out the dates with NaNs
    nan_dates = Q.index[Q.isna()]

    # %% we prepare for the daily calculations
    # get the dates for the daily calculations
    date = np.rec.fromarrays([Q.index.year, Q.index.month, Q.index.day],
                             dtype=[('Y', 'i4'), ('M', 'i4'), ('D', 'i4')])
    # interpolate the Nans and convert to 0D Array
    Q_array = Q.interpolate().values.astype(float).flatten()

    if area is not None:
        print('Assume that area is in km2, recompute to m2')
        area = area / 1000 / 1000

    b, KGEs = bf_package.separation(Q_array, date, area=area, method=methods)

    # convert results back to DataFrame
    bf_daily = pd.DataFrame.from_records(b)
    bf_daily.index = Q.index

    # remove the dates where original data was nan
    bf_daily.loc[nan_dates.values, :] = np.nan

    return bf_daily, KGEs


def melt_gauges(df, additional_columns=dict({'a': 'b'})):
    """
    Small function to melt gauge data and add information

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    additional_columns : TYPE, optional
        DESCRIPTION. The default is dict({'a':'b'}).

    Returns
    -------
    df_melted : TYPE
        DESCRIPTION.

    """
    df_melted = df.melt()
    for key, value in additional_columns.items():
        df_melted[key] = value
    # add date
    df_melted['Datum'] = pd.concat([pd.Series(df.index)] * len(df.columns), ignore_index=True)

    return df_melted


# %%We calculate baseflows using the baseflow package
def compute_baseflow(data_ts, data_meta, methods='all',
                     compute_bfi=True,
                     calculate_monthly=True):
    # prepare the output data
    bf_output = dict()

    bfs_monthly = pd.DataFrame()
    bfis_monthly = pd.DataFrame()
    gauges_attributes = pd.DataFrame()
    bfs_daily = pd.DataFrame()
    # loop trough all gauges
    i = 0
    for gauge in data_ts.columns:
        # if gauge!='zittau_6':
        #    continue

        i += 1
        print('compute baseflow for gauge....', gauge)
        Q = data_ts[gauge]

        # clean the data prior to processing
        Q = clean_gauge_ts(Q)
        if Q is None:
            print('No calculation possible for gauge', gauge)
            continue
        # call the baseflow module
        if 'basin_area' in data_meta.columns:
            basin_area = data_meta.loc[Q.name, 'basin_area']
        else:
            basin_area = None

        bf_daily, KGEs = call_bf_package(Q, methods=methods, area=basin_area)

        bf_daily_melted = bf_daily.reset_index().melt(id_vars='Datum')
        bf_daily_melted['gauge'] = gauge
        bfs_daily = pd.concat([bfs_daily, bf_daily_melted])

        bf_output.update({'bf_daily': bfs_daily.set_index('Datum')})

        if calculate_monthly:
            # get monthly values
            # from previous methods
            bf_monthly = bf_daily.resample('m').mean()
            # compute demuth method
            if methods == 'all' or 'demuth' in methods:
                # merge with monthly values from Demuth
                bf_demuth = baseflow_demuth(Q.to_frame(), gauge_name=gauge)
                # if demuth has wrong curve type, we just write the data nan
                if bf_demuth.curve_type.iloc[0] == 2:
                    print('Demuth Curve Typ 2, write values to NaN')
                    bf_demuth[gauge] = np.nan
                bf_monthly = pd.concat([bf_monthly, bf_demuth.rename(columns={gauge: 'demuth'})['demuth']], axis=1)

            # append daily and monthly data
            # monthly
            bfs_monthly = pd.concat([bfs_monthly, melt_gauges(bf_monthly, additional_columns=dict({'gauge': gauge}))])
            # daily

            # compute the statistics per gauge
            KGE_cols = ['kge_' + col for col in bf_daily]
            gauge_attributes = pd.DataFrame(dict(zip(KGE_cols, KGEs)), index=[gauge])
            if methods == 'all' or 'demuth' in methods:
                gauge_attributes['demuth_curve_type'] = bf_demuth['curve_type'].iloc[0]
            else:
                gauge_attributes['demuth_curve_type'] = np.nan
            gauge_attributes[['nmq_mean_' + col for col in bf_monthly]] = bf_monthly.mean()
            gauge_attributes[['nmq_std_' + col for col in bf_monthly]] = bf_monthly.std()

            # bfi computation if requested
            if compute_bfi:
                # we compute the BFI
                bfi_monthly = bf_monthly.divide(Q.resample('m').mean(), axis=0)
                # append
                bfis_monthly = pd.concat([bfis_monthly,
                                          melt_gauges(bfi_monthly, additional_columns=dict({'gauge': gauge})).set_index(
                                              'Datum')])
                # compute gauge attributes
                gauge_attributes[['bfi_mean_' + col for col in bf_monthly]] = bfi_monthly.mean()
                gauge_attributes[['bfi_std_' + col for col in bf_monthly]] = bfi_monthly.std()

                bf_output.update({'bfi_monthly': bfis_monthly})

            # append the gauge attributes
            gauges_attributes = pd.concat([gauges_attributes, gauge_attributes])

            print('compute baseflow for gauge....', gauge, '...done')

            # update the dictionary
            bf_output.update({'bf_attributes': gauges_attributes,
                              'bf_monthly': bfs_monthly.set_index('Datum')
                              }
                             )
    return bf_output


def add_gauge_stats(df_data, gauge_meta, col_name='bf', decadal_stats=False):
    """
    Adds the mean of the dataset and the std

    Parameters
    ----------
    df_data : TYPE
        DESCRIPTION.
    gauge_meta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # compute the mean and std of the dataframe
    mean = float(df_data.mean())
    std = float(df_data.std())
    # add mean, std, and cv to gauge_meta
    gauge_meta = gauge_meta.assign(
        **{col_name + '_mean': mean, col_name + '_std': std, col_name + '_cv': std / mean}
    )

    gauge_meta.index.names = ['gauge']
    if decadal_stats == True:
        # add a column for decade

        df_data['decade'] = [x[0:3] + '5' for x in df_data.index.strftime('%Y')]

        df_decadal = df_data.groupby('decade').mean().rename(columns={'value': col_name + '_dec_mean'})
        df_decadal[col_name + '_dec_std'] = df_data.groupby('decade').std()
        # remove nan
        df_decadal = df_decadal.dropna()
        # add coefficient of variation
        df_decadal[col_name + '_dec_cv'] = df_decadal[col_name + '_dec_std'] / df_decadal[col_name + '_dec_mean']

        # change index to actual gauge
        df_decadal['gauge'] = gauge_meta.index[0]
        df_decadal = df_decadal.reset_index().set_index('gauge')

        # merge

        gauge_meta = gauge_meta.merge(df_decadal, how='outer', left_index=True, right_index=True)

    return gauge_meta


def plot_bf_results(data=dict(), meta_data=pd.DataFrame(), meta_data_decadal=pd.DataFrame(),
                    parameters_to_plot=['bf_daily', 'bf_monthly', 'bfi_monthly'],
                    streams_to_plot=['spree', 'lausitzer_neisse', 'schwarze_elster'],
                    output_dir=os.path.join(os.getcwd(), 'bf_analysis', 'figures'),
                    decadal_plots=True
                    ):
    """
    Plot the results of the baseflow calculation

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is dict().
    meta_data : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    streams_to_plot : TYPE, optional
        DESCRIPTION. The default is ['spree','lausitzer_neisse','schwarze_elster'].

    Returns
    -------
    None.

    """
    # first we generate the output dir
    os.makedirs(output_dir, exist_ok=True)
    # first we plot some histogram and boxplots
    for parameter in parameters_to_plot:
        if parameter in data.keys():
            if parameter == 'bf_attributes':
                continue
            # reset index
            subset = data[parameter].reset_index()
            # histogram over all data
            fig = plt.figure()
            s1 = sns.histplot(data=subset.reset_index(), x='value', kde=True)
            plt.title('Distribution of ' + parameter)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'distribution_of_' + parameter + '.png'), dpi=300)
            plt.close()

            # for each method
            fig = plt.figure()
            s2 = sns.histplot(data=subset, x='value', hue='variable', kde=True)
            plt.legend(title=parameter + '_method', loc='upper left', labels=subset.variable.unique())
            plt.title(parameter + ' Distribution per Method')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, parameter + '_distribution_per_method.png'), dpi=300)
            plt.close()
            # for each gauge
            palette = sns.color_palette(['grey'], len(subset.gauge.unique()))
            fig, ax = plt.subplots()
            s3 = sns.kdeplot(data=subset, x='value', hue='gauge', legend=False,
                             palette=palette, linewidth=0.5)
            plt.xlabel(parameter)
            plt.title(parameter + ' Distribution over Gauges')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, parameter + '_distribution_over_gauges.png'), dpi=300)
            plt.close()
            # we further make some boxplots
            fig, ax = plt.subplots()
            s5 = sns.boxplot(data=subset, x='variable', y='value')
            plt.xticks(rotation=90)
            plt.xlabel(parameter + ' Method')
            plt.ylabel(parameter)
            plt.title(parameter + ' Statistical Difference by Method')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, parameter + '_statistical_difference_by_method.png'), dpi=300)
            plt.close()
            # %%
            fig, ax = plt.subplots()
            fig.set_size_inches(10.5, 30.5, forward=True)
            fig.set_dpi(300)
            s6 = sns.boxplot(data=subset, x='value', y='gauge')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, parameter + '_per_gauge.png'), dpi=300)
            plt.close()

    # next we plot the top 15 gauges with the largest deviations
    if len(meta_data) < 15:
        index_max = len(meta_data)
    else:
        index_max = 15
    # compute
    cv_cols = [col for col in meta_data.columns if '_cv' in col]
    for cv_col in cv_cols:
        fig, ax = plt.subplots()
        sns.barplot(data=meta_data.reset_index().sort_values(cv_col, ascending=False)[0:index_max], x=cv_col,
                    y='gauge').set(title=cv_col)
        fig.savefig(os.path.join(output_dir, 'Gauges_with_largest_' + cv_col + '.png'), dpi=300, bbox_inches="tight")
        plt.close()
    # we make lineplots along the river systems
    para_cols = [col for col in meta_data.columns if '_mean' in col]
    para_cols.extend(cv_cols)
    for para_col in para_cols:
        for stream in streams_to_plot:

            stream_gauges = meta_data[meta_data.gewaesser == stream].reset_index()
            if len(stream_gauges) == 0:
                print('no gauges along stream', stream)
                continue
            stream_gauges['river_km'] = stream_gauges['km_muendung_hauptfluss_model'].max() - stream_gauges[
                'km_muendung_hauptfluss_model']
            stream_gauges = stream_gauges.sort_values('river_km')
            stream_gauges = stream_gauges[stream_gauges['gauge'] != 'eisenhuettenstadt']
            if stream == 'schwarze_elster':
                stream_gauges = stream_gauges[stream_gauges['gauge'] != 'eisenhuettenstadt']
                gauge_ticklabels = stream_gauges['gauge'].unique().tolist()
            else:
                gauge_ticklabels = [label.split('_')[0] for label in stream_gauges['gauge'].unique()]

            fig, ax = plt.subplots()
            s6 = sns.lineplot(data=stream_gauges, x='river_km', y=para_col,
                              marker='o', linewidth=2, markersize=10, color='dodgerblue')
            # we give an error band if available
            if 'mean' in para_col:
                std_col = '_'.join([i for i in para_col.split('_')[:-1]]) + '_std'
                s6.fill_between(stream_gauges['river_km'], stream_gauges[para_col] - stream_gauges[std_col],
                                stream_gauges[para_col] + stream_gauges[std_col], alpha=0.2, color='k')

            plt.title(para_col + ' along stream ' + stream)
            plt.ylabel(para_col)
            plt.xlabel('River Kilometer')
            ax.set_xticks(stream_gauges['river_km'].unique())
            plt.xticks(rotation=90)
            ax.set_xticklabels(gauge_ticklabels)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, para_col + '_' + stream + '.png'), dpi=300)
            plt.close()

    if decadal_plots:
        print('We finally need the decadal plots')
        para_cols = [col for col in meta_data_decadal.columns if 'dec_mean' in col]
        para_cols.extend([col for col in meta_data_decadal.columns if 'dec_cv' in col])

        # loop through data
        for para_col in para_cols:
            for stream in streams_to_plot:

                stream_gauges = meta_data_decadal[meta_data_decadal.gewaesser == stream].reset_index()
                if len(stream_gauges) == 0:
                    print('no gauges along stream', stream)
                    continue
                # https://stackoverflow.com/questions/62004561/is-this-an-error-in-the-seaborn-lineplot-hue-parameter
                stream_gauges['river_km'] = stream_gauges['km_muendung_hauptfluss_model'].max() - stream_gauges[
                    'km_muendung_hauptfluss_model']
                stream_gauges = stream_gauges.sort_values('river_km')
                stream_gauges = stream_gauges[stream_gauges['gauge'] != 'eisenhuettenstadt']
                if stream == 'schwarze_elster':
                    stream_gauges = stream_gauges[stream_gauges['gauge'] != 'eisenhuettenstadt']
                    gauge_ticklabels = stream_gauges['gauge'].unique().tolist()
                else:
                    gauge_ticklabels = [label.split('_')[0] for label in stream_gauges['gauge'].unique()]

                fig, ax = plt.subplots()
                s6 = sns.lineplot(data=stream_gauges, x='river_km', y=para_col, hue='decade',
                                  marker='o', linewidth=2, markersize=10, palette='rocket',
                                  hue_order=stream_gauges['decade'].sort_values())
                plt.title(para_col + ' along stream ' + stream)
                plt.ylabel(para_col)
                plt.xlabel('River Kilometer')
                ax.set_xticks(stream_gauges['river_km'].unique())
                plt.xticks(rotation=90)
                ax.set_xticklabels(gauge_ticklabels)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, para_col + '_' + stream + '.png'), dpi=300)
                plt.close()
