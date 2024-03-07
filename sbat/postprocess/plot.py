"""
The plot module which summarizes all the plotting functionality 
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import logging
plot_logger = logging.getLogger('sbat.postprocess')

def plot_along_streamlines(stream_ts : pd.DataFrame(),
                           stream_name: str = 'river',
                           sort_column: str = 'river_km',
                           para_column: str = 'q_daily',
                           gauge_ticklabels: List[str] = None,                           
                           plot_context='talk',
                           fig_width=10,                           
                           output_dir: Union[str, Path] = Path.cwd() / 'bf_analysis' / 'figures',
                           yaxis_labels=dict({'BF':'BF [$m^{3}$/s]',
                                          'BFI': 'BFI [-]',
                                          'Q': 'Q [$m^{3}$/s]',
                                          'Q*': 'Q* [$m^{3}$/s]'}),
                           ) -> Tuple:
    """
    Plot a line chart of a given parameter (e.g. daily discharge) along the streamlines of a river system,
    and a separate line chart for each decade of data available.

    Parameters:
    -----------
    stream_ts: pd.DataFrame
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
    
    def _set_title_label(xlabel=str,
                         ylabel=str,
                         xticks=List,
                         gauge_ticklabels=List,
                         title=str):
        ax = plt.gca()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_xticks(xticks)
        plt.xticks(rotation=90)
        if gauge_ticklabels is not None:
            ax.set_xticklabels(gauge_ticklabels)
            
            
    #%%first we check whether there is actually data to plot
    if all([np.isnan(entry) for entry in stream_ts[para_column].unique()]):
        plot_logger.info(f'No estimates for parameter{para_column} in dataset, skip plotting')
        return
    if 'sample_id' not in stream_ts.columns:
        stream_ts['sample_id'] = 0
    if 'bf_method' not in stream_ts.columns:
        stream_ts['bf_method'] = 'default'
    stream_ts_decade = stream_ts.copy().reset_index()
    if 'decade' not in stream_ts_decade.columns:
        stream_ts_decade['decade'] = [x[0:3] + '5' for x in stream_ts_decade['date'].dt.strftime('%Y')]

    xticks= stream_ts[sort_column].unique()
    #if new parameter appears, just integrate as new key value pair
    if para_column not in yaxis_labels.keys():
        yaxis_labels.update({para_column:para_column})

    if gauge_ticklabels is not None:
        sort_label=dict({'river_km':'Gauging Station'})
    else:
        sort_label=dict({'river_km':'Distance from Source [km]'})
        
    #define the subset columns which are the only relevant together with groupby cols
    subset_cols = [para_column,sort_column]
        
    #%%plot over time and generate certain differences of spreading
    groupby_cols = ['gauge']
    data = stream_ts.groupby(groupby_cols)[subset_cols].mean().reset_index()
    
    
    sns.set_context(plot_context)
    fig, ax = plt.subplots(figsize=(fig_width,0.70744 *fig_width))
    sns.lineplot(data=data, x=sort_column, y=para_column,
                      marker='o', linewidth=2, markersize=10, color='dodgerblue')
    # we give an error band if available
    stream_ts=stream_ts.sort_values(sort_column)
    ax.fill_between(stream_ts.groupby('gauge').first().sort_values(sort_column)[sort_column], 
                    stream_ts.groupby(sort_column)[para_column].mean().sort_index() - stream_ts.groupby(sort_column)[para_column].std().sort_index(),
                    stream_ts.groupby(sort_column)[para_column].mean().sort_index() + stream_ts.groupby(sort_column)[para_column].std().sort_index(), 
                    alpha=0.2, color='k')
    
    _set_title_label(xlabel=sort_label[sort_column],
                         ylabel=yaxis_labels[para_column],
                         xticks=xticks,
                         gauge_ticklabels=gauge_ticklabels,
                         title=f'Mean {para_column} at {stream_name}')
    
    plt.tight_layout()
    fig.savefig(Path(output_dir, f'{stream_name}_{para_column.replace("*","")}_mean_along_streamlines.png'), dpi=300)
    plt.close()
    
    
    #%%make a plot of the CV value
    cv_col_name = f'{para_column}_cv'
    groupby_cols = ['gauge']
    data = stream_ts.groupby(groupby_cols)[subset_cols].mean(numeric_only=True).reset_index()
    data_std = stream_ts.groupby(groupby_cols)[para_column].std().reset_index()
    data[cv_col_name] =  data_std[para_column]/ data[para_column]
    
    sns.set_context(plot_context)
    fig, ax = plt.subplots(figsize=(fig_width,0.70744 *fig_width))
    sns.lineplot(data=data, x=sort_column, y=cv_col_name,
                      marker='o', linewidth=2, markersize=10, color='red')

    _set_title_label(xlabel=sort_label[sort_column],
                         ylabel=cv_col_name+ ' [-]',
                         xticks=xticks,
                         gauge_ticklabels=gauge_ticklabels,
                         title=f'{cv_col_name} at {stream_name}')

    plt.tight_layout()
    fig.savefig(Path(output_dir, f'{stream_name}_{cv_col_name.replace("*","")}_along_streamlines.png'), dpi=300)
    plt.close()
    
    
    #cv plot with decade
    groupby_cols = ['gauge','decade']    
    data = stream_ts.groupby(groupby_cols)[subset_cols].mean(numeric_only=True).reset_index()
    data_std = stream_ts.groupby(groupby_cols)[para_column].std().reset_index()
    data[cv_col_name] =  data_std[para_column]/ data[para_column]
    
    if all([np.isnan(entry) for entry in data[cv_col_name].unique()]):
        plot_logger.info(f'No estimates for parameter{cv_col_name} in dataset, skip plotting')
        
    else:
    
        sns.set_context(plot_context)
        fig, ax = plt.subplots(figsize=(fig_width,0.70744 *fig_width))
        sns.lineplot(data=data, x=sort_column, y=cv_col_name,hue='decade',
                          marker='o', linewidth=2, markersize=10, 
                          palette='rocket',
                          hue_order=data['decade'].sort_values())

        _set_title_label(xlabel=sort_label[sort_column],
                         ylabel=cv_col_name+ ' [-]',
                         xticks=xticks,
                         gauge_ticklabels=gauge_ticklabels,
                         title=f'{cv_col_name} at {stream_name} per decade')

        plt.tight_layout()
        fig.savefig(Path(output_dir, f'{stream_name}_{cv_col_name.replace("*","")}_decade_along_streamlines.png'), dpi=300)
        plt.close()
    
    
    #%% Next we make a lineplot where we show the confidence interval from the sampling
    #process
    groupby_cols = ['gauge','sample_id','bf_method']
    data = stream_ts.groupby(groupby_cols)[subset_cols].mean().reset_index()
    #plot
    sns.set_context(plot_context)
    fig, ax = plt.subplots(figsize=(fig_width,0.70744 *fig_width))
    sns.lineplot(data=data, x=sort_column, y=para_column, hue = 'bf_method',
                      marker='o', linewidth=2, markersize=10, palette='mako_r',errorbar=("pi", 100),err_style='bars')

    _set_title_label(xlabel=sort_label[sort_column],
                     ylabel=yaxis_labels[para_column],
                     xticks=xticks,
                     gauge_ticklabels=gauge_ticklabels,
                     title=f'Average {para_column} per method at {stream_name}')

    plt.tight_layout()
    fig.savefig(Path(output_dir, f'{stream_name}_{para_column.replace("*","")}_method_dependence_mean_along_streamlines.png'), dpi=300)
    plt.close()
    
    
    #%% We plot over each decade with error style band 
    groupby_cols = ['gauge','sample_id','decade']
    data = stream_ts_decade.groupby(groupby_cols)[subset_cols].mean().reset_index()

    #plot for each decade
    fig, ax = plt.subplots(figsize=(fig_width,0.70744 *fig_width))
    sns.lineplot(data=data, x=sort_column, y=para_column, hue='decade',
                      marker='o', linewidth=2, markersize=10, 
                      palette='rocket',
                      errorbar=("pi", 100),
                      err_style='bars',
                      hue_order=data['decade'].sort_values())
    
    _set_title_label(xlabel=sort_label[sort_column],
                     ylabel=yaxis_labels[para_column],
                     xticks=xticks,
                     gauge_ticklabels=gauge_ticklabels,
                     title=f'{para_column} at {stream_name} and decade')
    
    plt.tight_layout()
    fig.savefig(Path(output_dir, f'{stream_name}_{para_column.replace("*","")}_decadal_along_streamlines.png'), dpi=300)
    plt.close()
    
    return None