"""
This is the central Module which is a class from which the different functions are called
"""

import logging
from pathlib import Path
import sys
import yaml

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from shapely import Point

from bflow.bflow import compute_baseflow, add_gauge_stats, plot_bf_results
from recession.recession import analyse_recession_curves, plot_recession_results
from recession.aquifer_parameter import get_hydrogeo_properties
from waterbalance.waterbalance import get_section_waterbalance, map_time_dependent_cols_to_gdf

logger = logging.getLogger('sbat')
logger.setLevel(logging.INFO)

# define the logging output
Path(f'{Path(__file__).parents[1]}', 'output').mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f'{Path(__file__).parents[1]}/output/sbat.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class Model:
    def __init__(self, conf: dict, output: bool = True):
        """Initialization method for a new Model instance. Reads configuration, builds the working directory and reads
        the input data.

        Args:
            conf: Dictionary that contains the configurations from sbat.yaml
        """

        self.config = conf
        self.paths: dict = {"root": Path(__file__).parents[1]}
        self.output = output
        if not self.output:
            logging.info(f'Set output to {self.output} results in no plotting')
            self.config['file_io']['output']['plot_results'] = False

        self.gauge_ts = None
        self.gauges_meta = None

        self.bf_output = dict()
        self.recession_limbs_ts = None
        self.section_basin_map = None
        self.sections_meta = None
        self.q_diff = None
        self.network_map = None
        self.master_recession_curves = None

        self._build_wd()
        self._read_data()

    @staticmethod
    def read_config(config_file_path : Path
             ) -> dict:
        """Creates a dictionary out of a YAML file."""
        with open(config_file_path) as c:
            conf = yaml.safe_load(c)
        return conf

    def _build_wd(self):
        """Builds the working directory. Reads paths from configuration files and creates output directories."""

        self.paths["input_dir"] = Path(self.paths["root"],
                                       self.config['file_io']['input']['data_dir'])

        self.paths["output_dir"] = Path(self.paths["root"],
                                        self.config['file_io']['output']['output_directory'],
                                        self.config['info']['model_name'])
        if self.output:
            self.paths["output_dir"].mkdir(parents=True, exist_ok=True)
            Path(self.paths["output_dir"], 'data').mkdir(parents=True, exist_ok=True)

        self.paths["gauge_ts_path"] = Path(self.paths["input_dir"],
                                           self.config['file_io']['input']['gauges']['gauge_time_series'])
        self.paths["gauge_meta_path"] = Path(self.paths["input_dir"],
                                             self.config['file_io']['input']['gauges']['gauge_meta'])

    def _read_data(self):
        self.gauge_ts = pd.read_csv(self.paths["gauge_ts_path"], index_col=0, parse_dates=True)
        # all columns to lower case
        self.gauge_ts.columns = list(map(lambda x:x.lower(),self.gauge_ts.columns))
        # Slice gauge time series for start and end date
        self.gauge_ts = self.gauge_ts.loc[self.config['time']['start_date']:self.config['time']['end_date']]

        # we are only interested in metadata for which we have time series information; remove all nans
        self.gauge_ts = self.gauge_ts.dropna(axis=1, how='all')

        # log non-standard configuration
        if self.config['data_cleaning']['drop_na_axis'] is None:
            logger.info('No Nan Values are removed from time series data prior to computation')

        # todo: I do not really get what the following elifs do. Does the order of the axes make a difference? If no we
        #  we can drop over both axes by passing a tuple `axis=(0, 1)`. More important, NaNs are dropped over axis 1 by
        #  default (see some lines above)
        elif self.config['data_cleaning']['drop_na_axis'] == 1:
            logger.info('Remove Gauges which contain a nan entry')
            self.gauge_ts.dropna(axis=1, how='any').dropna(axis=0, how='any')
        elif self.config['data_cleaning']['drop_na_axis'] == 0:
            logging.info('Remove time steps which contain a nan entry')
            self.gauge_ts.dropna(axis=0, how='any').dropna(axis=1, how='any')

        try:
            self.gauge_ts.iloc[0]
        except IndexError:
            logger.exception('No data left after drop NA Values, consider to define dropna_axis as None or changing '
                             'start date and end_date')


        self.gauges_meta = pd.read_csv(self.paths["gauge_meta_path"], index_col=0)        
        #meta data also to lower case
        self.gauges_meta.index = list(map(lambda x:x.lower(),self.gauges_meta.index))
        self.gauges_meta.index.name = 'gauge'
        if self.config['data_cleaning']['valid_datapairs_only']:
            # reduce the metadata to the gauges for which we have actual time data
            self.gauges_meta = self.gauges_meta.loc[self.gauge_ts.columns]
            # reduce the datasets to all which have metadata
            self.gauge_ts = self.gauge_ts[self.gauges_meta.index]

            logger.info(f'{self.gauge_ts.shape[1]} gauges with valid meta data')
            
        #we add a new column called decade
        
        # if we want to compute for each decade we do this here
        # todo: I think there is a better way to solve this
        if self.config['time']['compute_each_decade']:
            logger.info('Statistics for each gauge will be computed for each decade')
            #get information how many decades with data we have per gauge
            gauge_stats_decade = self.gauge_ts.copy()
            gauge_stats_decade['decade'] = [x[0:3] + '5' for x in gauge_stats_decade.index.strftime('%Y')]
            gauge_stats_decade = gauge_stats_decade.groupby('decade').mean().unstack().dropna()
            # we reorganize the data so that we get all decades with measurements per gauge
            gauge_stats_decade=gauge_stats_decade.reset_index().drop(columns=0)
            decades_per_gauge=gauge_stats_decade.groupby('level_0').size()
            gauge_stats_decade=gauge_stats_decade.set_index('level_0')
            #we extend gauge_meta in order to
            gauge_meta_extend_list=list()
            for i in decades_per_gauge.keys():
                #extend the lines
                gauge_extend=pd.concat([self.gauges_meta.loc[i].to_frame().T] * decades_per_gauge[i])
                #add information on decades
                gauge_stat_decade = gauge_stats_decade.loc[i,:]                
                if isinstance(gauge_stat_decade,pd.Series):
                    gauge_stat_decade = gauge_stat_decade.to_frame().T
                gauge_extend=pd.concat([gauge_extend,gauge_stat_decade],axis=1)
                gauge_meta_extend_list.append(gauge_extend)
            
            #overwrite 
            self.gauges_meta=pd.concat(gauge_meta_extend_list)
            
            self.gauges_meta_decadal = pd.DataFrame()
        else:
            logger.info('Statistics for each gauge will be computed over the entire time series')
            self.gauges_meta.loc[:,'decade']=-9999

  
    # function which controls the baseflow module

    def get_baseflow(self):
        """
        Computes the baseflow for the gauge and updates metadata if required.
        
        Returns:
            None
        
        Example:
            #>>> gauge = Gauge(...)
            #>>> gauge.get_baseflow()
        """
        # first we compute the baseflow
        bfs_monthly = pd.DataFrame()
        bfis_monthly = pd.DataFrame()
        bfs_daily = pd.DataFrame()
        performances_metrics = pd.DataFrame()
        for gauge_name in self.gauge_ts.columns:
            basin_area = None
            
            if 'basin_area' in self.gauges_meta.columns:
                basin_area = self.gauges_meta.loc[gauge_name, 'basin_area']
                #for decadal stats we need to extract basin_area only once
                if isinstance(basin_area,pd.Series):
                    basin_area = basin_area.iloc[0]
                basin_area = None if np.isnan(basin_area) else basin_area


            bf_daily, bf_monthly, bfi_monthly, performance_metrics = compute_baseflow(
                                                self.gauge_ts[gauge_name],
                                                basin_area = basin_area,
                                                methods=self.config['baseflow']['methods'],
                                                compute_bfi=self.config['baseflow']['compute_baseflow_index']
                                                )
            
            bf_daily['gauge'] = gauge_name
            bf_monthly['gauge'] = gauge_name
            bfi_monthly['gauge'] = gauge_name
            performance_metrics = pd.DataFrame(performance_metrics,index=[gauge_name])
            #merge
            bfs_daily=pd.concat([bfs_daily,bf_daily])
            bfs_monthly = pd.concat([bfs_monthly,bf_monthly])
            bfis_monthly = pd.concat([bfis_monthly,bfi_monthly])
            performances_metrics = pd.concat([performances_metrics,performance_metrics])
            #add to output dictionary
            self.bf_output.update({'bf_daily': bfs_daily})
            self.bf_output.update({'bf_monthly': bfs_monthly})
            self.bf_output.update({'bfis_monthly': bfis_monthly})
            
        #compute the statistics
        if self.config['baseflow']['compute_statistics']:
            logger.info('Compute baseflow statistics')            
            
            for bf_key in self.bf_output.keys():
                #organizing the data that calculating the mean per method
                bf_subset = self.bf_output[bf_key].groupby(['gauge','date']).mean(numeric_only=True).reset_index()
                #pivot the data
                bf_subset = bf_subset.pivot(index='date',values='value',columns='gauge')
                #update the metadata
                self.gauges_meta = self.gauges_meta.apply(lambda x:add_gauge_stats(x,bf_subset,
                                                                    col_name=bf_key,
                                                                    ),
                                                          axis=1
                                                          )
            
            #add performance metrics
            self.gauges_meta = self.gauges_meta.merge(performances_metrics, 
                                                      how='left', 
                                                      left_index=True, 
                                                      right_index=True,
                                                      )

        if self.config['file_io']['output']['plot_results']:
            logger.info('plot_results of baseflow computation')
            for bf_parameter in self.bf_output.keys():

                plot_bf_results(ts_data=self.bf_output[bf_parameter], meta_data=self.gauges_meta,
                                parameter_name=bf_parameter,
                                plot_along_streams=True,
                                output_dir=Path(self.paths["output_dir"], 'figures','baseflow')
                                )
                    
        if self.output:
            #the meta data
            self.gauges_meta.to_csv(Path(self.paths["output_dir"], 'data', 'gauges_meta.csv'))
            for key in self.bf_output.keys():
                self.bf_output[key].to_csv(Path(self.paths["output_dir"], 'data',key+'.csv'))






    # %%function that adds discharge statistics

    def get_discharge_stats(self):
        """
        Calculates the daily and monthly discharge statistics for each gauge in the dataset.
        """

        #the daily discharge statistics
        self.gauges_meta = self.gauges_meta.apply(lambda x:add_gauge_stats(x,self.gauge_ts,
                                                            col_name=self.config['discharge']['col_name'],
                                                            ),axis=1)
        
        
        # if we want the monthly stats as well
        if self.config['discharge']['compute_monthly']:
            col_name = 'q_monthly'
            data = self.gauge_ts.copy(deep=True).resample('M').mean()        
            self.gauges_meta = self.gauges_meta.apply(lambda x:add_gauge_stats(x,data,
                                                                        col_name=col_name,
                                                                        ),axis=1)
        if self.output:
        #the meta data
            self.gauges_meta.to_csv(Path(self.paths["output_dir"], 'data', 'gauges_meta.csv'))
            
        if self.config['file_io']['output']['plot_results']:
            logger.info('plot_results daily and monthly results of discharge computation')
            discharge_ts_melt_daily = self.gauge_ts.melt(ignore_index=False,var_name='gauge')
            discharge_ts_melt_daily['variable']='q_daily'
            q_dict={'q_daily':discharge_ts_melt_daily}
            #append monthly if existing
            if self.config['discharge']['compute_monthly']:
                discharge_ts_melt_monthly = discharge_ts_melt_daily.groupby('gauge').resample('M').mean(numeric_only=True).reset_index().set_index('date')
                discharge_ts_melt_monthly['variable']='q_monthly'
                q_dict={'q_monthly':discharge_ts_melt_monthly}
            # we run the plotting algorithm from bf_flow
            for q_parameter in q_dict.keys():
                plot_bf_results(ts_data=q_dict[q_parameter], meta_data=self.gauges_meta,
                                parameter_name=q_parameter,
                                plot_along_streams=True,
                                output_dir=Path(self.paths["output_dir"], 'figures','discharge')
                                )
            
            
    # %%the function to call the resession curves
    def get_recession_curve(self):
        """Compute the recession curve for each gauge and decade."""

        logger.info('Started Recession Curve Analysis')

        # first we check whether we want to compute the recession of the water balance or of the hydrograph
        if self.config['recession']['curve_data']['curve_type'] == 'hydrograph':
            logger.info('Recession Analysis is conducted using the hydrograph data')

            # first we check whether baseflow data exist
            if self.config['recession']['curve_data']['flow_type'] == 'baseflow':
                if not self.bf_output:
                    logger.info('Calculate Baseflow first before baseflow water balance can be calculated')
                    self.get_baseflow()
                #convert data
                Q = self.bf_output['bf_daily']
                logger.info('we average the baseflow methods ')
                Q = Q.reset_index().groupby(['date', 'gauge']).mean(numeric_only=True).reset_index()
                # wide to long
                Q = Q.pivot(index='date', columns='gauge', values='value').copy()

            elif self.config['recession']['curve_data']['flow_type'] == 'discharge':
                Q = self.gauge_ts


        elif self.config['recession']['curve_data']['curve_type'] == 'waterbalance':

            logger.warning('Recession Analysis is conducted using the waterbalance data, which is experimental')
            # in the case of waterbalance we can not compute a master recession curve due to possibly negative values
            logger.warning('mrc_curve not defined for curve_type is waterbalance')
            self.config['recession']['fitting']['mastercurve_algorithm'] = None
            # checking whether the water_balance exist and if the same flow type has been used
            if not hasattr(self, 'sections_meta') or not self.config['recession']['curve_data']['flow_type'] == \
                                                         self.config['waterbalance']['flowtype']:
                logger.info('Water_Balance Model is run first in order to get the correct input data for recession')
                self.get_water_balance(flow_type=self.config['recession']['curve_data']['flow_type'])

                Q = self.sections_meta.pivot(columns='downstream_point', values='balance', index='Date')
                Q.index = pd.to_datetime(Q.index).rename('date')
                Q.columns.name = 'gauge'

        if self.config['time']['compute_each_decade']:
            Q['decade'] = [x[0:3] + '5' for x in Q.index.strftime('%Y')]
        else:
            Q['decade'] = -9999

        # start the recession
        metrics = list()
        recession_limbs = list()
        Q_mrcs = list()
        for decade, Q_decade in Q.groupby('decade'):
            # drop all gauges where no data is within the decade
            Q_decade = Q_decade.dropna(axis=1, how='all').drop(columns='decade')
            # we loop trough all gauges to get the recession curve
            for gauge in Q_decade.columns:
                logger.info(f'compute recession curves for gauge {gauge} within decade {decade}')
                Q_rc, Q_mrc, mrc_out = analyse_recession_curves(Q_decade[gauge],
                                mrc_algorithm=
                                self.config['recession']['fitting'][
                                    'mastercurve_algorithm'],
                                recession_algorithm=
                                self.config['recession']['fitting'][
                                    'recession_algorithm'],
                                smooth_window_size=
                                self.config['recession'][
                                    'curve_data'][
                                    'moving_average_filter_steps'],
                                minimum_recession_curve_length=
                                self.config['recession']['curve_data'][
                                    'minimum_recession_curve_length'],
                                maximum_reservoirs=
                                self.config['recession']['fitting'][
                                    'maximum_reservoirs'],
                                minimum_limbs=
                                self.config['recession']['curve_data'][
                                    'minimum_limbs'],
                                inflection_split=
                                self.config['recession'][
                                    'curve_data'][
                                    'split_at_inflection'],
                                )
                # if data is None we just continue
                if Q_rc is None:
                    logger.warning(f'No Recession curves computable for gauge {gauge} within decade {decade}')
                    continue
                                        
                # we will add data to the metric
                metric = pd.DataFrame(np.expand_dims(mrc_out,0),
                                      columns=['rec_Q0', 'rec_n', 'pearson_r'],
                                      index=[0]
                                      )
                metric['decade'] = decade
                metric['gauge'] = gauge
                metric = metric.reset_index(drop=True).set_index(['gauge', 'decade'])
                metrics.append(metric)
                
                # we will add data to the recession limbs
                Q_rc['gauge'] = gauge
                Q_rc['decade'] = decade
                Q_rc['mrc_algorithm'] = self.config['recession']['fitting']['mastercurve_algorithm']
                Q_rc['flow_type'] = self.config['recession']['curve_data']['flow_type']
                Q_rc['curve_type'] = self.config['recession']['curve_data']['curve_type']
                Q_rc['recession_algorithm'] = self.config['recession']['fitting']['recession_algorithm']

                recession_limbs.append(Q_rc)        

                # convert master recession array to data Series
                Q_mrc=Q_mrc.to_frame()
                Q_mrc['section_time'] = Q_mrc.index.values
                Q_mrc['gauge'] = gauge
                Q_mrc['decade'] = decade
                
                Q_mrcs.append(Q_mrc)
                

       #concatenating the data and transfriiing                                 

        self.recession_limbs_ts = pd.concat(recession_limbs, axis=0, sort=False).reset_index(drop = True)

        self.master_recession_curves = pd.concat(Q_mrcs, axis=0).reset_index(drop = True)

        # append the metrics data to the metadata
        self.gauges_meta.index.name = 'gauge'
        df_metrics = pd.concat(metrics, axis=0)        
        self.gauges_meta = pd.concat([self.gauges_meta.reset_index().set_index(['gauge','decade']), df_metrics], axis=1)
        #rearrange the gauge_meta
        self.gauges_meta = self.gauges_meta.reset_index().set_index('gauge')
        
        if self.config['file_io']['output']['plot_results']:
            logger.info('plot_results')
            plot_recession_results(meta_data = self.gauges_meta,
                                   limb_data = self.recession_limbs_ts,
                                   input_ts = Q,
                                   mrc_curve = self.master_recession_curves,
                                   parameters_to_plot=['rec_Q0', 'rec_n', 'pearson_r'],
                                   output_dir=Path(self.paths["output_dir"], 'figures','recession')
                                   )

        logger.info('Recession Curve Analysis Finished')

        # %%we infer the hydrogeological parameters if needed
        if self.config['recession']['hydrogeo_parameter_estimation']['activate']:
            # decide which kind of basins we need
            if self.config['recession']['curve_data']['curve_type'] == 'waterbalance':
                basins = self.section_basins
            elif self.config['recession']['curve_data']['curve_type'] == 'hydrograph':
                basins = gpd.read_file(Path(self.paths["input_dir"],
                                            self.config['file_io']['input']['geospatial']['gauge_basins'])
                                       )
                
                basins[self.config['waterbalance']['basin_id_col']] = basins[self.config['waterbalance']['basin_id_col']].apply(lambda x: x.lower())
                
                # we reduce the basins to the gauges for which we have meta information
                basins = basins.loc[basins[self.config['waterbalance']['basin_id_col']].isin(self.gauges_meta.index)]
            else:
                raise ValueError('curve type can either be waterbalance or hydrograph')
            # load the rasterio data
            try:
                gw_surface = rasterio.open(Path(self.paths["input_dir"],
                                                self.config['file_io']['input']['hydrogeology']['gw_levels']
                                                )
                                           )
            except Exception as e:
                logger.warning(e)
                logger.warning('As no gw data is provided, we try to enforce the simplify rorabaugh parameter estimation method')
                gw_surface = None
                
                self.config['recession']['hydrogeo_parameter_estimation']['rorabaugh_simplification'] = True
                    
            
            #define the conceptual model
            if self.config['recession']['hydrogeo_parameter_estimation']['rorabaugh_simplification']:
                if self.config['recession']['fitting']['recession_algorithm'].lower() != 'maillet':
                    raise ValueError('Rorabaugh method requires maillet based recession (exponential model), please change set up')
                else:
                    conceptual_model = 'rorabaugh'
            else:                    
                conceptual_model=self.config['recession']['fitting'][
                'recession_algorithm']
            
            logger.info(f'Hydrogeo Parameters will be infered based on the model of {conceptual_model}')
                
                    
                    


            network_geometry = gpd.read_file(Path(self.paths["input_dir"],
                                                  self.config['file_io']['input']['geospatial'][
                                                      'river_network'])
                                             )
            #write lower case
            network_geometry['reach_name'] = network_geometry['reach_name'].apply(lambda x: x.lower())
            # get the properties
            self.gauges_meta = get_hydrogeo_properties(gauge_data=self.gauges_meta,
                                                              basins = basins,
                                                              basin_id_col =self.config['waterbalance'][
                                                                  'basin_id_col'],
                                                              gw_surface = gw_surface,
                                                              network=network_geometry,
                                                              conceptual_model=conceptual_model,
                                                              plot = self.config['file_io']['output']['plot_results'],
                                                              plot_dir = Path(self.paths["output_dir"], 'figures','subsurface_properties'),
                                                              )
        
        if self.output:
            #the meta data
            self.gauges_meta.to_csv(Path(self.paths["output_dir"], 'data', 'gauges_meta.csv'))
            #the result of the recession
            self.master_recession_curves.to_csv(Path(self.paths["output_dir"], 'data', 'master_recession_curves.csv'))
            self.recession_limbs_ts.to_csv(Path(self.paths["output_dir"], 'data', 'recession_limbs_time_series.csv'))


    def get_water_balance(self, **kwargs):
        """Calculate water balance per section"""
        
        logger.info('We analyse the Water Balance per Section')

        # %% First we load the data
        self.gauges_meta.index.name = 'gauge'
        network_geometry = gpd.read_file(Path(self.paths["input_dir"],       
                                              self.config['file_io']['input']['geospatial']['river_network'])
                                         )

        network_geometry['reach_name'] = network_geometry['reach_name'].apply(lambda x: x.lower())
        
        if self.config['file_io']['input']['geospatial']['branches_topology'] is None:
            network_connections = pd.DataFrame(columns=['index',
                                                        'stream',
                                                        'main_stream',
                                                        'type',
                                                        'distance_junction_from_receiving_water_mouth'
                                                        ])
        else:
            network_connections = pd.read_csv(Path(self.paths["input_dir"],
                                                   self.config['file_io']['input']['geospatial'][
                                                       'branches_topology'])
                                              )
            
        #also write to lower case
        for col in ['stream','main_stream']:
            network_connections[col] = network_connections[col].apply(lambda x: x.lower())
        #add basin data
        if self.config['file_io']['input']['geospatial']['gauge_basins'] is not None:
            gauge_basins = gpd.read_file(Path(self.paths["input_dir"],
                                              self.config['file_io']['input']['geospatial']['gauge_basins'])
                                         )
            gauge_basins[self.config['waterbalance']['basin_id_col']] = gauge_basins[self.config['waterbalance']['basin_id_col']].apply(lambda x: x.lower())
            #rewrite to lower case
            gauge_basins[self.config['waterbalance']['basin_id_col']] = gauge_basins[self.config['waterbalance']['basin_id_col']].apply(lambda x: x.lower())
            # check whether flow type is given explicitely
        else:
            gauge_basins=None
            logger.info('No GIS Data for gauged basin geometry provided')

        if 'flow_type' in kwargs:
            flow_type = kwargs['flow_type']
        else:
            flow_type = self.config['waterbalance']['flow_type']

        if flow_type == 'baseflow':
            
            logger.info('Use baseflow time series')
            

            if self.config['waterbalance']['bayesian_updating']['activate']:
                logger.warning('Calculation of baseflow introduces additional uncertainty')

            # check whether the baseflow as already be computed
            if not self.bf_output:
                logger.info('Calculate Baseflow first before baseflow water balance can be calculated')
                self.get_baseflow()

            # prove whether explicitely daily values should be calculate otherwise we take monthly
            if self.config['waterbalance']['time_series_analysis_option'] == 'daily' and 'bf_' + \
                    self.config['waterbalance']['time_series_analysis_option'] in self.bf_output.keys():
                data_ts = self.bf_output['bf_daily'].copy()
            else:
                logger.info('Monthly Averaged values are used')
                data_ts = self.bf_output['bf_monthly'].copy()
                if self.config['waterbalance']['bayesian_updating']['activate']:
                    logger.warning('The provided uncertainty refer to individual measurements not monthly averages which have a reduced random error')

            # in any case for the baseflow we have to bring the format from long to wide
            logger.info('Average baseflow data for each gauge and time step')

            data_ts = data_ts.groupby(['gauge', 'date']).mean(numeric_only=True).reset_index().pivot(index='date',
                                                                                                     columns='gauge',
                                                                                                     values='value')

        elif flow_type == 'discharge':

            logger.info('Use daily discharge')
            data_ts = self.gauge_ts.copy()

        # start the calculation

        self.sections_meta, self.q_diff, self.network_map, self.section_basin_map,ts_stats = get_section_waterbalance(
            gauge_data=self.gauges_meta,
            data_ts=data_ts,
            stream_network=network_geometry,
            basins=gauge_basins,
            network_connections=network_connections,
            confidence_acceptance_level=self.config['waterbalance']['confidence_acceptance_level'],
            time_series_analysis_option=self.config['waterbalance']['time_series_analysis_option'],
            basin_id_col=self.config['waterbalance']['basin_id_col'],
            decadal_stats = self.config['time']['compute_each_decade'],
            bayesian_options=self.config['waterbalance']['bayesian_updating'],
        )
        
        #we map the mean_balance information on the geodataframes
        balance_mean = self.sections_meta.groupby(['downstream_point','decade']).mean(numeric_only=True).loc[:,'balance[m³/s]']
        
        #reorganize self_gauges_meta and add gauges_mean
        self.gauges_meta = self.gauges_meta.reset_index().set_index(['gauge','decade'])
        balance_mean.index.names = self.gauges_meta.index.names        
        self.gauges_meta = pd.concat([self.gauges_meta,balance_mean],axis=1)
        self.gauges_meta = self.gauges_meta.reindex(balance_mean.index, axis=0)

        # map results of analysis to geodataframes
        logger.info('Map statistics on stream network geodata')
        self.network_map=map_time_dependent_cols_to_gdf(self.network_map,
                                                            self.gauges_meta,
                                                            geodf_index_col='downstream_point',
                                                            time_dep_df_index_col ='gauge',
                                                            time_dep_df_time_col = 'decade',
                                                            )
        if self.section_basin_map is not None:
            logger.info('Map statistics on subbasin geodata')
            self.section_basin_map=map_time_dependent_cols_to_gdf(self.section_basin_map, 
                                                               self.gauges_meta.drop(columns='basin_area'),
                                                               geodf_index_col='basin',
                                                                time_dep_df_index_col ='gauge',
                                                                time_dep_df_time_col = 'decade',
                                                                )           
  
        if self.output:
            self.sections_meta.to_csv(Path(self.paths["output_dir"], 'data', 'sections_meta.csv'))
            self.q_diff.to_csv(Path(self.paths["output_dir"], 'data', 'q_diff.csv'))
            self.network_map.to_file(Path(self.paths["output_dir"], 'data', 'sections_streamlines.gpkg'),
                                         driver='GPKG')
            
            if self.section_basin_map is not None:
                self.section_basin_map.to_file(Path(self.paths["output_dir"], 'data', 'sections_subbasin.gpkg'), driver='GPKG')
            #the gauge meta data
            self.gauges_meta.to_csv(Path(self.paths["output_dir"], 'data', 'gauges_meta.csv'))
            gdf_gauge_meta = gpd.GeoDataFrame(data=self.gauges_meta,
                                            geometry=[Point(xy) for xy in zip(self.gauges_meta.easting, self.gauges_meta.northing)],
                                            crs=self.network_map.crs,
                            )
            gdf_gauge_meta.to_file(Path(self.paths["output_dir"], 'data', 'gauges_meta.gpkg'), driver='GPKG')


def main(config_file=None, output=True):
    if config_file:
        configuration = Model.read_config(config_file)
    else:
        configuration = Model.read_config(Path(Path(__file__).parents[1], "data/examples/sbat.yml"))

    sbat = Model(configuration, output)
    # get discharge data
    logger.info(f'discharge statistics activation is set to {sbat.config["discharge"]["activate"]}')
    if sbat.config['discharge']['activate']:
        sbat.get_discharge_stats()
        
    
    # get baseflow        
    logger.info(f'baseflow computation activation is set to {sbat.config["baseflow"]["activate"]}')
    if sbat.config['baseflow']['activate']:
        sbat.get_baseflow()
        
    # do the recession analysis
    logger.info(f'recession computation activation is set to {sbat.config["recession"]["activate"]}')
    if sbat.config['recession']['activate']:
        sbat.get_recession_curve()
        
    
    # water balance
    logger.info(f'water balance computation activation is set to {sbat.config["recession"]["activate"]}')
    if not hasattr(sbat, 'section_meta') and sbat.config['waterbalance']['activate'] :
        sbat.get_water_balance()

    logging.shutdown()
    return sbat



if __name__ == "__main__":


    if sys.argv == 1:
        cfg_file = sys.argv.pop(1)
        main(config_file=cfg_file)
    else:
        main()
    logging.shutdown()
