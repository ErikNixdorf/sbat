"""
This is the central Module which is a class from which the different functions are called
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio

from bflow.bflow import compute_baseflow, add_gauge_stats, plot_bf_results
from recession.recession import analyse_recession_curves, plot_recession_results
from recession.aquifer_parameter import get_hydrogeo_properties
from waterbalance.waterbalance import get_section_water_balance, map_time_dependent_cols_to_gdf


def iterdict(d):
    """
    Recursively iterates over a dictionary and converts any strings
    that can be converted to floats to float, and any lists that can
    be converted to floats to lists of floats.

    Args:
        d (dict): A dictionary to be iterated over.

    Returns:
        dict: The input dictionary with any applicable conversions made.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            iterdict(v)
        elif isinstance(v, list):
            try:
                d[k] = list(map(float, v))
            except ValueError:
                pass
        elif isinstance(v, str):
            try:
                d[k] = float(v)
            except ValueError:
                pass
    return d


# small function to convert to datetime
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')


class Model:
    def __init__(self, conf: dict = None):
        """Initialization method for a new Model instance. Reads configuration, builds the working directory and reads
        the input data.

        Args:

        """



        if 'logger' not in globals():
            global logger
            logger = logging.getLogger('sbat')
            logger.setLevel(logging.INFO)

        self.config = conf
        self.paths: dict = {"root": Path(__file__).parents[1]}

        self.gauge_ts = None
        self.gauge_meta = None

        # todo: Sort these attributes into groups they logically belong to
        self.bf_output = None
        self.gauge_meta_decadal = None
        self.recession_limbs_ts = None
        self.section_basins = None
        self.sections_meta = None
        self.q_diff = None
        self.gdf_network_map = None


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
                                        self.config['file_io']['output']['output_directory'])
        self.paths["output_dir"].mkdir(parents=True, exist_ok=True)

        self.paths["gauge_ts_path"] = Path(self.paths["input_dir"],
                                           self.config['file_io']['input']['gauges']['gauge_time_series'])
        self.paths["gauge_meta_path"] = Path(self.paths["input_dir"],
                                             self.config['file_io']['input']['gauges']['gauge_meta'])

    def _read_data(self):
        self.gauge_ts = pd.read_csv(self.paths["gauge_ts_path"], index_col=0, parse_dates=True)

        # todo: Is this test only for debugging? If yes it should be removed for release. For a public test case better
        #  reduce the input data itself
        if self.config['data_cleaning']['test_mode']:
            # logger.info('test case, focus on three gauges only')
            self.gauge_ts = self.gauge_ts.iloc[:, 0:3]

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

        self.gauge_meta = pd.read_csv(self.paths["gauge_meta_path"], index_col=0)

        if self.config['data_cleaning']['valid_datapairs_only']:
            # reduce the metadata to the gauges for which we have actual time data
            self.gauge_meta = self.gauge_meta.loc[self.gauge_ts.columns]
            # reduce the datasets to all which have metadata
            self.gauge_ts = self.gauge_ts[self.gauge_meta.index.to_list()]
            logger.info(f'{self.gauge_ts.shape[1]} gauges with valid meta data')

        # if we want to compute for each decade we do this here
        # todo: I think there is a better way to solve this
        if self.config['time']['compute_each_decade']:
            logger.info('Statistics for each gauge will be computed for each decade')
            self.gauge_meta_decadal = pd.DataFrame()
    # function which controls the baseflow module

    def get_baseflow(self):

        # first we compute the baseflow
        self.bf_output = compute_baseflow(self.gauge_ts,
                                          self.gauge_meta,
                                          methods=self.config['baseflow']['methods'],
                                          compute_bfi=self.config['baseflow']['compute_baseflow_index'],
                                          calculate_monthly=self.config['baseflow']['calculate_monthly'])

        # second we update the medatadata if required
        if self.config['baseflow']['update_metadata']:
            # get the monthly keys
            monthly_keys = [key for key in self.bf_output.keys() if len(self.bf_output[key]) > 0 and 'monthly' in key]

            if monthly_keys:
                logger.info('Updating metadata with the mean of monthly data')
                gauge_meta_updated = pd.concat([pd.concat([add_gauge_stats(subset.drop(columns=['gauge', 'variable']),
                                                                           self.gauge_meta.loc[gauge, :].to_frame().T,
                                                                           col_name=key,
                                                                           decadal_stats=self.config['time'][
                                                                               'compute_each_decade'],
                                                                           ).reset_index().reset_index().set_index(
                    ['index', 'gauge']) for gauge, subset in self.bf_output[key].groupby('gauge')]) for key in
                    monthly_keys]
                    , axis=1)
            # todo: If monthly_keys is False, gauge_meta_updated will not exist. Does it makes sense to move the next
            #  statement into the prior if-block?
            # drop duplicate columns
            gauge_meta_updated = gauge_meta_updated.loc[:, ~gauge_meta_updated.columns.duplicated()].reset_index().drop(columns='index')

            self.gauge_meta = gauge_meta_updated.groupby('gauge').first()

            if self.config['time']['compute_each_decade']:
                gauge_meta_decadal = gauge_meta_updated.set_index(['gauge', 'decade'])
                if hasattr(self, 'gauge_meta_decadal'):
                    new_cols = list(set(gauge_meta_decadal.columns) - set(self.gauge_meta_decadal.columns))
                    self.gauge_meta_decadal = pd.concat([self.gauge_meta_decadal, gauge_meta_decadal[new_cols]], axis=1)
                else:
                    self.gauge_meta_decadal = gauge_meta_decadal.copy()
                # clean the gauge_meta with no decades
                self.gauge_meta = self.gauge_meta.drop(
                    columns=[col for col in gauge_meta_updated.columns if '_dec' in col] + ['decade'])

        if self.config['file_io']['output']['plot_results']:
            logger.info('plot_results of baseflow computation')
            plot_bf_results(data=self.bf_output, meta_data=self.gauge_meta,
                            meta_data_decadal=self.gauge_meta_decadal,
                            parameters_to_plot=['bf_daily', 'bf_monthly', 'bfi_monthly'],
                            streams_to_plot=['spree', 'lausitzer_neisse', 'schwarze_elster'],
                            output_dir=Path(self.paths["output_dir"], 'bf_analysis', 'figures'),
                            decadal_plots=self.config['time']['compute_each_decade'],
                            )

    # %%function that adds discharge statistics

    def get_discharge_stats(self):
        # call the gauge stats function
        data = self.gauge_ts.copy(deep=True)

        gauge_meta_updated = pd.DataFrame()
        for gauge, subset in data.melt(var_name='gauge', ignore_index=False).groupby('gauge'):
            output = add_gauge_stats(subset.drop(columns=['gauge']),
                                     self.gauge_meta.loc[gauge, :].to_frame().T,
                                     col_name=self.config['discharge']['col_name'],
                                     decadal_stats=self.config['time']['compute_each_decade'])
            gauge_meta_updated = pd.concat([gauge_meta_updated, output])

        # the first                  column is the updated gauge_meta
        self.gauge_meta = gauge_meta_updated.groupby('gauge').first()

        # if decadal stats exist we save them

        if self.config['time']['compute_each_decade']:
            # we we have decadal data we append
            if hasattr(self, 'gauge_meta_decadal'):
                gauge_meta_updated = gauge_meta_updated.reset_index().set_index(['gauge', 'decade'])
                new_cols = list(set(gauge_meta_updated.columns) - set(self.gauge_meta_decadal.columns))
                self.gauge_meta_decadal = pd.concat(
                    [self.gauge_meta_decadal, gauge_meta_updated[new_cols].copy(deep=True)], axis=1)
            else:
                self.gauge_meta_decadal = gauge_meta_updated.set_index(['gauge', 'decade']).copy(deep=True)

            # remove decadal columns
            dec_cols = [col for col in output.columns if '_dec' in col]
            dec_cols.append('decade')
            self.gauge_meta = self.gauge_meta.drop(columns=dec_cols)

        # if we want the monthly stats as well
        if self.config['discharge']['compute_monthly']:
            col_name = 'q_monthly'
            data = self.gauge_ts.copy(deep=True).resample('M').mean()
            gauge_meta_updated = pd.DataFrame()
            for gauge, subset in data.melt(var_name='gauge', ignore_index=False).groupby('gauge'):
                output = add_gauge_stats(subset.drop(columns=['gauge']),
                                         self.gauge_meta.loc[gauge, :].to_frame().T, col_name=col_name,
                                         decadal_stats=False)
                gauge_meta_updated = pd.concat([gauge_meta_updated, output])
            # the first column is the updated gauge_meta
            # todo: Here self.gauge_meta gets overwritten. Is this intended/required? In total I find 12 references for
            #  this attribute, it gets changed a lot. This could have negative side effects. I think it would be more
            #  straightforward to change it at a more central location and maybe create temporary variables instead.
            #  We may need to check how this object change during the whole runtime.
            self.gauge_meta = gauge_meta_updated.groupby('gauge').first()

    # %%the function to call the resession curves
    def get_recession_curve(self):
        """Compute the recession curve for each gauge and decade.
        """

        def add_series_id(df, series_id):
            return df.assign(series_id=series_id)

        logger.info('Started Recession Curve Analysis')

        # first we check whether we want to compute the recession of the water balance or of the hydrograph
        if self.config['recession']['curve_data']['curve_type'] == 'hydrograph':
            logger.info('Recession Analysis is conducted using the hydrograph data')

            # first we check whether baseflow data exist
            if self.config['recession']['curve_data']['flow_type'] == 'baseflow':
                if not hasattr(self, 'bf_output'):
                    raise ValueError('Compute baseflow with function get_baseflow first')
                else:
                    Q = self.bf_output['bf_daily']
                    logger.info('we average the baseflow methods ')
                    Q = Q.reset_index().groupby(['date', 'gauge']).mean(numeric_only=True).reset_index()
                    # wide to long
                    Q = Q.pivot(index='date', columns='gauge', values='value').copy()

            elif self.config['recession']['curve_data']['curve_type'] == 'discharge':
                Q = self.gauge_ts


        elif self.config['recession']['curve_data']['curve_type'] == 'waterbalance':

            logger.info('Recession Analysis is conducted using the waterbalance data')
            # in the case of waterbalance we can not compute a master recession curve due to possibly negative values
            logger.info('mrc_curve not defined for curve_type is waterbalance')
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
                                self.config['recession']['fitting'][
                                    'minimum_recession_curve_length'],
                                maximum_reservoirs=
                                self.config['recession']['fitting'][
                                    'maximum_reservoirs'],
                                minimum_limbs=
                                self.config['recession']['fitting'][
                                    'minimum_limbs']
                                )
                # if data is None we just continue
                if Q_rc is None:
                    logger.warning(f'No Recession curves computable for gauge {gauge} within decade {decade}')
                    continue
                                        
                # we will add data to the metric
                metric = pd.DataFrame(np.expand_dims(mrc_out,0),
                                      columns=['Q0_rec', 'n0_rec', 'pearson_r'],
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
                Q_mrc=pd.DataFrame(data = Q_mrc,
                                   index = range(len(Q_mrc)),
                                   columns=['Q_mrc']
                                   )
                Q_mrc['section_time'] = range(len(Q_mrc))
                Q_mrc['gauge'] = gauge
                Q_mrc['decade'] = decade
                
                Q_mrcs.append(Q_mrc)
                

       #concatenating the data and transfriiing                                 

        self.recession_limbs_ts = pd.concat(recession_limbs, axis=0, sort=False).reset_index(drop = True)

        self.master_recession_curves = pd.concat(Q_mrcs, axis=0).reset_index(drop = True)             

        # append the metrics data to the metadata
        df_metrics = pd.concat(metrics, axis=0)
        if self.config['time']['compute_each_decade']:
            self.gauge_meta_decadal = pd.concat([self.gauge_meta_decadal, df_metrics], axis=1)
        else:

            self.gauge_meta = pd.concat(
                [self.gauge_meta, df_metrics.reset_index().set_index('gauge').drop(columns=['decade'])], axis=1)

        if self.config['file_io']['output']['plot_results']:
            logger.info('plot_results')
            plot_recession_results(meta_data=self.gauge_meta,
                                   meta_data_decadal=self.gauge_meta_decadal,
                                   parameters_to_plot=['Q0_rec', 'pearson_r', 'n0_rec'],
                                   streams_to_plot=['spree', 'lausitzer_neisse', 'schwarze_elster'],
                                   output_dir=Path(self.paths["output_dir"], 'recession_analysis', 'figures'),
                                   decadal_plots=self.config['time']['compute_each_decade'],
                                   )

        logger.info('Recession Curve Analysis Finished')

        # %%we infer the hydrogeological parameters if needed
        if self.config['recession']['fitting']['infer_hydrogeological_parameters']:
            # decide which kind of basins we need
            if self.config['recession']['curve_data']['curve_type'] == 'waterbalance':
                basins = self.section_basins
            elif self.config['recession']['curve_data']['curve_type'] == 'hydrograph':
                basins = gpd.read_file(Path(self.paths["input_dir"],
                                            self.config['file_io']['input']['geospatial']['gauge_basins'])
                                       )
                # we reduce the basins to the gauges for which we have meta information
                basins = basins.loc[basins[self.config['waterbalance']['basin_id_col']].isin(self.gauge_meta.index)]
            else:
                raise ValueError('curve type can either be waterbalance or hydrograph')
            # load the rasterio data
            gw_surface = rasterio.open(Path(self.paths["input_dir"],
                                            self.config['file_io']['input']['hydrogeology']['gw_levels']
                                            )
                                       )

            network_geometry = gpd.read_file(Path(self.paths["input_dir"],
                                                  self.config['file_io']['input']['geospatial'][
                                                      'river_network'])
                                             )

            # get the properties
            if self.config['time']['compute_each_decade']:
                self.gauge_meta_decadal = get_hydrogeo_properties(gauge_data=self.gauge_meta_decadal,
                                                                  basins=basins,
                                                                  basin_id_col=self.config['waterbalance'][
                                                                      'basin_id_col'],
                                                                  gw_surface=gw_surface,
                                                                  network=network_geometry,
                                                                  conceptual_model=self.config['recession']['fitting'][
                                                                      'recession_algorithm'])
            else:
                self.gauge_meta = get_hydrogeo_properties(gauge_data=self.gauge_meta,
                                                          basins=basins,
                                                          basin_id_col=self.config['waterbalance']['basin_id_col'],
                                                          gw_surface=gw_surface,
                                                          network=network_geometry,
                                                          conceptual_model=self.config['recession']['fitting'][
                                                              'recession_algorithm'])

    def get_water_balance(self, **kwargs):
        """Calculate water balance per section"""

        logger.info('We analyse the Water Balance per Section')

        # %% First we load the data

        network_geometry = gpd.read_file(Path(self.paths["input_dir"],
                                              self.config['file_io']['input']['geospatial']['river_network'])
                                         )

        network_connections = pd.read_csv(Path(self.paths["input_dir"],
                                               self.config['file_io']['input']['geospatial'][
                                                   'branches_topology'])
                                          )

        gauge_basins = gpd.read_file(Path(self.paths["input_dir"],
                                          self.config['file_io']['input']['geospatial']['gauge_basins'])
                                     )
        # check whether flow type is given explicitely

        if 'flow_type' in kwargs:
            flow_type = kwargs['flow_type']
        else:
            flow_type = self.config['waterbalance']['flow_type']

        if flow_type == 'baseflow':
            logger.info('Use baseflow time series')
            # check whether the baseflow as already be computed
            if not hasattr(self, 'bf_output'):
                raise ValueError('Calculate Baseflow first before baseflow water balance can be calculated')

            # prove whether explicitely daily values should be calculate otherwise we take monthly
            if self.config['waterbalance']['time_series_analysis_option'] == 'daily' and 'bf_' + \
                    self.config['waterbalance']['time_series_analysis_option'] in self.bf_output.keys():
                data_ts = self.bf_output['bf_daily'].copy()
            else:
                logger.info('Monthly Averaged values are used')
                data_ts = self.bf_output['bf_monthly'].copy()

            # in any case for the baseflow we have to bring the format from long to wide
            logger.info('Average baseflow data for each gauge and time step')

            data_ts = data_ts.groupby(['gauge', 'date']).mean(numeric_only=True).reset_index().pivot(index='date',
                                                                                                     columns='gauge',
                                                                                                     values='value')

        elif flow_type == 'discharge':

            logger.info('Use daily discharge')
            data_ts = self.gauge_ts.copy()

        # start the calculation

        self.sections_meta, self.q_diff, self.gdf_network_map, self.section_basins = get_section_water_balance(
            gauge_data=self.gauge_meta,
            data_ts=data_ts,
            network=network_geometry,
            basins=gauge_basins,
            network_connections=network_connections,
            confidence_acceptance_level=self.config['waterbalance']['confidence_acceptance_level'],
            time_series_analysis_option=self.config['waterbalance']['time_series_analysis_option'],
            basin_id_col=self.config['waterbalance']['basin_id_col'],
        )
        
        #we map the mean_balance information on the geodataframes
        if self.config['time']['compute_each_decade']:
            #we update the meta_data with the decadal average balance
            balance_mean = self.sections_meta.groupby(['downstream_point','decade']).mean(numeric_only=True).loc[:,'balance']
            self.gauge_meta_decadal = pd.concat([self.gauge_meta_decadal,balance_mean],axis=1)
            self.gauge_meta_decadal.index.names=('gauge','decade')

            # map the data from the recession analysis
            self.gdf_network_map=map_time_dependent_cols_to_gdf(self.gdf_network_map,
                                                                self.gauge_meta_decadal,
                                                                geodf_index_col='downstream_point',
                                                                time_dep_df_index_col ='gauge',
                                                                time_dep_df_time_col = 'decade',
                                                                )
            
            self.section_basins=map_time_dependent_cols_to_gdf(self.section_basins, 
                                                               self.gauge_meta_decadal.drop(columns='basin_area'),
                                                               geodf_index_col='basin',
                                                                time_dep_df_index_col ='gauge',
                                                                time_dep_df_time_col = 'decade',
                                                                )
            
        
        elif not self.config['time']['compute_each_decade']:
            #Update the metadata by balance
            balance_mean = self.sections_meta.groupby('downstream_point').mean().loc[:,'balance']
            self.gauge_meta = pd.concat([self.gauge_meta,balance_mean],axis=1)
            
            # metadata added to geodataframes
            self.gdf_network_map = pd.concat([self.gdf_network_map,
                                              self.gauge_meta.reset_index()],
                                             axis=1
                                             )
            #add the information for the basin_area

            self.section_basins = pd.concat([self.section_basins.reset_index(),
                                             self.gauge_meta.reset_index().drop(columns='basin_area')
                                             ],
                                            axis=1)




def main(config_file=None, output=True):
    configuration = Model.read_config(Path(Path(__file__).parents[1], "sbat.yml"))
    sbat = Model(configuration)
    # get discharge data

    sbat.get_discharge_stats()

    # get baseflow
    sbat.get_baseflow()
    # compute the master recession curve

    sbat.get_recession_curve()
    # water balance
    if not hasattr(sbat, 'section_meta'):
        sbat.get_water_balance()
    # write the output
    if output:
        Path(sbat.paths["output_dir"], 'data').mkdir(parents=True, exist_ok=True)
        sbat.sections_meta.to_csv(Path(sbat.paths["output_dir"], 'data', 'section_meta.csv'))
        sbat.q_diff.to_csv(Path(sbat.paths["output_dir"], 'data', 'q_diff.csv'))
        sbat.gdf_network_map.to_file(Path(sbat.paths["output_dir"], 'data', 'section_streamlines.gpkg'), driver='GPKG')
        sbat.section_basins.to_file(Path(sbat.paths["output_dir"], 'data', 'section_subbasins.gpkg'), driver='GPKG')
    
    logging.shutdown()
    return sbat



if __name__ == "__main__":

    if sys.argv == 1:
        cfg_file = sys.argv.pop(1)
        main(config_file=cfg_file)
    else:
        main()
    logging.shutdown()
