"""
This is the central Module which is a class from which the different functions are called
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Optional
import logging
from .baseflow.baseflow import compute_baseflow,add_gauge_stats,plot_bf_results
from .recession.recession import analyse_recession_curves,plot_recession_results
from .waterbalance.waterbalance import get_section_water_balance
from datetime import datetime
class Model:
    def __init__(self,
                 gauge_time_series=pd.DataFrame(),
                 gauge_network=gpd.GeoDataFrame(),
                 gauge_metadata=pd.DataFrame(),                 
                 output_dir: Optional[str] = None,
                 valid_datapairs_only=True,
                 decadal_stats: bool = True,
                 start_date: str = "1990-01-01",
                 end_date: str = "2021-12-31",
                 dropna_axis: Optional[int] = None,
                 ):
        """
        A class for processing gauge time series data.
        
        Parameters
        ----------
        gauge_time_series : pd.DataFrame, optional
            A dataframe containing gauge time series data, by default pd.DataFrame()
        gauge_network : gpd.GeoDataFrame, optional
            A dataframe containing gauge network data, by default gpd.GeoDataFrame()
        gauge_metadata : pd.DataFrame, optional
            A dataframe containing gauge metadata, by default pd.DataFrame()
        output_dir : str, optional
            The path to the output directory, by default None
        valid_datapairs_only : bool, optional
            Whether to only include data pairs that have valid metadata, by default True
        decadal_stats : bool, optional
            Whether to compute decadal statistics, by default True
        start_date : str, optional
            The start date for the time series data, by default '1990-01-01'
        end_date : str, optional
            The end date for the time series data, by default '2021-12-31'
        dropna_axis : int or None, optional
            Whether to drop rows or columns with NaN values, by default None
        
        Returns
        -------
        None
        """

        # Define the model and output paths
        self.model_path=os.getcwd()
        self.output_dir = output_dir or os.path.join(self.model_path, "output")        
        os.makedirs(self.output_dir,exist_ok=True)
        
        # we load the dataframes to our class
        self.gauge_ts=gauge_time_series.copy()
        self.network=gauge_network.copy()
        self.gauge_meta=gauge_metadata.copy()
        self.gauge_meta_decadal=pd.DataFrame()
        self.decadal_stats=decadal_stats
        self.start_date=start_date
        self.end_date=end_date
        
        #we are only interested in meta data for which we have time series information
        
        #remove all nans
        self.gauge_ts=self.gauge_ts.dropna(axis=1,how='all')
        
        #reduce to time steps
        self.gauge_ts=self.gauge_ts.loc[datetime.strptime(start_date,'%Y-%m-%d'):datetime.strptime(end_date,'%Y-%m-%d'),:]
        
        #remove nan values
        if dropna_axis==None:
            print('No Nan Values are removed from time series data prior to computation')
        elif dropna_axis == 1:
            print('Remove Gauges which contain a nan entry')
            self.gauge_ts.dropna(axis=1,how='any').dropna(axis=0,how='any')
        elif dropna_axis == 0:
            print('Remove Gauges which contain a nan entry')
            self.gauge_ts.dropna(axis=0,how='any').dropna(axis=1,how='any')
            
        if len(self.gauge_ts)==0:
            raise ValueError('No data left after drop NA Values, consider to define dropna_axis as None or changing start date and end_date')
            
            
            
        
        
        
        if valid_datapairs_only:
            #reduce the metadata to the gauges for which we have actual time data
            self.gauge_meta=self.gauge_meta.iloc[self.gauge_meta.index.isin(self.gauge_ts.columns),:]
            #reduce the datasets to all which have metadata
            self.gauge_ts=self.gauge_ts[self.gauge_meta.index.to_list()]
            print(self.gauge_ts.shape[1], 'gauges with valid meta data')
        
        
    #function which controls the baseflow module
    
    def get_baseflow(self,methods='all',
                     compute_bfi=True,
                     update_metadata=True,
                     plot=True,
                     calculate_monthly=True):
        
        
        #first we compute the baseflow
        self.bf_output=compute_baseflow(self.gauge_ts,self.gauge_meta,
                         methods=methods,compute_bfi=compute_bfi,
                         calculate_monthly=calculate_monthly)
        
        #second we update the medatadata if required
        if update_metadata:
            #get the monthly keys
            monthly_keys = [key for key in self.bf_output.keys() if len(self.bf_output[key]) > 0 and 'monthly' in key]
            
            if monthly_keys:
                print('Updating metadata with the mean of monthly data')
                gauge_meta_updated=pd.concat([pd.concat([add_gauge_stats(subset.drop(columns=['gauge', 'variable']), 
                                                                self.gauge_meta.loc[gauge, :].to_frame().T, 
                                                                col_name=key, 
                                                                decadal_stats=self.decadal_stats,
                                                                ).reset_index().reset_index().set_index(['index','gauge']) for gauge, subset in self.bf_output[key].groupby('gauge')]) for key in monthly_keys]
                            ,axis=1)
                
            #drop duplicate columns
            gauge_meta_updated=gauge_meta_updated.loc[:,~gauge_meta_updated.columns.duplicated()].reset_index()
            
            self.gauge_meta=gauge_meta_updated.groupby('gauge').first()
            
            if self.decadal_stats:
                gauge_meta_decadal = gauge_meta_updated.set_index(['gauge', 'decade'])
                if hasattr(self, 'gauge_meta_decadal'):
                    new_cols = set(gauge_meta_decadal.columns) - set(self.gauge_meta_decadal.columns)
                    self.gauge_meta_decadal = pd.concat([self.gauge_meta_decadal, gauge_meta_decadal[new_cols]], axis=1)
                else:
                    self.gauge_meta_decadal = gauge_meta_decadal.copy()
                #clean the gauge_meta with no decades
                self.gauge_meta = self.gauge_meta.drop(columns=[col for col in gauge_meta_updated.columns if '_dec' in col] + ['decade'])
            
        if plot:
            print('plot_results')
            plot_bf_results(data=self.bf_output,meta_data=self.gauge_meta,
                            meta_data_decadal=self.gauge_meta_decadal,
                            parameters_to_plot=['bf_daily','bf_monthly','bfi_monthly'],
                            streams_to_plot=['spree','lausitzer_neisse','schwarze_elster'],
                            output_dir=os.path.join(self.output_dir,'bf_analysis','figures'),
                            decadal_plots=self.decadal_stats,
                            )
            
            
        
    #function that adds discharge statistics    

    
    def get_discharge_stats(self,col_name='q_daily',compute_monthly=True):
        #call the gauge stats function
        data=self.gauge_ts.copy(deep=True)
        
        
        gauge_meta_updated=pd.DataFrame()
        for gauge,subset in data.melt(var_name='gauge',ignore_index=False).groupby('gauge'):
            
            output=add_gauge_stats(subset.drop(columns=['gauge']),self.gauge_meta.loc[gauge,:].to_frame().T,col_name=col_name,
                                            decadal_stats=self.decadal_stats)
            gauge_meta_updated=pd.concat([gauge_meta_updated,output])
        
        #the first                  column is the updated gauge_meta
        self.gauge_meta=gauge_meta_updated.groupby('gauge').first()
        
        #if decadal stats exist we save them

        if self.decadal_stats:
            #we we have decadal data we append
            if hasattr(self,'gauge_meta_decadal'):
                gauge_meta_updated=gauge_meta_updated.reset_index().set_index(['gauge','decade'])
                new_cols=set(gauge_meta_updated.columns)-set(self.gauge_meta_decadal.columns)
                self.gauge_meta_decadal=pd.concat([self.gauge_meta_decadal,gauge_meta_updated[new_cols].copy(deep=True)],axis=1)
            else:                    
                self.gauge_meta_decadal=gauge_meta_updated.set_index(['gauge','decade']).copy(deep=True)
            
 
            #remove decadal columns
            dec_cols=[col for col in output.columns if '_dec' in col]
            dec_cols.append('decade')
            self.gauge_meta=self.gauge_meta.drop(columns=dec_cols)
            
        #if we want the monthly stats as well
        if compute_monthly:
            col_name='q_monthly'
            data=self.gauge_ts.copy(deep=True).resample('M').mean()
            gauge_meta_updated=pd.DataFrame()
            for gauge,subset in data.melt(var_name='gauge',ignore_index=False).groupby('gauge'):
                
                output=add_gauge_stats(subset.drop(columns=['gauge']),self.gauge_meta.loc[gauge,:].to_frame().T,col_name=col_name,
                                                decadal_stats=False)
                gauge_meta_updated=pd.concat([gauge_meta_updated,output])
            #the first column is the updated gauge_meta
            self.gauge_meta=gauge_meta_updated.groupby('gauge').first()
            
            
    #the function to call the resession curves
    def get_recession_curve(self,
                            curve_type: str = "baseflow",
                            mrc_algorithm: str = "demuth",
                            recession_algorithm: str = "boussinesq",
                            moving_average_filter_steps: int = 3,
                            minimum_recession_curve_length: int = 10,
                            minimum_limbs: int = 20,
                            maximum_reservoirs: int = 3,
                            plot: bool = True,
                        ):
        """Compute the recession curve for each gauge and decade.
    
        Parameters
        ----------
        curve_type: str, optional (default: "baseflow")
            The type of curve to use, either "baseflow" or "discharge".
        mrc_algorithm: str, optional (default: "demuth")
            The method used to compute the master recession curve.
        recession_algorithm: str, optional (default: "boussinesq")
            The method used to fit the recession curve to the data.
        moving_average_filter_steps: int, optional (default: 3)
            The number of steps used to smooth the data.
        minimum_recession_curve_length: int, optional (default: 10)
            The minimum number of recession points to be considered valid.
        minimum_limbs: int, optional (default: 20)
            The minimum number of limbs required to be considered valid.
        maximum_reservoirs: int, optional (default: 3)
            The maximum number of reservoirs used in the Boussinesq algorithm.
        plot: bool, optional (default: True)
            Whether to plot the results or not.
        """
        def add_series_id(df, series_id):
            return df.assign(series_id=series_id)
        
        print('Start Recession Curve Analysis')
        #first we create a new object where we store the time_series
        self.recession_limbs_ts=pd.DataFrame()
        
        #first we check whether baseflow data exist
        if curve_type=='baseflow':            
            if not hasattr(self, 'bf_output'):
                raise ValueError('Compute baseflow with function get_baseflow first')
            else:
                Q=self.bf_output['bf_daily']
                print('we average the baseflow methods ')
                Q=Q.reset_index().groupby(['Datum','gauge']).mean().reset_index()
                #wide to long
                Q=Q.pivot(index='Datum',columns='gauge',values='value').copy()
        
        if curve_type=='discharge':
            Q=self.gauge_ts
            
        if self.decadal_stats:
            Q['decade']=[x[0:3]+'5' for x in Q.index.strftime('%Y')]
        else:
            Q['decade']=-9999
            
        
        #start the recession
        logging.info('Started Recession Curve Analysis')
        for decade,Q_decade in Q.groupby('decade'):
            #drop all gauges where no data is within the decade
            Q_decade = Q_decade.dropna(axis=1, how='all').drop(columns='decade')
            #we loop trough all gauges to get the recession curve
            recession_results = Q_decade.apply(lambda x: analyse_recession_curves(x,
                                             mrc_algorithm=mrc_algorithm,
                                             recession_algorithm=recession_algorithm,
                                             smooth_window_size=moving_average_filter_steps,
                                             minimum_recession_curve_length=minimum_recession_curve_length,
                                             maximum_reservoirs=maximum_reservoirs,minimum_limbs=minimum_limbs
                                            ), axis=0, result_type='expand')
            #get the results
            Q0_mrc, n_mrc, r_mrc = recession_results.iloc[1,:]
            recession_limbs = recession_results.iloc[0,:]
            
            for i in recession_limbs.index:
                recession_limbs[i]['gauge']=i
            
            if self.decadal_stats:
                self.gauge_meta_decadal.loc[(recession_limbs.index, decade), ['Q0_mrc', 'n_mrc', 'pearson_r']] = [Q0_mrc, n_mrc, r_mrc]
            else:
                self.gauge_meta.loc[recession_limbs.index, ['Q0_mrc', 'n_mrc', 'pearson_r']] = [Q0_mrc, n_mrc, r_mrc]
            
            #we finally create a long version
            recession_limbs=pd.concat(list(recession_limbs),axis=0)            
            recession_limbs['decade']=decade
            recession_limbs['mrc_algorithm'] = mrc_algorithm
            recession_limbs['curve_type'] = curve_type
            recession_limbs['recession_algorithm'] = recession_algorithm
            self.recession_limbs_ts = pd.concat([self.recession_limbs_ts,recession_limbs], axis=1, sort=False)

        #reduce the data for gauge meta
        #self.gauge_meta=self.gauge_meta_decadal.groupby('gauge').first()
        logging.info('Recession Curve Analysis Finished')        

            
        if plot==True:
            print('plot_results')
            plot_recession_results(meta_data=self.gauge_meta,meta_data_decadal=self.gauge_meta_decadal,
                                parameters_to_plot=['Q0_mrc','pearson_r','n'],
                                streams_to_plot=['spree','lausitzer_neisse','schwarze_elster'],
                                output_dir=os.path.join(self.output_dir,'recession_analysis','figures'),
                                decadal_plots=self.decadal_stats,
                                )
            
            
    def get_waterbalance(self,
                         network_geometry=gpd.GeoDataFrame(),
                         tributary_connections=pd.DataFrame(),
                         distributary_connections=pd.DataFrame(),
                         flow_type='baseflow',
                         confidence_acceptance_level=0.05,
                         ts_analysis_option='daily'):
        
        print('We analyse the Water  Balance per Section')
        
        if flow_type=='baseflow':
            print('Use baseflow time series')
            #check whether the baseflow as already be computed
            if not hasattr(self, 'bf_output'):
                raise ValueError('Calculate Baseflow first before baseflow water balance can be calculated')
            
            #prove whether explicitely daily values should be calculate otherwise we take monthly    
            if ts_analysis_option == 'daily' and 'bf'+ts_analysis_option in self.bf_output.keys():
                data_ts=self.bf_output['bf'+ts_analysis_option].copy()
            else:
                print('Monthly Averaged values are used')
                data_ts=self.bf_output['bf_monthly'].copy()
            
            #in any case for the baseflow we have to bring the format from long to wide
            print('Average baseflow data for each gauge and time step')
            data_ts=data_ts.groupby(['gauge','Datum']).mean().reset_index().pivot(index='Datum',columns='gauge',values='value')
            
        elif flow_type =='discharge':
            print('Use daily discharge')
            data_ts=self.gauge_ts.copy()
        
        
        #start the calculation

        
        sections_meta,q_diff,gdf_network_map=get_section_water_balance(gauge_data=self.gauge_meta,
                                  data_ts=data_ts,
                                  network=network_geometry,
                                  tributary_connections=tributary_connections,
                                  distributary_connections=distributary_connections,
                                  confidence_acceptance_level=confidence_acceptance_level,
                                  ts_analyse_option=ts_analysis_option)
        
        return sections_meta,q_diff,gdf_network_map

   
        
            
            
        
        
        

        



            
        
        
        
        
        
        
        