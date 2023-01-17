"""
This is the central Module which is a class from which the different functions are called
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from .baseflow.baseflow import compute_baseflow,add_gauge_stats,plot_bf_results
from .recession.recession import analyse_recession_curves,plot_recession_results
from .waterbalance.waterbalance import get_section_water_balance
from datetime import datetime
class model:
    def __init__(self,
                 gauge_time_series=pd.DataFrame(),
                 gauge_network=gpd.GeoDataFrame(),
                 gauge_metadata=pd.DataFrame(),                 
                 output_dir=None,valid_datapairs_only=True,
                 decadal_stats=True,
                 start_date='1890-01-01',
                 end_date='2021-12-31',dropna=True):
        """
        

        Parameters
        ----------
        gauge_time_series : TYPE, optional
            DESCRIPTION. The default is pd.DataFrame().
        gauge_network : TYPE, optional
            DESCRIPTION. The default is gpd.GeoDataFrame().
        gauge_metadata : TYPE, optional
            DESCRIPTION. The default is pd.DataFrame().
        output_dir : TYPE, optional
            DESCRIPTION. The default is None.
        valid_datapairs_only : TYPE, optional
            DESCRIPTION. The default is True.
        decadal_stats : TYPE, optional
            DESCRIPTION. The default is True.
        start_date : TYPE, optional
            DESCRIPTION. The default is '1890-01-01'.
        end_date : TYPE, optional
            DESCRIPTION. The default is '2021-12-31'.
        dropna : TYPE, If True, for all time steps onl
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """

        #first we define a model_path and an output_path    
        self.model_path=os.getcwd()
        if output_dir is None:
            self.output_dir=os.path.join(self.model_path,'output')
        
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
                     plot=True):
        
        
        #first we compute the baseflow
        self.bf_output=compute_baseflow(self.gauge_ts,self.gauge_meta,
                         methods=methods,compute_bfi=compute_bfi)
        
        #second we update the medatadata if required
        if update_metadata:
            gauge_meta_updated=pd.DataFrame()
            print('We update the metadata with the mean of monthly data')
            for key in self.bf_output.keys():
                if len(self.bf_output[key])>0 and 'monthly' in key:
                    #loop trough all gauges
                    output_gauges=pd.DataFrame()
                    for gauge,subset in self.bf_output[key].groupby('gauge'):
                        output=add_gauge_stats(subset.drop(columns=['gauge','variable']),
                                               self.gauge_meta.loc[gauge,:].to_frame().T,
                                               col_name=key,
                                               decadal_stats=self.decadal_stats
                                               )
                        #append
                        output_gauges=pd.concat([output_gauges,output],axis=0)
                    if self.decadal_stats:
                        gauge_meta_updated=pd.concat([gauge_meta_updated,output_gauges.set_index(['gauge','decade'])],axis=1)
                    else:
                        gauge_meta_updated=pd.concat([gauge_meta_updated,output_gauges],axis=1)

                    #remove duplicate columns
                    gauge_meta_updated = gauge_meta_updated.loc[:,~gauge_meta_updated.columns.duplicated()].copy()
            #reset_index
            gauge_meta_updated=gauge_meta_updated.reset_index(drop=False)
            #the first column is the updated gauge_meta
            self.gauge_meta=gauge_meta_updated.groupby('gauge').first()
            
            #if decadal stats exist we save them
            if self.decadal_stats:
                #we we have decadal data we append
                if hasattr(self,'gauge_meta_decadal'):
                    gauge_meta_updated=gauge_meta_updated.set_index(['gauge','decade'])
                    new_cols=set(gauge_meta_updated.columns)-set(self.gauge_meta_decadal.columns)

                    self.gauge_meta_decadal=pd.concat([self.gauge_meta_decadal,gauge_meta_updated[new_cols].copy(deep=True)],axis=1)

                else:                    
                    self.gauge_meta_decadal=gauge_meta_updated.set_index(['gauge','decade']).copy(deep=True)
                
     
                #remove decadal columns
                dec_cols=[col for col in gauge_meta_updated.columns if '_dec' in col]
                dec_cols.append('decade')
                self.gauge_meta=self.gauge_meta.drop(columns=dec_cols)
            
            

        if plot==True:
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
        
        #the first column is the updated gauge_meta
        self.gauge_meta=gauge_meta_updated.groupby('gauge').first()
        
        #if decadal stats exist we save them

        if self.decadal_stats:
            #we we have decadal data we append
            if hasattr(self,'gauge_meta_decadal'):
                gauge_meta_updated=gauge_meta_updated.set_index(['gauge','decade'])
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
    def get_recession_curve(self,curve_type='baseflow',mrc_algorithm='demuth',
                            recession_algorithm='boussinesq',
                            moving_average_filter_steps=3,
                            minimum_recession_curve_length=10,
                            minimum_limbs = 20,
                            plot=True):
        """
        

        Parameters
        ----------
        curve_type : TYPE, ['baseflow','discharge']
            DESCRIPTION. The default is 'baseflow'.
        mrc_algorithm : TYPE, optional
            DESCRIPTION. The default is ['demuth','matching_strip'].
        recession_algorithm : TYPE, ['boussinesq','maillet']
            DESCRIPTION. The default is 'boussinesq'.
        moving__average_filter_steps : TYPE, optional
            DESCRIPTION. The default is 3.
        minimum_recession_curve_length : TYPE, optional
            DESCRIPTION. The default is 10.
        minimum_limbs : The number of minimum recession limbs to have a valid result
            DESCRIPTION. The default is 20
        Returns
        -------
        None.

        """
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
                Q=Q.pivot(index='Datum',columns='gauge',values='value')
        
        if curve_type=='discharge':
            Q=self.gauge_ts
        
        
        #check whether we want to calculate for each decade or not
        if self.decadal_stats==False:
        #we loop trough all gauges to get the recession curve
            for gauge_name in Q:
                Q0,n,r_mrc,recession_limbs=analyse_recession_curves(Q[gauge_name],mrc_algorithm=mrc_algorithm,
                    recession_algorithm=recession_algorithm,
                    moving_average_filter_steps=moving_average_filter_steps,
                    minimum_recession_curve_length=minimum_recession_curve_length
                    )
                
                #add to meta_data
                #get number of limbs
                n_limbs=len(recession_limbs.section_id.unique())
                self.gauge_meta.loc[gauge_name,'no_of_limbs']=n_limbs
                
                #if no_of limbs is below threshold results are not representative
                if n_limbs<minimum_limbs:
                    print('Number of recession limbs for gauge ',gauge_name,' below threshold of ',minimum_limbs)
                    self.gauge_meta.loc[gauge_name,'Q0']=np.nan
                    self.gauge_meta.loc[gauge_name,'n']=np.nan
                    self.gauge_meta.loc[gauge_name,'pearson_r']=np.nan                    
                else:
                    self.gauge_meta.loc[gauge_name,'Q0']=Q0
                    self.gauge_meta.loc[gauge_name,'n']=n
                    self.gauge_meta.loc[gauge_name,'pearson_r']=r_mrc
                
                    #add also data to the limb time series
                    recession_limbs['gauge']=gauge_name
                    recession_limbs['mrc_algorithm']=mrc_algorithm
                    recession_limbs['curve_type']=curve_type
                    recession_limbs['recession_algorithm']=recession_algorithm
                    #merge
                    self.recession_limbs_ts=pd.concat([self.recession_limbs_ts,recession_limbs],axis=0)
            
            print('Recession Curve Analysis Finished')
        
        
        else:
            print('Compute Recession curves for each decade')
            Q['decade']=[x[0:3]+'5' for x in Q.index.strftime('%Y')]
            
            for decade,Q in Q.groupby('decade'):
                #drop all gauges where no data is within the decade
                Q=Q.dropna(axis=1, how='all')
                Q=Q.drop(columns='decade')
                #we loop trough all gauges to get the recession curve
                for gauge_name in Q:

                    Q0,n,r_mrc,recession_limbs=analyse_recession_curves(Q[gauge_name],mrc_algorithm=mrc_algorithm,
                        recession_algorithm=recession_algorithm,
                        moving_average_filter_steps=moving_average_filter_steps,
                        minimum_recession_curve_length=minimum_recession_curve_length
                        )
                    
                    #add to meta_data
                    #get number of limbs
                    n_limbs=len(recession_limbs.section_id.unique())
                    self.gauge_meta_decadal.loc[(gauge_name,decade),'no_of_limbs']=n_limbs
                    
                    if n_limbs<minimum_limbs:
                        print('Number of recession limbs for gauge ',gauge_name,'and decade ',decade,' below threshold of ',minimum_limbs)
                        self.gauge_meta_decadal.loc[(gauge_name,decade),'Q0']=np.nan
                        self.gauge_meta_decadal.loc[(gauge_name,decade),'n']=np.nan
                        self.gauge_meta_decadal.loc[(gauge_name,decade),'pearson_r']=np.nan                    
                    else:
                        self.gauge_meta_decadal.loc[(gauge_name,decade),'Q0']=Q0
                        self.gauge_meta_decadal.loc[(gauge_name,decade),'n']=n
                        self.gauge_meta_decadal.loc[(gauge_name,decade),'pearson_r']=r_mrc
                        #add also data to the limb time series
                        recession_limbs['gauge']=gauge_name
                        recession_limbs['decade']=decade
                        recession_limbs['mrc_algorithm']=mrc_algorithm
                        recession_limbs['curve_type']=curve_type
                        recession_limbs['recession_algorithm']=recession_algorithm
                        #merge
                        self.recession_limbs_ts=pd.concat([self.recession_limbs_ts,recession_limbs],axis=0)
                    
            print('Recession Curve Analysis Finished')
            
        if plot==True:
            print('plot_results')
            plot_recession_results(meta_data=self.gauge_meta,meta_data_decadal=self.gauge_meta_decadal,
                                parameters_to_plot=['Q0','pearson_r','n'],
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

   
        
            
            
        
        
        

        



            
        
        
        
        
        
        
        