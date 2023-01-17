"""
We compute the recession curve parameters for each gauge
"""



#%% import modules
import os
import pandas as pd
import numpy as np
from recession_curves import analyse_recession_curves
from datetime import datetime
dateparse_q = lambda x: datetime.strptime(x, '%Y-%m-%d')
dateparse_bf =lambda x: datetime.strptime(x, '%d.%m.%Y')


#%% settings
gauge_meta_path=os.path.join(os.path.dirname(__file__),'input','gauges','pegel_uebersicht.csv')
bf_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'_2_Basisabflussberechnung','output','bfs_daily_mean.csv')
q_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'_2_Basisabflussberechnung','input','pegel_ts.csv')

moving__average_filter_steps=3 #daily
minimum_recession_curve_length=10
mrc_algorithm='matching_strip'
recession_algorithm='boussinesq'

curve_type='discharge'

#%%load input data
gauge_meta=pd.read_csv(gauge_meta_path,index_col='index').drop(columns=['Unnamed: 0'])
gauge_meta['Q0']=np.nan
gauge_meta['n']=np.nan
gauge_meta['rmse_recession']=np.nan
gauge_meta['no_of_limbs']=np.nan
#%% loop through different gauge types
gauge_meta_out=pd.DataFrame()
limb_ts=pd.DataFrame()
for curve_type in ['baseflow','discharge']:
    for mrc_algorithm in ['demuth','matching_strip']:
        for recession_algorithm in ['boussinesq','maillet']:
            gauge_meta['curve_type']=curve_type
            gauge_meta['recession_algorithm']=recession_algorithm
            gauge_meta['mrc_algorithm']=mrc_algorithm
            #decide with kind of input data we take
            if curve_type=='discharge':
                Q=pd.read_csv(q_path,index_col=0,parse_dates=['Datum'], date_parser=dateparse_q)
                
            elif curve_type=='baseflow':
                Q=pd.read_csv(bf_path,index_col=0,parse_dates=['Datum'], date_parser=dateparse_q)
                
                
            #%% We loop trough all gauges to get the master_recession_curve_parameters
            for gauge_name in Q:
                print('Compute recession curves for gauge',gauge_name)
                if gauge_name=='hagenwerder_3':
                    Q0=np.nan
                    n=np.nan
                    continue
                #if gauge_name!='zittau_6':
                #    continue
                #run the recession curve algorithm
                Q0,n,r_mrc,recession_limbs=analyse_recession_curves(Q[gauge_name],mrc_algorithm=mrc_algorithm,
                recession_algorithm=recession_algorithm,
                moving__average_filter_steps=moving__average_filter_steps,
                minimum_recession_curve_length=minimum_recession_curve_length
                )
                
                #add to meta_data
                gauge_meta.loc[gauge_meta.pegelname==gauge_name,'Q0']=Q0
                gauge_meta.loc[gauge_meta.pegelname==gauge_name,'n']=n
                gauge_meta.loc[gauge_meta.pegelname==gauge_name,'pearson_r']=r_mrc
                #get number of limbs
                n_limbs=len(recession_limbs.section_id.unique())
                gauge_meta.loc[gauge_meta.pegelname==gauge_name,'no_of_limbs']=n_limbs
                
                #add also data to the limb time series
                recession_limbs['gauge']=gauge_name
                recession_limbs['mrc_algorithm']=mrc_algorithm
                recession_limbs['curve_type']=curve_type
                recession_limbs['recession_algorithm']=recession_algorithm
                #merge
                limb_ts=pd.concat([limb_ts,recession_limbs],axis=0)

            gauge_meta_out=pd.concat([gauge_meta_out,gauge_meta])
#%%
gauge_meta_out.to_csv(os.path.join(os.path.dirname(__file__),'output','pegel_uebersicht.csv'))
limb_ts.to_csv(os.path.join(os.path.dirname(__file__),'output','recession_limbs.csv'))
