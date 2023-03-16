"""
This module computes monthly baseflow using the separation method of Kille (1970),
extended and formalized by Demuth 1993
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import linregress

#functiton to dateparse
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')

#%% Functions for our tool
#get monthly minima
def get_monthly_nmq(Q,min_days=15,minimum_years=10):
    """
    

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    min_days : TYPE, optional
        DESCRIPTION. The default is 15.
    minimum_years : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    
    #calculate the data points per month
    data_per_month=Q.groupby(pd.Grouper(freq='M')).count()
    #calculate the nmq
    nmq=Q.groupby(pd.Grouper(freq='M')).min()
    #add dates per month
    nmq['dates_per_month']=data_per_month
    #reduce all with less then minimum days
    nmq=nmq[nmq.dates_per_month>=min_days]
    nmq=nmq.drop(columns=['dates_per_month'])
    #check for minimum length of time series
    if len(nmq.index.year.unique())>=minimum_years:
        return nmq.squeeze().rename('nmq')
    else:
        print('Time Series length is less than the minimum of ',minimum_years, 'years')
        return None

#sort split dataset prior to     
def sort_split_ts(nmq):
    """
    We sort and split the dataset into our two halfes

    Parameters
    ----------
    nmq : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #first we calculate the thresholds
    nmq_median=nmq.median()
    nmq_min=nmq.quantile(0.05)
    
    #we reset the index and sort the dataset
    nmq_sorted=nmq.sort_values().copy()
    nmq_sorted=nmq_sorted.reset_index()
    
    month_indices=dict(zip(nmq_sorted.index,nmq_sorted['Datum']))
    #change to series
    nmq_sorted=nmq_sorted['nmq']
    
    #get smaller half
    nmq_s_half=nmq_sorted[nmq_sorted<=nmq_median]
    
    #delete all below 5% threshold
    nmq_s_half=nmq_s_half[nmq_s_half>=nmq_min]
    
    #we get the second half of data
    nmq_l_half=nmq_sorted[nmq_sorted>nmq_median]    
    
    
    return month_indices,nmq_s_half,nmq_l_half


#do the regression 
def nmq_regression(nmq_s_half,nmq_l_half,label_curve_type=True):
    
    #start with an initial linear regression
    reg_result_init=linregress(nmq_s_half.index,nmq_s_half.values)
    #get the initial r_value
    rvalue_init=reg_result_init.rvalue
    
    #we add larger values one by one until there will be no improvement of correlation
    counter=1
    rvalue=rvalue_init
    while rvalue>=rvalue_init:
        nmq_extended=pd.concat([nmq_s_half,nmq_l_half[0:counter]])
        reg_result=linregress(nmq_extended.index,nmq_extended.values)
        rvalue=reg_result.rvalue
        counter+=1
        if counter>len(nmq_l_half):
            break
    
    # we compute the linearized nmq
    nmq_int=pd.Series(reg_result.intercept+reg_result.slope*np.array(range(0,nmq_l_half.index.max()+1)),name='baseflow').to_frame()
    
    if label_curve_type:
        #we check whether the critical point is in the upper 2/3 of the upper half of data or the intercept is negative
        if counter>len(nmq_l_half)/3 or reg_result.intercept<0:
            
            nmq_int['curve_type']=2
        else:
            nmq_int['curve_type']=1
    else:
        nmq_int['curve_type']=np.nan
    
    return nmq_int


    
#%% the main function
def baseflow_demuth(Q,gauge_name='gauge',reduce_excess_baseflow=True):
    """
    Calculation of baseflow after Demuth(1993) and Kille(1970)

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    
    reduce_excess_baseflow: If this is TRUE, baseflow results larger than the actual discharge will be set equal to discharge
    

    Returns
    -------
    None.

    """
    #get monthy minimum
    nmq= get_monthly_nmq(Q)
    
    #if no data is included we drop it
    if nmq is None:
        print('No Calculation possible for gauge ',gauge_name, ' due to lack of data')
        #just write nans
        baseflow=Q.resample('m').mean()
        baseflow[gauge_name]=np.nan
        baseflow['curve_type']=0
        return baseflow
    
    #sort the data
    month_indices,nmq_s_half,nmq_l_half=sort_split_ts(nmq)
    
    #do the regression
    baseflow=nmq_regression(nmq_s_half,nmq_l_half,label_curve_type=True)
    
    #use the months as index again
    baseflow['Datum']=baseflow.index.map(month_indices)
    baseflow.set_index('Datum',drop=True,inplace=True)
    
    # we finally overwrite if baseflow is larger than the original flow by the nmq
    if reduce_excess_baseflow:
        #this way we also change the infinity ones
        baseflow.loc[~(baseflow.baseflow/nmq<=1),'baseflow']=nmq[~(baseflow.baseflow/nmq<=1)]

    baseflow.rename(columns={'baseflow':gauge_name},inplace=True)
    print('demuth baseflow was calculated for gauge ',gauge_name)
    return baseflow


#%% some testing script
def test_demuth(filepath=os.path.join(os.path.dirname(__file__),'input','example.csv')):
    Q=pd.read_csv(filepath,index_col=0,parse_dates=['Datum'], date_parser=dateparse)
    ts_name=Q.columns[0]
    
    baseflow=baseflow_demuth(Q,gauge_name=ts_name)
    
    return baseflow


