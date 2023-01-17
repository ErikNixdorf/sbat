"""
We extract the recession curves from the hydrograph,
the method is directly implemented based on the descriptions of 
Tomas Carlotto and Pedro Luiz Borges Chaffe (1)
We provide the option to either do the analysis with baseflow data or with discharge itself
Also Inspired by Posavec 2006
Currently two methods (boussinesq and maillet as well as two types of Master recession curve algorithms are supported)
"""


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from datetime import datetime
import numpy as np
dateparse_q = lambda x: datetime.strptime(x, '%Y-%m-%d')
dateparse_p = lambda x: datetime.strptime(x, '%Y%m%d')
from scipy.optimize import curve_fit

def get_rmse(simulations, evaluation):
    """Root Mean Square Error (RMSE).
    :Calculation Details:
        .. math::
           E_{\\text{RMSE}} = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}[e_i-s_i]^2}
        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series.
    """
    rmse_ = np.sqrt(np.mean((evaluation - simulations) ** 2,
                            axis=0, dtype=np.float64))

    return rmse_

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
    #remove starting and ending nan    
    first_idx = Q.first_valid_index()
    last_idx = Q.last_valid_index()
    Q=Q.loc[first_idx:last_idx]
    
    #if there are only nan we do not need the data:
    if Q.isna().sum()==len(Q):
        raise ValueError('No Valid data, return None')
    return Q

#define the regression function
def bousinesq_func(x,Q_0,n):
        return Q_0/np.power((1+n*x),2)
    
def maillet_func(x,Q_0,n):
        return Q_0*np.exp(-n*x)
    
def bousinesq_func_inv(Q,Q_0,n):
    """
    Inverted function to get the time where the value appears

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    Q_0 : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    t=np.divide(np.sqrt(Q_0/Q),n)-np.divide(1,n)
    
    return t


def maillet_func_inv(Q,Q_0,n):
    """
    Inverted function to get the time where the value appears    

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    Q_0 : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    t=np.log(Q_0/Q)/n
    return t
    
    
def fit_boussinesq (t,Q,Q_0,constant_Q_0=True):
    """
    proposes the analytical solution of the nonlinear
    differential flow equation assuming a Depuit–Boussinesq
    aquifer model

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    Q : TYPE
        DESCRIPTION.
    Q_0 : TYPE
        DESCRIPTION.
    constant_Q_0 : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    popt : TYPE
        DESCRIPTION.
    pcov : TYPE
        DESCRIPTION.

    """
    if constant_Q_0:
        popt, pcov = curve_fit(bousinesq_func, t, Q, bounds=([Q_0*0.9999, 0], [Q_0*1.0001, 1]))
    else:
        popt, pcov = curve_fit(bousinesq_func, t, Q, bounds=([0, 0.001], [1000, 1]))
    
    return popt,pcov

def fit_maillet(t,Q,Q_0,constant_Q_0=True):
    """
    assumes a linear relationship between storage
    and flow (𝑆 = 𝑘𝑄) so that the recession curve can be adjusted by
    an exponential model:

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    Q : TYPE
        DESCRIPTION.
    Q_0 : TYPE
        DESCRIPTION.
    constant_Q_0 : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    if constant_Q_0:
        popt, pcov = curve_fit(maillet_func, t, Q, bounds=([Q_0*0.9999, 0], [Q_0*1.0001, 1]))
    else:
        popt, pcov = curve_fit(maillet_func, t, Q, bounds=([0, 0.001], [1000, 1]))
    
    return popt,pcov



def analyse_recession_curves(Q,mrc_algorithm='demuth',
                             recession_algorithm='boussinesq',
                             moving_average_filter_steps=3,
                             minimum_recession_curve_length=10
                             ):
    """
    

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    mrc_algorithm : TYPE, optional
        DESCRIPTION. The default is 'demuth'.
    recession_algorithm : TYPE, optional
        DESCRIPTION. The default is 'boussinesq'.
    moving__average_filter_steps : TYPE, optional
        DESCRIPTION. The default is 3.
    minimum_recession_curve_length : TYPE, optional
        DESCRIPTION. The default is 10.


    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    #%% Clean
    Q=clean_gauge_ts(Q)
    
    #%% Apply filter for rolling mean
    if moving_average_filter_steps>0:
        Q=Q.rolling(int(moving_average_filter_steps)).mean()
    #%% Get numbers for all slopes with the same direction
    #inspired by https://stackoverflow.com/questions/55133427/pandas-splitting-data-frame-based-on-the-slope-of-data
    Q=Q.rename('Q')
    Q=Q.to_frame()
    Q['diff']=Q.diff().fillna(0)
    Q.loc[Q['diff'] < 0, 'diff'] = -1
    Q.loc[Q['diff'] > 0, 'diff'] = 1
    Q['section_id'] = (~(Q['diff'] == Q['diff'].shift(1))).cumsum()
    
    #remove all sections which are ascending, rising limb of hydrograph
    Q=Q[Q['diff']==-1]
    #we check for sections with a minum length
    section_length=Q.groupby('section_id').size()
    Q['section_length']=Q['section_id']
    Q['section_length']=Q['section_length'].replace(section_length)
    #remove all below threshold
    Q=Q[Q['section_length']>=minimum_recession_curve_length]
    
    #replace each section length by ascending numbers (the event length)
    Q['section_time'] = Q.groupby('section_id').cumcount()
    
    #get the largest discharge for each sedgment
    Q0= Q[['Q','section_id']].groupby('section_id').max().squeeze()
    Q['Q0']=Q['section_id'].replace(Q0)
    Q['Q0_inv']=1/Q['Q0']
    

    #sort them in decending style
    #Q_sorted=Q.sort_values(['Q0','section_id'],ascending=False)
    
    if len(Q)==0:
        print('No Recession limb within the dataset')
        return None,None,None,pd.DataFrame(columns=['section_id'])
    
    #%% master Recession Curve, either matching Strip or correlation Method
    if  mrc_algorithm == 'matching_strip':
        
        if recession_algorithm=='boussinesq':
            initDf=True
            
            
            for _,limb in Q.groupby('Q0_inv'):
                #we calculate the fit for the initial recession limb
                if initDf:
                    Q_data=limb['Q'].values
                    Q_0=limb['Q0'].iloc[0]            
                    fit_parameter, pcov=fit_boussinesq(limb['section_time'].values, limb['Q'].values, Q_0)
                    Q_rec_merged=bousinesq_func(limb['section_time'].values,Q_0,fit_parameter[1])
                    df_rec_merged=pd.Series(Q_rec_merged,limb['section_time'].values).rename('Q')
                    
                    initDf=False
                else:
                    
                    #fit the proper location in the already merged part
                    t_shift= bousinesq_func_inv(limb['Q0'].iloc[0],fit_parameter[0],fit_parameter[1])
                    #add t_shift to section time
                    limb['section_time']=limb['section_time']+t_shift
                    #add the limb with shifted time to the extending dataset
                    df_merged=pd.concat([pd.Series(Q_data,df_rec_merged.index).rename('Q'),limb.set_index('section_time')['Q']]).sort_index()
        
                    #compute the recession curve and parameters for the combined ones
                    fit_parameter, pcov=fit_boussinesq(df_merged.index.values, df_merged.values, Q_0)
                    Q_rec_merged=bousinesq_func(df_merged.index.values,Q_0,fit_parameter[1])
                    df_rec_merged=pd.Series(Q_rec_merged,df_merged.index.values).rename('Q')
                    Q_data=np.append(Q_data,limb['Q'].values)
    
            #after we got the final regression line we can calculate some performance
            df_rec_merged=df_rec_merged.to_frame()
            df_rec_merged['Q_data']=Q_data
            r_mrc=df_rec_merged.corr().to_numpy()[0,1]
            
        if recession_algorithm=='maillet':
            initDf=True
            for _,limb in Q.groupby('Q0_inv'):
                #we calculate the fit for the initial recession limb
                if initDf:
                    Q_data=limb['Q'].values
                    Q_0=limb['Q0'].iloc[0]            
                    fit_parameter, pcov=fit_maillet(limb['section_time'].values, limb['Q'].values, Q_0)
                    Q_rec_merged=maillet_func(limb['section_time'].values,Q_0,fit_parameter[1])
                    df_rec_merged=pd.Series(Q_rec_merged,limb['section_time'].values).rename('Q')
                    
                    initDf=False
                else:
                    
                    #fit the proper location in the already merged part
                    t_shift= maillet_func_inv(limb['Q0'].iloc[0],fit_parameter[0],fit_parameter[1])
                    #add t_shift to section time
                    limb['section_time']=limb['section_time']+t_shift
                    #add the limb with shifted time to the extending dataset
                    df_merged=pd.concat([pd.Series(Q_data,df_rec_merged.index).rename('Q'),limb.set_index('section_time')['Q']]).sort_index()
        
                    #compute the recession curve and parameters for the combined ones
                    fit_parameter, pcov=fit_maillet(df_merged.index.values, df_merged.values, Q_0)
                    Q_rec_merged=maillet_func(df_merged.index.values,Q_0,fit_parameter[1])
                    df_rec_merged=pd.Series(Q_rec_merged,df_merged.index.values).rename('Q')
                    Q_data=np.append(Q_data,limb['Q'].values)
    
            #after we got the final regression line we can calculate some performance
            df_rec_merged=df_rec_merged.to_frame()
            df_rec_merged['Q_data']=Q_data
            r_mrc=df_rec_merged.corr().to_numpy()[0,1]
            
    
            
    if mrc_algorithm == 'demuth':
        #According to demuth method we first compute an initial fit for all data
        if recession_algorithm=='boussinesq':
            Q_data=Q['Q'].values
            Q_0=Q['Q0'].mean()            
            fit_parameter, pcov=fit_boussinesq(Q['section_time'].values, Q['Q'].values, Q_0,constant_Q_0=False)
            Q_rec=bousinesq_func(Q['section_time'].values,Q_0,fit_parameter[1])
            r_init=np.corrcoef(Q_data,Q_rec)
            #we replace the first fit parameter with the actual Q_0, moving in upward direction
    
            Q0_max=Q['Q0'].max()
            # Every recession limb will be shifted in t_direction on the new base limp
            df_merged=pd.Series(dtype=float)
            for _,limb in Q.groupby('Q0_inv'):
                t_shift= bousinesq_func_inv(limb['Q0'].iloc[0],Q0_max,fit_parameter[1])
                #add t_shift to section time
                limb['section_time']=limb['section_time']+t_shift
                df_merged=df_merged.append(limb.set_index('section_time')['Q'])
            
            #we compute a new mean fitting model of the shifted time series
            df_merged=df_merged.sort_index()
            fit_parameter, pcov=fit_boussinesq(df_merged.index.values, df_merged.values, Q0_max,constant_Q_0=True)
            
            Q_rec_merged=bousinesq_func(df_merged.index.values,fit_parameter[0],fit_parameter[1])
            r_mrc=np.corrcoef(df_merged.values,Q_rec_merged)[0,1]
            
        if recession_algorithm=='maillet':
            Q_data=Q['Q'].values
            Q_0=Q['Q0'].mean()            
            fit_parameter, pcov=fit_maillet(Q['section_time'].values, Q['Q'].values, Q_0,constant_Q_0=False)
            Q_rec=maillet_func(Q['section_time'].values,Q_0,fit_parameter[1])
            r_init=np.corrcoef(Q_data,Q_rec)
            #we replace the first fit parameter with the actual Q_0, moving in upward direction
    
            Q0_max=Q['Q0'].max()
            # Every recession limb will be shifted in t_direction on the new base limp
            df_merged=pd.Series(dtype=float)
            for _,limb in Q.groupby('Q0_inv'):
                t_shift= maillet_func_inv(limb['Q0'].iloc[0],Q0_max,fit_parameter[1])
                #add t_shift to section time
    
                limb['section_time']=limb['section_time']+t_shift
                df_merged=df_merged.append(limb.set_index('section_time')['Q'])
            
            #we compute a new mean fitting model of the shifted time series
            df_merged=df_merged.sort_index()
            fit_parameter, pcov=fit_maillet(df_merged.index.values, df_merged.values, Q0_max,constant_Q_0=True)
            
            Q_rec_merged=maillet_func(df_merged.index.values,fit_parameter[0],fit_parameter[1])
            r_mrc=np.corrcoef(df_merged.values,Q_rec_merged)[0,1]
    

    print('pearson r of method',mrc_algorithm, 'with recession model',recession_algorithm, ' is ', np.round(r_mrc,2))
    

    return fit_parameter[0],fit_parameter[1],r_mrc,Q



#%% plotting
def plot_recession_results(meta_data=pd.DataFrame(),meta_data_decadal=pd.DataFrame(),
                    parameters_to_plot=['Q0','pearson_r','n'],
                    streams_to_plot=['spree','lausitzer_neisse','schwarze_elster'],
                    output_dir=os.path.join(os.getcwd(),'bf_analysis','figures'),
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
    coef_log_scale={'Q0':True,
                'pearson_r':False,
                'n':True}
    
    #first we generate the output dir
    os.makedirs(output_dir,exist_ok=True)
    
    if not decadal_plots:
    
        
        #next we plot the top 15 gauges with the largest deviations
        if len(meta_data)<15:
            index_max=len(meta_data)
        else:
            index_max=15
        #compute
        para_col='pearson_r'
        fig,ax = plt.subplots()
        sns.barplot(data=meta_data.reset_index().sort_values(para_col,ascending=False)[0:index_max],x=para_col,y='gauge').set(title='Gauges with weakest Performance')
        fig.savefig(os.path.join(output_dir,'Gauges_with_weakest_performance'+'.png'),dpi=300, bbox_inches = "tight")
        plt.close()
        
        #we make lineplots along the river systems
    
        para_cols=parameters_to_plot
        for para_col in para_cols:
            for stream in streams_to_plot:
                
                stream_gauges=meta_data[meta_data.gewaesser==stream].reset_index()
                if len(stream_gauges)==0:
                    print('no gauges along stream',stream)
                    continue
                stream_gauges['river_km']=stream_gauges['km_muendung_hauptfluss_model'].max()-stream_gauges['km_muendung_hauptfluss_model']
                stream_gauges=stream_gauges.sort_values('river_km')
                stream_gauges=stream_gauges[stream_gauges['gauge']!='eisenhuettenstadt']
                if stream=='schwarze_elster':
                    stream_gauges=stream_gauges[stream_gauges['gauge']!='eisenhuettenstadt']
                    gauge_ticklabels=stream_gauges['gauge'].unique().tolist()
                else:
                    gauge_ticklabels=[label.split('_')[0] for label in stream_gauges['gauge'].unique()]            
                
                fig,ax = plt.subplots()
                s6=sns.lineplot(data=stream_gauges,x='river_km',y=para_col,
                                marker='o',linewidth=2,markersize=10,color='dodgerblue')
                #we give an error band if available
                    
                plt.title(para_col+' along stream '+stream)
                plt.ylabel(para_col)
                plt.xlabel('River Kilometer')
                ax.set_xticks(stream_gauges['river_km'].unique())
                plt.xticks(rotation=90)
                ax.set_xticklabels(gauge_ticklabels)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir,para_col+'_'+stream+'.png'),dpi=300)
                plt.close()      
    
    elif decadal_plots:
        print('We finally need the decadal plots')
        
        #loop through data
        for para_col in parameters_to_plot:
            for stream in streams_to_plot:
                
                stream_gauges=meta_data_decadal[meta_data_decadal.gewaesser==stream].reset_index()
                if len(stream_gauges)==0:
                    print('no gauges along stream',stream)  
                    continue
                #https://stackoverflow.com/questions/62004561/is-this-an-error-in-the-seaborn-lineplot-hue-parameter
                stream_gauges['river_km']=stream_gauges['km_muendung_hauptfluss_model'].max()-stream_gauges['km_muendung_hauptfluss_model']
                stream_gauges=stream_gauges.sort_values('river_km')
                stream_gauges=stream_gauges[stream_gauges['gauge']!='eisenhuettenstadt']
                if stream=='schwarze_elster':
                    stream_gauges=stream_gauges[stream_gauges['gauge']!='eisenhuettenstadt']
                    gauge_ticklabels=stream_gauges['gauge'].unique().tolist()
                else:
                    gauge_ticklabels=[label.split('_')[0] for label in stream_gauges['gauge'].unique()]            
                
                fig,ax = plt.subplots()
                s6=sns.lineplot(data=stream_gauges,x='river_km',y=para_col,hue='decade',
                                marker='o',linewidth=2,markersize=10,palette='rocket',
                                hue_order=stream_gauges['decade'].sort_values())
                plt.title(para_col+' along stream '+stream)
                plt.ylabel(para_col)
                plt.xlabel('River Kilometer')
                ax.set_xticks(stream_gauges['river_km'].unique())
                plt.xticks(rotation=90)
                ax.set_xticklabels(gauge_ticklabels)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir,para_col+'_'+stream+'.png'),dpi=300)
                plt.close()     


def test_recession_curve_analysis():
    
    #%% definitions
    moving__average_filter_steps=3 #daily
    minimum_recession_curve_length=10
    mrc_algorithm='matching_strip'
    recession_algorithm='boussinesq'
    Q=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','discharge','example.csv'),
                  index_col=0,parse_dates=['Datum'], 
                  date_parser=dateparse_q,
                  squeeze=True)    
    Q_0,n=analyse_recession_curves(Q,mrc_algorithm=mrc_algorithm,
                                 recession_algorithm=recession_algorithm,
                                 moving__average_filter_steps=moving__average_filter_steps,
                                 minimum_recession_curve_length=minimum_recession_curve_length)


#%% run the test case
#test_recession_curve_analysis()

        

