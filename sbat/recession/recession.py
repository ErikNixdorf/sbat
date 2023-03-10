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
from scipy.signal import savgol_filter
from copy import copy

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

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
    Cleans the gauge time series from NaNs

    Parameters
    ----------
    Q : pandas.Series
        Gauge time series.

    Returns
    -------
    Q : pandas.Series
        Cleaned gauge time series.

    Raises
    ------
    ValueError
        If there are no valid data.

    """
    #remove starting and ending nan    
    first_idx, last_idx = Q.first_valid_index(), Q.last_valid_index()
    Q=Q.loc[first_idx:last_idx]
    
    #if there are only nan we do not need the data:
    if Q.isna().all():
        raise ValueError('No Valid data, return None')
    return Q

#define the regression function
#https://ngwa.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1745-6584.2002.tb02539.x
def boussinesq_1(x,Q_0,n_0,x_0):
        return Q_0/np.power((1+n_0*(x-x_0)),2)
def boussinesq_2(x,Q_0,n_0,x_0,Q_1,n_1,x_1):
        return Q_0/np.power((1+n_0*(x-x_0)),2)+Q_1/np.power((1+n_1*(x-x_1)),2)
def boussinesq_3(x,Q_0,n_0,x_0,Q_1,n_1,x_1,Q_2,n_2,x_2):
        return Q_0/np.power((1+n_0*(x-x_0)),2)+Q_1/np.power((1+n_1*(x-x_1)),2)+Q_2/np.power((1+n_2*(x-x_2)),2)
    
def maillet_1(x,Q_0,n_0,x_0):
        return Q_0*np.exp(-n_0*(x-x_0))
def maillet_2(x,Q_0,n_0,x_0,Q_1,n_1,x_1):
        return Q_0*np.exp(-n_0*(x-x_0))+Q_1*np.exp(-n_1*(x-x_1))
def maillet_3(x,Q_0,n_0,x_0,Q_1,n_1,x_1,Q_2,n_2,x_2):
        return Q_0*np.exp(-n_0*(x-x_0))+Q_1*np.exp(-n_1*(x-x_1))+Q_2*np.exp(-n_2*(x-x_2))
    
def boussinesq_inv(Q,Q_0,n):
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


def maillet_inv(Q,Q_0,n):
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
    
    
def fit_reservoir_function (t,Q,Q_0,
                            constant_Q_0=True,
                            no_of_partial_sums=3,
                            min_improvement_ratio=1.01,
                            recession_algorithm='boussinesq'):
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
    
    #define the acceptable maximum of partial sums
    max_sums = 3
    
    
    # we first check whether the user requests more than the maximum of three partial sums
    if no_of_partial_sums > max_sums:
        print(f"Maximum of partial sums is {max_sums}, reducing to allowed maximum.")
        no_of_partial_sums = max_sums
        
  
    #next we overwrite Q_0 if it is not constant, this makes only sene if there is one reservoir only
    if constant_Q_0 and no_of_partial_sums == 1:
        Q_0_min = Q_0 * 0.9999
        Q_0_max = Q_0 * 1.0001
    else:
        Q_0_min = 0.000001
        Q_0_max = Q_0 * 1.0001
        
    # Define n_min and n_max
    n_min = 10e-10
    n_max = 10
    
    # Define t0_min and t0_max, which define the time lag of certain reservoir
    t0_min = 0
    t0_max = t.max()+0.0000001
    
    # depending on the number of number of reservoirs and the curve type we do a different fitting
    r_cor_old=0.0001
    output=()
    for reservoirs in range(0,no_of_partial_sums):
        if reservoirs>no_of_partial_sums:
            continue
        # we calculate for each individual case
        model_function=globals()[recession_algorithm+'_'+str(reservoirs+1)]
        
        # Define the initial parameters for curve_fit
        if reservoirs == 0:
            p0 = [Q_0, 0.05, 0]
        elif reservoirs == 1:
            p0 = [Q_0, 0.05, 0, Q_0/2, 0.005, int(t.mean())]
        elif reservoirs == 2:
            p0 = [Q_0, 0.05, 0, Q_0/(1/3), 0.005, int(t.mean()/2), Q_0/(2/3), 0.0005, int(t.mean()*1.5)]
            
            # Define the bounds for curve_fit
        bounds_min = [Q_0_min, n_min, t0_min] * (reservoirs+1)
        bounds_max = [Q_0_max, n_max, t0_max] * (reservoirs+1)
        bounds = (bounds_min, bounds_max)

        # Fit the function
        fit_parameter, pcov = curve_fit(model_function, t, Q, p0=p0, bounds=bounds, maxfev=2000)


        Q_int = model_function(t, *fit_parameter)
        
        #get the correlation
        r_cor=np.corrcoef(Q,Q_int)[0,1]
        
        #if correlation does not improve we refuse to add more reservoirs
        if abs(r_cor/r_cor_old)<min_improvement_ratio:            
            break
        else:
            #we write the output tuple
            output=(fit_parameter,Q_int,r_cor,reservoirs+1)
            r_cor_old=copy(r_cor)
            
    #we return the results:
    return output


def find_recession_limbs(Q,smooth_window_size=15,
                         minimum_recession_curve_length=10,
                         split_at_inflection_points=True):
    """
    Identifies and extracts recession limbs from a time series of streamflow.

    Parameters
    ----------
    Q : pandas.DataFrame or pandas.Series
        A pandas DataFrame or Series object containing a time series of streamflow data, with a DateTimeIndex
        as the index and a column named 'Q' containing the streamflow values.
    smooth_window_size : int, optional
        The size of the window (in number of observations) used for smoothing the streamflow data using a 
        Savitzky-Golay filter. The default value is 15, which corresponds to a smoothing window size of 5% of 
        the length of the streamflow time series.
    minimum_recession_curve_length : int, optional
        The minimum length (in number of observations) of a recession limb. Recession limbs shorter than this 
        value will be discarded. The default value is 10.
    split_at_inflection_points : bool, optional
        If True, the function will attempt to split recession limbs at inflection points, i.e., points where the
        second derivative of the streamflow curve changes sign. The default value is True.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the extracted recession limbs, with the following columns:
        - 'Q': the streamflow values of the recession limb
        - 'section_id': an integer identifier for the recession limb
        - 'section_length': the length (in number of observations) of the recession limb
        - 'section_time': the time (in number of observations) elapsed since the start of the recession limb
        - 'max_flow': the maximum streamflow value within the recession limb

    Raises
    ------
    ValueError
        If the input data contains no recession limbs (i.e., all sections are ascending or constant).

    Notes
    -----
    The function works by first smoothing the streamflow data using a Savitzky-Golay filter, then identifying 
    all sections of the smoothed streamflow curve with a negative slope (i.e., recession limbs). The function 
    then discards any sections shorter than the minimum length and optionally attempts to split the remaining 
    sections at inflection points. Finally, the function computes various statistics for each extracted 
    recession limb, including the maximum flow rate, and returns them as a pandas DataFrame.

    """
    #%% Clean
    Q=clean_gauge_ts(Q)
    Q=Q.rename('Q')
    Q=Q.to_frame()    
    #%% We apply a  savgol filter
    if smooth_window_size>0:
        Q['Q']=savgol_filter(Q.values.flatten(), int(round_up_to_odd(smooth_window_size)), 2)
        

    
    #%% Get numbers for all slopes with the same direction
    #inspired by https://stackoverflow.com/questions/55133427/pandas-splitting-data-frame-based-on-the-slope-of-data

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
    
    if len(Q.index) < minimum_recession_curve_length:
        return Q
    
    
    if split_at_inflection_points:
        #we compute the inflection_points of the dataset
        #https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python
        d2y_dx2=np.gradient(np.gradient(Q['Q'].values.flatten()))
        d2y_dx2=np.nan_to_num(d2y_dx2)
        infls0 = np.where(np.diff(np.sign(d2y_dx2)))[0]
        
        #remove all reflection points which are too close to each other
        infl=[infls0[0]]
        for infl0 in infls0:
            if infl0-infl[-1]>minimum_recession_curve_length:
                infl.append(infl0)
        
        #we add the inflection points to the dataset
        Q=Q.reset_index()
        Q['inflection_point']=False        
        Q.loc[infl,'inflection_point']=True
        
        #we make an very ugly loop which allows us to split sections by their inflection point
        section_id_new=0
        Q_with_inflection=pd.DataFrame()
        # initialize empty list for subsection DataFrames
        subsection_dfs = []
        for sec_id,section in Q.groupby('section_id'):
            
            # get location of inflection points
            inflection_ids = section.index[section['inflection_point'] == True].tolist()
        
            # handle sections with no inflection points
            if len(inflection_ids) == 0:
                df_subsection = section.copy()
                df_subsection['section_id_new'] = section_id_new
                section_id_new += 1
                subsection_dfs.append(df_subsection)
            
            else:
                # handle sections with inflection points
        
                # handle sections with inflection points
                if inflection_ids[0] - section.index[0] < minimum_recession_curve_length:
                    section.loc[inflection_ids[0], 'inflection_point'] = False
                    inflection_ids[0] = section.index[0]
                if section.index[-1] - inflection_ids[-1] < minimum_recession_curve_length:
                    section.loc[inflection_ids[-1], 'inflection_point'] = False
                    inflection_ids[-1] = section.index[-1]
            
                # loop through the row ids
                if len(inflection_ids) > 1:
                    for i in range(len(inflection_ids) - 1):
                        df_subsection = section.loc[inflection_ids[i]:inflection_ids[i+1]]
                        df_subsection['section_id_new'] = section_id_new
                        section_id_new += 1
                        subsection_dfs.append(df_subsection)
                else:
                    df_subsection = section.copy()
                    df_subsection['inflection_point'] = False
                    df_subsection['section_id_new'] = section_id_new
                    section_id_new += 1
                    subsection_dfs.append(df_subsection)
                    
        # concatenate subsection DataFrames into a single DataFrame
        Q_with_inflection = pd.concat(subsection_dfs)
        
        #rebuild the function
        Q=Q_with_inflection.copy(deep=True)
        Q['section_id']=Q['section_id_new']
        Q=Q.set_index('Datum').drop(columns=['section_id_new'])
        section_length=Q.groupby('section_id').size()
        Q['section_length']=Q['section_id']
        Q['section_length']=Q['section_length'].replace(section_length)

    

    
    #replace each section length by ascending numbers (the event length)
    Q['section_time'] = Q.groupby('section_id').cumcount()
    
    #get the largest discharge for each sedgment

    Q0= Q[['Q','section_id']].groupby('section_id').max().to_dict()['Q']
    Q['Q0']=Q['section_id'].replace(Q0)
    Q['Q0_inv']=1/Q['Q0']           

        
    return Q



def analyse_recession_curves(Q,mrc_algorithm: str ='demuth',
                             recession_algorithm: str ='boussinesq',
                             smooth_window_size: int = 3,
                             minimum_recession_curve_length: int = 10,
                             define_falling_limb_intervals: bool = True,
                             maximum_reservoirs: int = 3,
                             minimum_limbs: int = 20,
                             ):
    """
    Analyze recession curves using the specified algorithm and parameters.

    Parameters
    ----------
    Q : pandas.DataFrame or pandas.Series
        A time series of discharge values. If a pandas.Series is provided, it must have a name
        and will be converted to a single-column pandas.DataFrame with the name 'Q'.
    mrc_algorithm : str, optional
        The algorithm used to compute the master recession curve. The default is 'demuth'.
    recession_algorithm : str, optional
        The algorithm used to fit individual recession curves. The default is 'boussinesq'.
    smooth_window_size : int, optional
        The size of the window used to smooth the discharge data. The default is 3.
    minimum_recession_curve_length : int, optional
        The minimum length of a recession curve in the input data. The default is 10.
    define_falling_limb_intervals : bool, optional
        Whether to compute the intervals of the falling limbs prior to fitting. The default is True.
    maximum_reservoirs : int, optional
        The maximum number of reservoirs allowed in each limb of the recession curve. The default is 3.
    minimum_limbs : int, optional
        The minimum number of limbs required for the input data. The default is 20.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the fitted recession curves and associated parameters, or None
        if there are no recession curves within the input data.
    tuple of None or pandas.Series
        A tuple containing the master recession curve fit parameters, the performance of the fit,
        and None (since this functionality is not yet implemented).
    """

    #%% we define output mrc_data, first two are master curve fit para, third is performance
    mrc_out=tuple((None,None,None))
    
    if isinstance(Q,pd.Series):
        Q=Q.rename('Q').to_frame()    
    
    #%% First we check whether we need to compute the section intervals
    if define_falling_limb_intervals==True or 'section_id' not in Q.columns:
        #print('Find the recession parts of the time series prior to fitting')


        Q=find_recession_limbs(Q['Q'],smooth_window_size=smooth_window_size,
                                     minimum_recession_curve_length=minimum_recession_curve_length)
    
    #if there are no falling limbs within the interval we just return the data
    if len(Q)==0 or len(Q['section_id'].unique())<minimum_limbs:
        print('No Recession limb within the dataset')
        Q=None        
        return Q,mrc_out
    
    #%% if mrc_algorithm is zero, we just compute_individual branches
   #%% if we are not interested in a master_recession curve we just calculate single coefficients
    limb_sections_list=list()
    for _,limb in Q.groupby('section_id'):
            #raise ValueError('Implementation for Christoph is Missing')
        fit_parameter,limb_int,r_coef,reservoirs=fit_reservoir_function(limb['section_time'].values, 
                                                                        limb['Q'].values, 
                                                                        limb['Q0'].iloc[0],
                                                                        constant_Q_0=True,
                                                                        recession_algorithm=recession_algorithm,
                                                                        no_of_partial_sums=maximum_reservoirs
                                                                        )            

        
        #add data to the section
        for reservoir in range(reservoirs):
            limb.loc[:, f'section_n_{reservoir}'] = fit_parameter[3*(reservoir+1)-2]
            limb.loc[:, f'section_Q0_{reservoir}'] = fit_parameter[3*(reservoir+1)-3]
            limb.loc[:, f'section_x_{reservoir}'] = fit_parameter[3*(reservoir+1)-1]
        limb.loc[:, 'section_corr'] = r_coef
        limb.loc[:, 'Q_interp'] = limb_int
        #merge sections
        limb_sections_list.append(limb)
    # Concatenate all the groups in the list into a single DataFrame
    limb_sections = pd.concat(limb_sections_list, ignore_index=True).reset_index()
    # Drop the old index column
    limb_sections.drop(columns='index', inplace=True)
    #reset index and overwrite Q
    Q=limb_sections
    
    #%% master Recession Curve, either matching Strip or correlation Method
    #we first find the inversion function to stick the recession curves together
    #define the inversion_function
    inv_func=globals()[recession_algorithm+'_inv']    
       
    
    if  mrc_algorithm == 'matching_strip':
        
        #we first get the order of recession beginning with the highest initial values
        section_order=Q.groupby('section_id')['Q0'].max().sort_values(ascending=False).index.tolist()
        initDf=True
        for section_id in section_order:
            
            limb=Q[Q['section_id']==section_id]
            #we calculate the fit for the initial recession limb
            if initDf:
                Q_data=limb['Q'].values
                Q_0=limb['Q0'].iloc[0]
                fit_parameter,Q_rec_merged,r_coef,_=fit_reservoir_function(limb['section_time'].values, 
                                                                       limb['Q'].values, 
                                                                       limb['Q0'].iloc[0],
                                                                       constant_Q_0=True,
                                                                       no_of_partial_sums=1,
                                                                       recession_algorithm=recession_algorithm)
                df_rec_merged=pd.Series(Q_rec_merged,limb['section_time'].values).rename('Q')
                
                initDf=False
            else:
                
                #fit the proper location in the already merged part
                t_shift= inv_func(limb['Q0'].iloc[0],fit_parameter[0],fit_parameter[1])
                #add t_shift to section time
                limb.loc[:,'section_time']=limb.loc[:,'section_time']+t_shift
                #add the limb with shifted time to the extending dataset
                df_merged=pd.concat([pd.Series(Q_data,df_rec_merged.index).rename('Q'),limb.set_index('section_time')['Q']]).sort_index()

                fit_parameter,Q_rec_merged,r_coef,_=fit_reservoir_function(df_merged.index.values, 
                                                                           df_merged.values, 
                                                                           Q_0,
                                                                           constant_Q_0=True,
                                                                           no_of_partial_sums=1,
                                                                           recession_algorithm=recession_algorithm)

                #compute the recession curve and parameters for the combined ones
                df_rec_merged=pd.Series(Q_rec_merged,df_merged.index.values).rename('Q')
                Q_data=np.append(Q_data,limb['Q'].values)

        #after we got the final regression line we can calculate some performance
        df_rec_merged=df_rec_merged.to_frame()
        df_rec_merged['Q_data']=Q_data
        r_mrc=df_rec_merged.corr().to_numpy()[0,1]
            
        #update the output_data
        mrc_out=((fit_parameter[0],fit_parameter[1],r_mrc))
        print('pearson r of method',mrc_algorithm, 'with recession model',recession_algorithm, ' is ', np.round(r_mrc,2))
                    

    if mrc_algorithm == 'demuth':
        #According to demuth method we first compute an initial fit for all data

        Q_data=Q['Q'].values
        Q_0=Q['Q0'].mean()
        fit_parameter,Q_rec,r_init,_=fit_reservoir_function(Q['section_time'].values, 
                                                               Q_data, 
                                                               Q_0,
                                                               constant_Q_0=False,
                                                               no_of_partial_sums=1,
                                                               min_improvement_ratio=1.05,
                                                               recession_algorithm=recession_algorithm)
        

        #we replace the first fit parameter with the actual Q_0, moving in upward direction
        Q0_max=Q['Q0'].max()
        

        # Every recession limb will be shifted in t_direction on the new base limp
        df_merged=pd.Series(dtype=float)
        for _,limb in Q.groupby('section_id'):
            t_shift= inv_func(limb['Q0'].iloc[0],Q0_max,fit_parameter[1])
            #add t_shift to section time
            limb['section_time']=limb['section_time']+t_shift
            df_merged=df_merged.append(limb.set_index('section_time')['Q'])
        
        #we compute a new mean fitting model of the shifted time series
        df_merged=df_merged.sort_index()

        fit_parameter,Q_rec_merged,r_mrc,_=fit_reservoir_function(df_merged.index.values, 
                                                               df_merged.values, 
                                                               Q0_max,
                                                               constant_Q_0=True,
                                                               no_of_partial_sums=1,
                                                               min_improvement_ratio=1.05,
                                                               recession_algorithm=recession_algorithm) 
   
        #update the output_data
        mrc_out=((fit_parameter[0],fit_parameter[1],r_mrc))
        print(f'pearson r of method {mrc_algorithm} with recession model {recession_algorithm} is {np.round(r_mrc,2)}')
        

    return Q,mrc_out



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

        

