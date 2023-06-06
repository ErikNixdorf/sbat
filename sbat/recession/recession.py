"""
We extract the recession curves from the hydrograph,
the method is directly implemented based on the descriptions of 
Tomas Carlotto and Pedro Luiz Borges Chaffe (1)
We provide the option to either do the analysis with baseflow data or with discharge itself
Also Inspired by Posavec 2006
Currently two methods (boussinesq and maillet as well as two types of Master recession curve algorithms are supported)
"""

from copy import copy
from datetime import datetime
import logging
from pathlib import Path

from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import seaborn as sns

from .mastercurve import get_master_recession_curve
from bflow.bflow import plot_along_streamlines
recession_logger = logging.getLogger('sbat.recession')

dateparse_q = lambda x: datetime.strptime(x, '%Y-%m-%d')
dateparse_p = lambda x: datetime.strptime(x, '%Y%m%d')


def round_up_to_odd(f: float) -> int:
    """Round up a given float to the nearest odd integer.

    Args:
        f (float): The float number to be rounded up to the nearest odd integer.
    
    Returns:
        int: The nearest odd integer to the input float number after rounding up.
    """
    return np.ceil(f) // 2 * 2 + 1


def get_rmse(simulations: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
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


def clean_gauge_ts(Q: pd.Series) -> pd.Series:
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
    # remove starting and ending nan
    first_idx, last_idx = Q.first_valid_index(), Q.last_valid_index()
    Q = Q.loc[first_idx:last_idx]

    # if there are only nan we do not need the data:
    if Q.isna().all():
        raise ValueError('No Valid data, return None')
    return Q


# define the regression function
# https://ngwa.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1745-6584.2002.tb02539.x
def boussinesq_1(x: float, Q_0: float, n_0: float, x_0: float) -> float:
    """
    Calculate the water discharge using Boussinesq's equation
    for one reservoir    
    """
    return Q_0 / np.power((1 + n_0 * (x - x_0)), 2)


def boussinesq_2(x: float, Q_0: float, n_0: float, 
                 x_0: float, Q_1: float, 
                 n_1: float, x_1: float) -> float:
    """
    Calculate the water discharge using Boussinesq's equation
    for two reservoirs    
    """
    return Q_0 / np.power((1 + n_0 * (x - x_0)), 2) + Q_1 / np.power((1 + n_1 * (x - x_1)), 2)


def boussinesq_3(x: float, Q_0: float, n_0: float, x_0: float, Q_1: float, n_1: float,
                 x_1: float, Q_2: float, n_2: float, x_2: float) -> float:
    """
    Calculate the water discharge using Boussinesq's equation
    for three reservoirs    
    """    
    return Q_0 / np.power((1 + n_0 * (x - x_0)), 2) + Q_1 / np.power((1 + n_1 * (x - x_1)), 2) + Q_2 / np.power(
        (1 + n_2 * (x - x_2)), 2)


def maillet_1(x: float, Q_0: float, n_0: float, x_0: float) -> float:
    """
    Calculate the water discharge using Maillet's equation
    for one reservoir    
    """
    return Q_0 * np.exp(-n_0 * (x - x_0))


def maillet_2(x: float, Q_0: float, n_0: float, 
                 x_0: float, Q_1: float, 
                 n_1: float, x_1: float) -> float:
    """
    Calculate the water discharge using Maillet's equation
    for two reservoir    
    """
    return Q_0 * np.exp(-n_0 * (x - x_0)) + Q_1 * np.exp(-n_1 * (x - x_1))


def maillet_3(x: float, Q_0: float, n_0: float, x_0: float, Q_1: float, n_1: float,
                 x_1: float, Q_2: float, n_2: float, x_2: float) -> float:
    """
    Calculate the water discharge using Maillet's equation
    for three reservoirs   
    """
    return Q_0 * np.exp(-n_0 * (x - x_0)) + Q_1 * np.exp(-n_1 * (x - x_1)) + Q_2 * np.exp(-n_2 * (x - x_2))



def boussinesq_inv(Q: float, Q_0: float, n: float) -> float:
    """
    Inverted function to get the time where the value appears

    This function is used in the context of hydraulics to calculate the time
    when a certain value of water discharge from the basin appears in a channel, based on
    Boussinesq's equation.

    Args:
        Q (float): The value of water discharge for which to calculate the time.
        Q_0 (float): The reference value of water discharge used in Boussinesq's equation.
        n (float): The reservoir storage coefficient used in Boussinesq's equation.

    Returns:
        float: The time when the specified value of water discharge appears in the channel.
    """
    
    t = np.divide(np.sqrt(Q_0 / Q), n) - np.divide(1, n)

    return t


def maillet_inv(Q: float, Q_0: float, n: float) -> float:
    """
    Inverted function to get the time where the value appears

    This function is used in the context of hydraulics to calculate the time
    when a certain value of water discharge from the basin appears in a channel, based on
    Maillet's equation.

    Args:
        Q (float): The value of water discharge for which to calculate the time.
        Q_0 (float): The reference value of water discharge used in Boussinesq's equation.
        n (float): The reservoir storage coefficient used in Boussinesq's equation.

    Returns:
        float: The time when the specified value of water discharge appears in the channel.
    """
    
    t = np.log(Q_0 / Q) / n
    return t


def fit_reservoir_function(t: np.ndarray, Q: np.ndarray, Q_0: float,
                           constant_Q_0: bool = True,
                           no_of_partial_sums: int = 3,
                           min_improvement_ratio: float = 1.01,
                           recession_algorithm: str = 'boussinesq',
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    proposes the analytical solution of the nonlinear
    differential flow equation assuming a Depuit–Boussinesq
    aquifer model

    Parameters
    ----------
    t (numpy.ndarray): Array of time steps
    Q (numpy.ndarray): Array of observed discharge values at each time step
    Q_0 (float): A float value representing the initial discharge
    constant_Q_0 (bool, optional): A boolean indicating whether Q_0 is constant. 
        Defaults to True.
    no_of_partial_sums (int, optional): An integer representing the number of reservoirs to use. 
        Defaults to 3.
    min_improvement_ratio (float, optional): A float value representing the minimum ratio of improvement in R value 
        between two consecutive iterations. Defaults to 1.01.
    recession_algorithm (str, optional): A string representing the recession algorithm to use. 
        Defaults to 'boussinesq'.


    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy ndarrays representing 
        the optimized parameters for the model and the covariance matrix.
    """

    # define the acceptable maximum of partial sums
    max_sums = 3

    # we first check whether the user requests more than the maximum of three partial sums
    if no_of_partial_sums > max_sums:
        recession_logger.warning(f"Maximum of partial sums is {max_sums}, reducing to allowed maximum.")
        no_of_partial_sums = max_sums

    # next we overwrite Q_0 if it is not constant, this makes only sene if there is one reservoir only
    if constant_Q_0 and no_of_partial_sums == 1:
        Q_0_min = Q_0 * 0.9999
        Q_0_max = Q_0 * 1.0001
    else:
        if Q_0 > 0:
            Q_0_min = 0.000001
            Q_0_max = Q_0 * 1.0001
        elif Q_0 < 0:
            Q_0_min = min(Q) * 1.0001
            Q_0_max = max(Q) * 0.9999
        else:
            raise Warning('Q_0 equals zero is not covered, return none')
            return None

    # Define n_min and n_max
    n_min = 10e-10
    n_max = 10

    # Define t0_min and t0_max, which define the time lag of certain reservoir
    t0_min = 0
    if no_of_partial_sums == 1:
        t0_max=0.000000000001
    else:
        t0_max = t.max() + 0.0000001

    # depending on the number of number of reservoirs and the curve type we do a different fitting
    r_cor_old = 0.0001
    output = ()
    for reservoirs in range(0, no_of_partial_sums):
        if reservoirs > no_of_partial_sums:
            continue
        # we calculate for each individual case
        model_function = globals()[recession_algorithm + '_' + str(reservoirs + 1)]

        # Define the initial parameters for curve_fit
        if reservoirs == 0:
            p0 = [Q_0, 0.05, 0]
        elif reservoirs == 1:
            p0 = [Q_0, 0.05, 0, Q_0_min+(1/2)*(Q_0_max-Q_0_min), 0.005, int(t.mean())]
        elif reservoirs == 2:
            p0 = [Q_0, 0.05, 0, Q_0_min+(1/3)*(Q_0_max-Q_0_min), 0.005, int(t.mean()/2), Q_0_min+(2/3)*(Q_0_max-Q_0_min), 0.0005, int(t.mean()*1.5)]
    
            # Define the bounds for curve_fit
        bounds_min = [Q_0_min, n_min, t0_min] * (reservoirs + 1)
        bounds_max = [Q_0_max, n_max, t0_max] * (reservoirs + 1)
        bounds = (bounds_min, bounds_max)

        # Fit the function

        fit_parameter, pcov = curve_fit(model_function, t, Q, p0=p0, bounds=bounds, maxfev=2000)

        Q_int = model_function(t, *fit_parameter)

        # get the correlation
        r_cor = np.corrcoef(Q, Q_int)[0, 1]

        # if correlation does not improve we refuse to add more reservoirs
        if abs(r_cor / r_cor_old) < min_improvement_ratio:
            break
        else:
            # we write the output tuple
            output = (fit_parameter, pd.Series(index=t,data=Q_int,name='q_rec'), r_cor, reservoirs + 1)
            r_cor_old = copy(r_cor)

    # we return the results:
    return output


def find_recession_limbs(Q: Union[pd.DataFrame, pd.Series], 
                         smooth_window_size: int = 15,
                         minimum_recession_curve_length: int = 10,
                         split_at_inflection_points: bool = False) -> pd.DataFrame:
    
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
    Q = clean_gauge_ts(Q)
    Q = Q.rename('Q')
    Q = Q.to_frame()    
    
    #%% if length is below window_size we return None
    if len(Q) <= smooth_window_size:
        Q = None
        return Q
    
    #%% We apply a  savgol filter
        
    if smooth_window_size > 0:
        #first we check where are the nans in the data and mask it
        nan_mask = Q['Q'].isna().values
        #apply filter on interpolated data
        Q['Q'] = savgol_filter(Q.interpolate().values.flatten(), int(round_up_to_odd(smooth_window_size)), 2)        
        #use the mask again to get the nan back
        Q.loc[nan_mask,'Q'] = np.nan
        

    
    #%% Get numbers for all slopes with the same direction
    #inspired by https://stackoverflow.com/questions/55133427/pandas-splitting-data-frame-based-on-the-slope-of-data

    Q['diff'] = Q.diff().fillna(0)
    Q.loc[Q['diff'] < 0, 'diff'] = -1
    Q.loc[Q['diff'] > 0, 'diff'] = 1

    Q['section_id'] = (~(Q['diff'] == Q['diff'].shift(1))).cumsum()

    # remove all sections which are ascending, rising limb of hydrograph
    Q = Q[Q['diff'] == -1]
    # we check for sections with a minum length
    section_length = Q.groupby('section_id').size()
    Q['section_length'] = Q['section_id']
    Q['section_length'] = Q['section_length'].replace(section_length)
    # remove all below threshold
    Q = Q[Q['section_length'] >= minimum_recession_curve_length]

    if len(Q.index) < minimum_recession_curve_length:
        return Q

    if split_at_inflection_points:
        # we compute the inflection_points of the dataset
        # https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python
        d2y_dx2 = np.gradient(np.gradient(Q['Q'].values.flatten()))
        d2y_dx2 = np.nan_to_num(d2y_dx2)
        infls0 = np.where(np.diff(np.sign(d2y_dx2)))[0]

        # remove all reflection points which are too close to each other
        infl = [infls0[0]]
        for infl0 in infls0:
            if infl0 - infl[-1] > minimum_recession_curve_length:
                infl.append(infl0)

        # we add the inflection points to the dataset
        Q = Q.reset_index()
        Q['inflection_point'] = False
        Q.loc[infl, 'inflection_point'] = True

        # we make an very ugly loop which allows us to split sections by their inflection point
        section_id_new = 0
        Q_with_inflection = pd.DataFrame()
        # initialize empty list for subsection DataFrames
        subsection_dfs = []
        for sec_id, section in Q.groupby('section_id'):

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
                        df_subsection = section.loc[inflection_ids[i]:inflection_ids[i + 1]].copy()
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

        # rebuild the function
        Q = Q_with_inflection.copy(deep=True)
        Q['section_id'] = Q['section_id_new']
        Q = Q.set_index('date').drop(columns=['section_id_new'])
        section_length = Q.groupby('section_id').size()
        Q['section_length'] = Q['section_id']
        Q['section_length'] = Q['section_length'].replace(section_length)

    # replace each section length by ascending numbers (the event length)
    Q['section_time'] = Q.groupby('section_id').cumcount()

    # get the largest discharge for each sedgment

    Q0 = Q[['Q', 'section_id']].groupby('section_id').max().to_dict()['Q']
    Q['Q0'] = Q['section_id'].replace(Q0)
    Q['Q0_inv'] = 1 / Q['Q0']

    return Q


def analyse_recession_curves(Q, mrc_algorithm: str = 'demuth',
                             recession_algorithm: str = 'boussinesq',
                             smooth_window_size: int = 3,
                             minimum_recession_curve_length: int = 10,
                             define_falling_limb_intervals: bool = True,
                             maximum_reservoirs: int = 3,
                             minimum_limbs: int = 20,
                             inflection_split: bool = False,
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

    # %% we define output mrc_data, first two are master curve fit para, third is performance
    gauge_name = Q.name
    recession_logger.info(f'Analyse recession curves for gauge {gauge_name}')
    mrc_out = tuple((None, None, None))

    if isinstance(Q, pd.Series):
        Q = Q.rename('Q').to_frame()

        # %% First we check whether we need to compute the section intervals
    if define_falling_limb_intervals == True or 'section_id' not in Q.columns:
        # recession_logger.info('Find the recession parts of the time series prior to fitting')

        Q = find_recession_limbs(Q['Q'], smooth_window_size=smooth_window_size,
                                 minimum_recession_curve_length=minimum_recession_curve_length,
                                 split_at_inflection_points = inflection_split)

    #if there are no falling limbs within the interval we just return the data
    if Q is None:
        Q_mrc = None
        recession_logger.info(f'No falling limbs within constraints for gauge {gauge_name}')
        return Q, Q_mrc, mrc_out
    elif len(Q) == 0 or len(Q['section_id'].unique()) < minimum_limbs:
        recession_logger.info(f'less than {minimum_limbs} falling limbs within constraints for gauge {gauge_name}')
        Q = None
        Q_mrc = None        
        return Q, Q_mrc, mrc_out

    # %% if mrc_algorithm is zero, we just compute_individual branches
    # %% if we are not interested in a master_recession curve we just calculate single coefficients
    limb_sections_list = list()
    for _, limb in Q.groupby('section_id'):
        # raise ValueError('Implementation for Christoph is Missing')
        fit_parameter, limb_int, r_coef, reservoirs = fit_reservoir_function(limb['section_time'].values,
                                                                             limb['Q'].values,
                                                                             limb['Q0'].iloc[0],
                                                                             constant_Q_0=True,
                                                                             recession_algorithm=recession_algorithm,
                                                                             no_of_partial_sums=maximum_reservoirs
                                                                             )

        # add data to the section
        for reservoir in range(reservoirs):
            limb.loc[:, f'rec_n_{reservoir}'] = fit_parameter[3 * (reservoir + 1) - 2]
            limb.loc[:, f'rec_Q0_{reservoir}'] = fit_parameter[3 * (reservoir + 1) - 3]
            if maximum_reservoirs > 1:
                limb.loc[:, f'section_x_{reservoir}'] = fit_parameter[3 * (reservoir + 1) - 1]
        limb.loc[:, 'pearson_r'] = r_coef
        limb.loc[:, 'Q_interp'] = limb_int.values
        #we store the year and doy as an additional column
        limb['date']=limb.index.values
        limb['year'] = limb.index.year
        limb['doy'] = limb.index.day_of_year
        # merge sections
        limb_sections_list.append(limb)
    # Concatenate all the groups in the list into a single DataFrame
    limb_sections = pd.concat(limb_sections_list, ignore_index=True).reset_index()
    # Drop the old index column
    limb_sections.drop(columns='index', inplace=True)
    # reset index and overwrite Q
    Q = limb_sections

    # %% master Recession Curve,
    if mrc_algorithm is None:
        recession_logger.warning('No Master Curve Recession Algorithm defined')        
        Q_mrc = None        
        return Q, Q_mrc, mrc_out
    
    #define the hyperparameters
    mrc_hyperparameters={'recession_algorithm' : recession_algorithm,
                             'inv_func' : globals()[recession_algorithm + '_inv'],
                             'fit_reservoir_function' : fit_reservoir_function,
                             'maximum_reservoirs' : maximum_reservoirs,
                             }
    
    
    Q_mrc , mrc_out = get_master_recession_curve(mrc_algorithm,
                                                 Q, 
                                                 mrc_hyperparameters,
                                                 )   
       

    recession_logger.info(f'pearson r of method {mrc_algorithm} with recession model {recession_algorithm} is {np.round(mrc_out[-1], 2)}')


    return Q, Q_mrc, mrc_out

# %% plotting
def plot_recession_results(meta_data: pd.DataFrame, 
                           limb_data: pd.DataFrame, 
                           input_ts: pd.DataFrame,
                           mrc_curve: pd.DataFrame, 
                           parameters_to_plot: list[str] = ['Q0', 'pearson_r', 'n'],
                           output_dir: Path = Path(Path.cwd(), 'bf_analysis', 'figures')
                           )-> None:
    """
    Plot the results of the baseflow calculation.

    Parameters
    ----------
    meta_data : pandas.DataFrame
        Metadata for each gauge station, with columns 'gauge', 'lat', 'lon', 'stream', 'distance_to_mouth', and 'altitude'.
    limb_data : pandas.DataFrame
        Dataframe containing recession limb parameters for each gauge station and limb, with columns 'gauge', 'section_id',
        'decade', 'Q0', 'pearson_r', 'n', 'a', 'b', 'k', and 'Q_interp'.
    input_ts : pandas.DataFrame
        Time series of water flow for each gauge station, with columns representing dates and rows representing water flow values.
    mrc_curve : pandas.DataFrame
        Master recession curve for each gauge station, with columns 'gauge', 'section_id', 'decade', 'section_time', and 'q_rec'.
    parameters_to_plot : list of str, optional
        List of the names of the parameters to plot along the streamlines. Default is ['Q0', 'pearson_r', 'n'].
    output_dir : pathlib.Path, optional
        Output directory to save the generated figures. Default is 'bf_analysis/figures' in the current working directory.

    Returns
    -------
    None
    """
    #set up
    # first we generate the output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    #default seaborn setting
    sns.set_context('paper')
    
    #%% lets plot the parameters along the streamline
    recession_logger.info('Plotting the mrc recession parameters along the streamline')
    for stream,stream_gauges in meta_data.reset_index().groupby('stream'):        
        #get river km
        stream_gauges['river_km'] = stream_gauges['distance_to_mouth'].max() - stream_gauges[
            'distance_to_mouth']
        stream_gauges = stream_gauges.sort_values('river_km')
        gauge_ticklabels = [label.split('_')[0] for label in stream_gauges['gauge'].unique()]        
        #plot for each parameter
        for para_col in parameters_to_plot:
            plot_along_streamlines(stream_gauges = stream_gauges,
                                       stream_name = stream+'_mrc_',
                                       sort_column = 'river_km',
                                       para_column = para_col,
                                       gauge_ticklabels = gauge_ticklabels,
                                       output_dir = output_dir)
    
    

    #%% We provide a boxplot to get the individual limbs
    recession_logger.info('Plotting the recession parameters of the individual limbs')
    for gauge_name,subset in limb_data.groupby('gauge'):
        for parameter_to_plot in parameters_to_plot:
            #check whether we have more than 1, e.g by using multiple reservoirs
            para_cols =[col for col in subset.columns if parameter_to_plot in col]
            if len(para_cols)==0:                
                continue
            else:
                for parameter_name in para_cols:                    
                    fig, ax = plt.subplots()
                    sns.boxplot(data=subset, x='gauge', y=parameter_name)
                    plt.xticks(rotation=90)
                    plt.xlabel('gauge')
                    plt.ylabel(parameter_name)
                    plt.title(f'{parameter_name} boxplot at {gauge_name}')
                    plt.tight_layout()
                    fig.savefig(Path(output_dir, f'{gauge_name}_boxplot_{parameter_name}.png'), dpi=300)
                    plt.close()
                    #across all decades
                    fig, ax = plt.subplots()
                    sns.boxplot(data=subset.reset_index(), x=parameter_name,y='decade')
                    plt.title(f'{parameter_name} decade boxplot at {gauge_name}')
                    plt.tight_layout()
                    fig.savefig(Path(output_dir, f'{gauge_name}_decade_boxplot_{parameter_name}.png'), dpi=300)
                    plt.close()
    #%% plot the time series and the location of the limbs
    recession_logger.info('Plotting the time series of input data and recession limbs')
    if 'decade' in input_ts.columns:
        input_ts=input_ts.drop(columns=['decade'])
    
    for gauge_name,limb_subset in limb_data.groupby('gauge'):
        input_ts_subset=input_ts.loc[:,gauge_name]
        p1=input_ts_subset.plot(linewidth=2)
        
        for grouper,section in limb_subset.groupby(['section_id','decade']):
            section=section.set_index('date')['Q_interp']
            section.plot(ax=p1.axes,linestyle='--',color='k',linewidth=0.5)
            p1.axes.axvline(x=section.index[0],color='grey',linewidth=0.2, alpha = 0.5)
            p1.axes.axvline(x=section.index[-1],color='grey',linewidth=0.2, alpha = 0.5)
            p1.axes.text(x=section.index.mean(),
                         y=section.max(),
                         s=str(grouper[0]),
                         horizontalalignment='center',
                         )
        
        plt.ylabel('water flow')
        plt.xlabel('date')
        plt.title(f'Time_series with recession limbs at {gauge_name}')
        h,l = p1.get_legend_handles_labels()
        plt.legend(h[:2], l[:2])
        plt.tight_layout()
        fig=p1.figure        
        fig.savefig(Path(output_dir, f'{gauge_name}_flow_timeseries_with_recession_limbs.png'), dpi=300)
        plt.close()

    
    #%% plot the master curve per decade
    for gauge_name,subset in mrc_curve.groupby('gauge'):
        fig, ax = plt.subplots()
        s1=sns.lineplot(data=subset,x='section_time',y='q_rec',hue='decade')
        plt.xlabel('timestep_from_peak')
        plt.ylabel('water flow')
        plt.title(f'MRC Curve per decade at gauge {gauge_name}')
        plt.tight_layout()
        fig.savefig(Path(output_dir, f'{gauge_name}_mrc_decadal_curves.png'), dpi=300)
        plt.close()
    
    return None


