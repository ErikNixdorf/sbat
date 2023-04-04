"""
The module provides several function to calculate the recession master curve based on different concepts
"""
#from recession import fit_reservoir_function
import logging
from typing import Tuple, Union, Dict, Any, Callable
import pandas as pd
import numpy as np
mrc_logger = logging.getLogger('sbat.aquifer_parameter')

def adaptive_matching_strip_method(Q: pd.DataFrame, kwargs: dict) -> tuple:
    """
    This is the Python implementation of the adapative matching strip method introduced by Posavec et al 2006
    https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/j.1745-6584.2006.00226.x

    Parameters
    ----------
    Q : pandas.DataFrame
        A DataFrame with columns 'section_id', 'section_time', 'Q', and 'Q0'.
    kwargs : dict
        A dictionary with the following keys:
        - 'fit_reservoir_function': a function that fits a reservoir model to a time series of discharge.
        - 'recession_algorithm': a string specifying the algorithm to use for detecting recession limbs.
        - 'inv_func': a function that computes the inverse of the reservoir function.

    Returns
    -------
    Q_rec_merged : pandas.Series
        A time series of discharge values computed using the master curve method.
    mrc_out : tuple
        A tuple containing the parameters of the master curve function and the goodness-of-fit.
    """
    
            # we first get the order of recession beginning with the highest initial values
    section_order = Q.groupby('section_id')['Q0'].max().sort_values(ascending=False).index.tolist()
    initDf = True
    for section_id in section_order:

        limb = Q[Q['section_id'] == section_id]
        # we calculate the fit for the initial recession limb
        if initDf:
            Q_data = limb['Q'].values
            Q_0 = limb['Q0'].iloc[0]
            fit_parameter, Q_rec_merged, r_coef, _ = kwargs['fit_reservoir_function'](limb['section_time'].values,
                                                                            limb['Q'].values,
                                                                            limb['Q0'].iloc[0],
                                                                            constant_Q_0=True,
                                                                            no_of_partial_sums=1,
                                                                            recession_algorithm=kwargs['recession_algorithm'])
            df_rec_merged = pd.Series(Q_rec_merged, limb['section_time'].values).rename('Q')

            initDf = False
        else:

            # fit the proper location in the already merged part
            t_shift = kwargs['inv_func'](limb['Q0'].iloc[0], fit_parameter[0], fit_parameter[1])
            # add t_shift to section time
            limb.loc[:, 'section_time'] = limb.loc[:, 'section_time'] + t_shift
            # add the limb with shifted time to the extending dataset
            df_merged = pd.concat([pd.Series(Q_data, df_rec_merged.index).rename('Q'),
                                   limb.set_index('section_time')['Q']]).sort_index()

            fit_parameter, Q_rec_merged, r_coef, _ = kwargs['fit_reservoir_function'](df_merged.index.values,
                                                                            df_merged.values,
                                                                            Q_0,
                                                                            constant_Q_0=True,
                                                                            no_of_partial_sums=1,
                                                                            recession_algorithm=kwargs['recession_algorithm'])

            # compute the recession curve and parameters for the combined ones
            df_rec_merged = pd.Series(Q_rec_merged, df_merged.index.values).rename('Q')
            Q_data = np.append(Q_data, limb['Q'].values)

    # after we got the final regression line we can calculate some performance
    Q_rec_merged = df_rec_merged.values
    r_mrc = np.corrcoef(Q_data,Q_rec_merged)[0,1]

    # update the output_data
    mrc_out = (fit_parameter[0], fit_parameter[1], r_mrc)
    
    return Q_rec_merged, mrc_out

def demuth_method(Q: pd.DataFrame, kwargs: dict) -> tuple:  
    """
    This is the python implementation of the master curve method introduced by Demuth 1993.
    In his manuscript (https://www.netzwerk-hydrologie.de/content/bandvolume_1_demuth_s_1993_untersuchungen_zum_niedrigwasser_west_europa)
    The method is mentioned as DEREC2

    Parameters
    ----------
    Q : pandas.DataFrame
        A DataFrame with columns 'section_id', 'section_time', 'Q', and 'Q0'.
    kwargs : dict
        A dictionary with the following keys:
        - 'fit_reservoir_function': a function that fits a reservoir model to a time series of discharge.
        - 'recession_algorithm': a string specifying the algorithm to use for detecting recession limbs.
        - 'inv_func': a function that computes the inverse of the reservoir function.

    Returns
    -------
    Q_rec_merged : pandas.Series
        A time series of discharge values computed using the master curve method.
    mrc_out : tuple
        A tuple containing the parameters of the master curve function and the goodness-of-fit.
    """
    
    # Check inputs
    required_keys = ['fit_reservoir_function', 'recession_algorithm', 'inv_func']
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required argument: {key}")
            
    if not isinstance(Q, pd.DataFrame) or {'section_id', 'section_time', 'Q', 'Q0'} - set(Q.columns):
        raise ValueError("Invalid input DataFrame, check whether section_id, section_time, Q, Q0 are named as columns")
    
    # According to demuth method we first compute an initial fit for all data
    Q_data = Q['Q'].values
    Q_0 = Q['Q0'].mean()
    fit_parameter, Q_rec, r_init, _ = kwargs['fit_reservoir_function'](Q['section_time'].values,
                                                             Q_data,
                                                             Q_0,
                                                             constant_Q_0=False,
                                                             no_of_partial_sums=1,
                                                             min_improvement_ratio=1.05,
                                                             recession_algorithm=kwargs['recession_algorithm'])
  
    # we replace the first fit parameter with the actual Q_0, moving in upward direction
    Q0_max = Q['Q0'].max()
  
    # Every recession limb will be shifted in t_direction on the new base limp
    df_merged = pd.Series(dtype=float)
    for _, limb in Q.groupby('section_id'):
        t_shift = kwargs['inv_func'](limb['Q0'].iloc[0], Q0_max, fit_parameter[1])
        # add t_shift to section time
        limb['section_time'] = limb['section_time'] + t_shift
        df_merged = pd.concat([df_merged, limb.set_index('section_time')['Q']])
  
    # we compute a new mean fitting model of the shifted time series
    df_merged = df_merged.sort_index()
  
    fit_parameter, Q_rec_merged, r_mrc, _ = kwargs['fit_reservoir_function'](df_merged.index.values,
                                                                   df_merged.values,
                                                                   Q0_max,
                                                                   constant_Q_0=True,
                                                                   no_of_partial_sums=1,
                                                                   min_improvement_ratio=1.05,
                                                                   recession_algorithm=kwargs['recession_algorithm'])
  
    # update the output_data
    mrc_out = (fit_parameter[0], fit_parameter[1], r_mrc)
    
    return Q_rec_merged, mrc_out


def tabulating_method(Q,kwargs):
    """
    This MCRS algorithm is the Python Implementation of the tabulating method by Johnseon and Dils
    which has been described in Toebes and Strang
    Toebes, C.; Strang, D.D. 1964. On recessioncurves, 1 ñ Recession equations. Journal ofHydrology (NZ) 3(2): 2-15.
    
    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    
      
def get_master_recession_curve(keyword: str, Q: Any, mcr_parameter: Dict = {}) -> Any:
    """
    This is a wrappper function which calls the methods to estimate
    the master recession curves based on the provided keyword

    Parameters
    ----------
    keyword : str
        The keyword to determine the method to estimate the master recession curve.
    Q : Any
        Input data for the method.
    mcr_parameter : Dict, optional
        Optional parameters for the method. The default is {}.

    Returns
    -------
    Any
        The estimated master recession curve.

    Raises
    ------
    KeyError
        If the provided keyword is not in the supported keyword list.
    """

    mrc_function_map: Dict[str, Callable]={'adaptive_matching_strip':adaptive_matching_strip_method,
                      'demuth':demuth_method
                      }
    
    if keyword not in mrc_function_map:
        raise ValueError(f"No function to calculate master recession curve found for keyword: {keyword}")
    mrc_function = mrc_function_map[keyword]
    
    return mrc_function(Q, mcr_parameter)
    
      
