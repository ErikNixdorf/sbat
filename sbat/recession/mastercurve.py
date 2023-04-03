"""
The module provides several function to calculate the recession master curve based on different concepts
"""
#from recession import fit_reservoir_function
import logging
from typing import Tuple, Union, Dict
import pandas as pd
import numpy as np
mrc_logger = logging.getLogger('sbat.aquifer_parameter')

def mrc_ams_method(Q,**kwargs):
    
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
    df_rec_merged = df_rec_merged.to_frame()
    df_rec_merged['Q_data'] = Q_data
    r_mrc = df_rec_merged.corr().to_numpy()[0, 1]

    # update the output_data
    mrc_out = (fit_parameter[0], fit_parameter[1], r_mrc)
    
    return Q, mrc_out

def mrc_demuth_method(Q,kwargs):   
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
      
      return Q, mrc_out
  
def get_master_recession_curve(keyword,Q,mcr_parameter: Dict = {}):
    
    mrc_function_map={'adaptive_matching_strip':mrc_ams_method,
                      'demuth':mrc_demuth_method
                      }
    
    if keyword not in mrc_function_map:
        raise ValueError(f"No function to calculate master recession curve found for keyword: {keyword}")
    mrc_function = mrc_function_map[keyword]
    
    return mrc_function(Q, mcr_parameter)
    
      
