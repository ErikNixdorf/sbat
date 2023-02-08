from datetime import datetime

from sbat.sbat import model
import pandas as pd
import os
import numpy as np
import geopandas as gpd
from copy import deepcopy
import seaborn as sns
from matplotlib import pyplot as plt
#functiton to dateparse
dateparse = lambda x: datetime.strptime(x, '%d/%m/%y')

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

rolling_mean_span=1

#%% load data 
gauge_ts=pd.read_csv(os.path.join(os.path.dirname(__file__),'input','ahlequelle.csv'),
                     index_col=0,
                     parse_dates=['Datum'], 
                     date_parser=dateparse,encoding='latin1')

meta_data=pd.DataFrame()
#%% We interpolate to daily values
gauge_ts=gauge_ts.rolling(1).mean().resample('d').mean().interpolate()
gauge_ts=gauge_ts.rename(columns={'Schüttung (m³/h)':'ahlequelle'})

"""
#%% We test several applications how to get an Hand on the different sections of the curve
import numpy as np
from scipy.signal import savgol_filter
#gauge_ts=gauge_ts.rolling(1).mean().resample('d').mean().interpolate()
Q=gauge_ts.copy().resample('d').mean().interpolate()

#test smoothing schemes
Q=savgol_filter(Q.values.flatten(), int(round_up_to_odd(0.05*len(Q))), 2)

d2y_dx2=np.gradient(np.gradient(Q.flatten()))
d2y_dx2=np.nan_to_num(d2y_dx2)
infls0 = np.where(np.diff(np.sign(d2y_dx2)))[0]
dy_dx=np.gradient(Q.flatten())
dy_dx=np.nan_to_num(dy_dx)
curvature = d2y_dx2 / (1 + dy_dx**2)**(3/2)
infls2 = np.where(np.diff(np.sign(curvature)))[0]
plt.subplots()
# plot results
plt.subplots()
plt.plot(gauge_ts.copy().resample('d').mean().interpolate().values/np.max(gauge_ts.copy().resample('d').mean().interpolate().values), label='Noisy Data')
plt.plot(Q/np.max(Q), label='Smoothed Data')
plt.plot(d2y_dx2 / np.max(d2y_dx2), label='Second Derivative (scaled)')

#we sort according to condition

infl_old=infls0.tolist()
infl_new=[infl_old[0]]
i=2
threshold=60
for infl in infls0:
    if infl-infl_new[-1]>threshold:
        infl_new.append(infl)

    

#infls0=infls0[np.where(np.diff(infls0)>30)]
for i, infl in enumerate(infl_new, 1):
    plt.axvline(x=infl, color='k')
plt.legend(bbox_to_anchor=(1.55, 1.0))
haha
#test an approach where we set the filter on the derivative
Q=gauge_ts.copy().resample('d').mean().interpolate().values
d2y_dx2=np.gradient(np.gradient(Q.flatten()))
d2y_dx2=np.nan_to_num(d2y_dx2)
d2y_dx2=savgol_filter(d2y_dx2, int(round_up_to_odd(0.1*len(d2y_dx2))), 3)
infls = np.where(np.diff(np.sign(d2y_dx2)))[0]
dy_dx=np.gradient(Q.flatten())
dy_dx=np.nan_to_num(dy_dx)
curvature = d2y_dx2 / (1 + dy_dx**2)**(3/2)
infls2 = np.where(np.diff(np.sign(curvature)))[0]

# plot results
plt.subplots()
plt.subplots()
plt.plot(Q/np.max(Q), label='Noisy Data')
#plt.plot(Q/np.max(Q), label='Smoothed Data')
plt.plot(d2y_dx2 / np.max(d2y_dx2), label='Second Derivative (scaled)')
for i, infl in enumerate(infls, 1):
    plt.axvline(x=infl, color='k')
plt.legend(bbox_to_anchor=(1.55, 1.0))

#curvature model




#%% compute inflection points

import numpy as np
from scipy.ndimage import gaussian_filter1d
gauge_smooth = gaussian_filter1d(gauge_ts.values, 10)
print(sum((gauge_ts-gauge_smooth).values))
haha
#gauge_ts=gauge_ts.rolling(60).mean().dropna()
smooth_d2 = np.gradient(np.gradient(gauge_smooth,axis=0),axis=0).flatten()
#compute inflection points
# find switching points
infls = np.where(np.diff(np.sign(smooth_d2)))[0]


# plot
import matplotlib.pyplot as plt

plt.plot(gauge_ts.values/np.max(gauge_ts.values), label='Noisy Data')
plt.plot(gauge_smooth/np.max(gauge_smooth), label='Smoothed Data')
haha
plt.plot(smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')

for i, infl in enumerate(infls, 1):
    plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
plt.legend(bbox_to_anchor=(1.55, 1.0))
haha
"""
#%% load the control function
sbat=model(gauge_time_series=gauge_ts,
           gauge_network=gpd.GeoDataFrame(),
           gauge_metadata=meta_data,
           valid_datapairs_only=False,
           decadal_stats=False,
           dropna_axis=0)

sbat.get_recession_curve(curve_type='discharge',plot=False,mrc_algorithm=None,
                         minimum_limbs = 1,multiple_aquifers_per_recession=True,
                         minimum_recession_curve_length=30)


#%% We plot their results


haha
#%% recession curves
#it is a little tricky, we first calculate the sections based on the mean
bf_daily_mean=bf_daily.groupby(['Datum','gauge']).mean()
sbat.bf_output['bf_daily']=bf_daily_mean

df_recession_limbs=sbat.recession_limbs_ts.copy(deep=True).set_index('Datum').dropna(axis=1)[['section_id', 'section_length', 'section_time']]
#now we loop trough all methods but using the limbs found for the ensemble but replacing the Q0 data

#compute per method
recession_results=pd.DataFrame()
for method,section in bf_daily.groupby('variable'):
    #we reduce to the time steps to the ones of the recession limbs

    x=pd.concat([section,df_recession_limbs],axis=1).dropna()

    section=pd.concat([section,df_recession_limbs],axis=1).dropna()
    #we reduce to the time steps to the ones of the recession limbs
    section=section.rename(columns={'value':'Q'})
    Q0= section[['Q','section_id']].groupby('section_id').max().squeeze()
    section['Q0']=section['section_id'].replace(Q0)
    section['Q0_inv']=1/section['Q0']     

    sbat.bf_output['bf_daily']=section
    
    
    
    #we first calculate the recession curve for the mean

    

    print('recession for method ',method)
    for rec_algorithm in ['boussinesq','maillet']:
        sbat.get_recession_curve(curve_type='baseflow',plot=False,mrc_algorithm=None,
                                 minimum_limbs = 1,
                                 recession_algorithm=rec_algorithm,
                                 define_falling_limb_intervals=False)
        df_recession_result=sbat.recession_limbs_ts.copy()
        df_recession_result['bf_method']=method
        recession_results=pd.concat([recession_results,df_recession_result])
    
#clean a little bit
drop_cols=[col for col in ['diff','Q0_inv','mrc_algorithm'] if col in recession_results.columns]
recession_results=recession_results.drop(columns=drop_cols)



#%% For later plotting comparison we need to synchronize 


recession_results=recession_results.reset_index()
#%% check some results for consistentce
fig,ax2=plt.subplots()
df2=recession_results[recession_results.bf_method=='Willems']
df3=bf_daily[bf_daily.variable=='Willems']
df2=df2.set_index('Datum')
df3.value.plot(ax=ax2)
df2.Q.plot(ax=ax2)

#%% First we plot the curve for baseflow and Q
sns.set_context('talk')
fig,ax=plt.subplots()

#first we plot the time series
gauge_ts.plot(ax=ax)

#then the baseflow, we first have to interpolate

bf_daily_plot=bf_daily.reset_index()
sns.lineplot(data=bf_daily_plot,ax=ax,x='Datum',y='value')
plt.legend([gauge_ts.columns[0],('BQ '+gauge_ts.columns)[0]])
ax.set_ylabel('Q [m3/s]')
ax.set_xlabel('Date')
#we plot vertical lines for each 

#%% Second we have to rearrange the data in 

#%%
haha

recession_results.to_csv(os.path.join(os.path.dirname(__file__),'output','ahlequelle_recession.csv'))
