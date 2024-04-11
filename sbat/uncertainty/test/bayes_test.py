#%%
import pandas as pd
import xarray as xr
from bayes import waterbalance_uncertainity
#%%
#for pandas
#file_path = 'C:/Schoenfeldt.E/Eigene Dateien/BGR/03_Projekte/08_SBATBayes/Daten/testdaten_fuer_elli_2/q_diff.csv'
#df = pd.read_csv(file_path)

#%%
#for xarray
file_path_xarray = 'C:/Schoenfeldt.E/Eigene Dateien/BGR/03_Projekte/08_SBATBayes/Daten/testdaten_fuer_elli_2/data_for_elli.nc'
da = xr.open_dataset(file_path_xarray)
da_test = da.sel(date = ['2015-01-01'])

#%%
init = {'mu_prior': 0,
              'sigma_prior': 0.1,
                'sample_num': 2000,
                'tune_num': 1000, 
                'target_accept': 0.8, 
                'cores_num': 1, 
                'positive_percentage': 'active',
                'gauge': ['boxberg'],
                'year': ['2020-01-01','2020-12-31']}
#%%

sbat_BayesInference = waterbalance_uncertainity(bayes_options=init, 
                                          observed_data = da)
#%%
summary_mean, summary_std = sbat_BayesInference.bayes_update_xarray()

#%%
file_path = 'C:/Schoenfeldt.E/Eigene Dateien/BGR/03_Projekte/08_SBATBayes/Daten/testdaten_fuer_elli_2/summary_mean_jeankendorf.csv'
summary = pd.read_csv(file_path)
# %%
summary_reichwalde = summary[summary['Unnamed: 0'].str.contains('reichwalde')]
summary_boxberg = summary[summary['Unnamed: 0'].str.contains('boxberg')]
# %%
sbat_BayesInference.plot_posterior_station(summary_reichwalde, start = 0, stop = 365)
sbat_BayesInference.plot_posterior_station(summary_boxberg, start = 0, stop = 365)
sbat_BayesInference.plot_positive_percentage(summary_reichwalde, start = 0, stop = 365)
# %%
sbat_BayesInference.plot_posterior_station(summary, start = 0, stop = 365)
sbat_BayesInference.plot_positive_percentage(summary, start = 0, stop = 365)
# %%