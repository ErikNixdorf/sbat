#%%
import sys
sys.path.append('..')

import pandas as pd
import xarray as xr
import bayes as waterbalance_uncertainity
import yaml

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
with open("sbat_ex4.yaml") as stream:
    try:
        yaml = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
#%%
init = yaml['waterbalance']['bayesian_updating']
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