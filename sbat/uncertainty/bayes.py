
"""
Initialize Bayes inference class when bayesian_updating "activate" is set as true in sbat yaml file. 

Parameters
----------
observed_data: 
    NetCDF containing the gauge name and uncertainity measurements for each day. Each day should contain at least 50 values.  
bayes_options:
    Dictionary containing Bayesian updating information (sbat.yml file)
    Necessary entries: 
        - mu_prior (int): Prior knowledge about possible outcome of waterbalance values. Default: 0
        - sigma_prior (int): Estimation of uncertainity within prior. Default: 0.1
        - sample_num (int): Samples drawn from the posterior distribution. Default: 2000,
        - tune_num (int): Number of iterations used for tuning the parameters. Default: 1000, 
        - target_accept (int):  Represents the target acceptance probability for proposed steps in the MCMC algorithm. 
                                It is usually set between 0.5 and 0.95. Default: 0.8, 
        - cores_num (int): Chose number of cores for calculation. Note that higher values than 1 might lead to an error. Default: 1, 
        - positive_percentage (str): Can be set as 'inactive' or 'active' (default)
        - gauge (lst): Select specific gauge by name e.g. ['boxberg']. Minimum one entry has to be given or the option "all" can be chosen.
        - date_select (lst): Select subset of measurements for calaculation based on datum. 
                    Most be given in string as "YYYY-MM-DD" with first entry being the first day and next entry being the last. 
                    Example: ['2020-01-01','2020-12-31'] #TODO Can more dates than two be included? A: At the moment not -> Should be included
                    Default: 'None' 

Returns
-------

Mean & Standard deviation (dict)
    keys: name of station and date of measurement
    Includes the following parameters: 
        - mean
        - sd
        - hdi_3% (highest density interval)
        - hdi_97% 
        - mcse_mean 
        - mcse_sd 
        - ess_bulk
        - ess_tail
        - r_hat: should be 1 or 0.99. If less either include more measurements or increase sample_num in sbat yaml file
        - positive_perc: If set true: percentage of positive samples drawn from the posterior distribution. 

"""
#load libraries
import numpy as np
import pymc3 as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import re

class waterbalance_uncertainity():

    def __init__(self,
                bayes_options: dict,
                observed_data = pd.DataFrame()):
        self.bayes_options = bayes_options
        self.observed_data = observed_data
        
        self.prepare_data()
###      

    def is_netdcf(self, obj):
        ''' Check if passed data is a xarray file'''
        return isinstance(obj, (xr.Dataset, xr.DataArray))


###    
    def prepare_data(self):
        ''' Save information from yaml file internally'''
        
        self.mu_prior = self.bayes_options['mu_prior']
        self.sigma_prior = self.bayes_options['sigma_prior']
        self.target_accept = self.bayes_options['target_accept']
        self.tune_num = self.bayes_options['tune_num']
        self.sample_num  = self.bayes_options['sample_num']
        self.cores_num = self.bayes_options['cores_num']

###
    def select_date(self):
        ''' Use the specified dates from the yaml file to slice the dataset.'''

        if self.bayes_options['date_select'] == 'None': #No specific selection of dates
            None
        else: 
            if (isinstance(self.bayes_options['date_select'], list) and #test if a list was given
                len(self.bayes_options['date_select']) == 2 and #test if it has two entries
                all(isinstance(item, str) for item in self.bayes_options['date_select'])): #test if entries are strings

                self.observed_data = self.observed_data.sel(date = slice(self.bayes_options['date_select'][0],self.bayes_options['date_select'][1]))
            else:
                print('''To select a specific date set the parameter "date_select" to a list of strings in the format "YYYY-MM-DD". 
                      'Please check the yaml file.''')
    
    def select_loc(self):
        ''' Create gauge vriable based on the specified gauges in the yaml file or select all gauges from the dataset.'''

        if self.bayes_options['gauge'] == 'all':
            self.gauge = list(self.observed_data.data_vars.keys())
        else:
            self.gauge = self.bayes_options['gauge']

###
    def update(self): 
        ''' Bayesian inference using the uncertainity estimations from the waterbalance output'''
        if self.is_netdcf(self.observed_data): #test if an xarray dataset was passed
            self.select_date() #slice date
            self.select_loc()
            results_df_mean = pd.DataFrame() #prepare empty df for results
            results_df_std = pd.DataFrame()  #prepare empty df for results
            for loc in self.gauge: #iterate through every given location
                for time in self.observed_data.date.values: #iterate through every day
                        

                        model_name = str(loc) + str(time) #create key for df for each location and day
                        self.bayes_data = self.observed_data[loc].sel(date = time).values #extract values for specific gauge and day
                        summary_bayes = self.bayes_model() #bayes model

                        summary_mean = summary_bayes.iloc[0] #extract results for mean
                        summary_std = summary_bayes.iloc[1] #extract results for std
                        summary_mean.rename(model_name, inplace = True) #rename key; use model name
                        summary_std.rename(model_name, inplace = True)

                        results_df_mean = pd.concat([results_df_mean, summary_mean], axis = 1) #merge results
                        results_df_std = pd.concat([results_df_std, summary_std], axis = 1)

            
            return results_df_mean.T, results_df_std.T
        
        else:
            print('The passed data does not contain an xarray Dataset or DataArray.')
###
    def bayes_model(self):
        '''Built Bayes Model'''

        with pm.Model() as hydro_model:
        # Prior distributions for the parameters


            sigma_squared = pm.HalfNormal('sigma_squared', sigma=self.sigma_prior)  # Prior for variance of true flow rate
            mu = pm.Normal('mu', mu = self.mu_prior, sigma = pm.math.sqrt(sigma_squared))  

            # Likelihood function
            likelihood = pm.Normal('likelihood', 
                                   mu = mu, 
                                   observed=self.bayes_data)

            # Sampling
            trace = pm.sample(self.sample_num, 
                            tune=self.tune_num, 
                            cores=self.cores_num, 
                            target_accept = self.target_accept)
            

            print(pm.summary(trace))
            summary_bayes = pm.summary(trace)
            
            if self.bayes_options['positive_percentage'] == True:
                summary_bayes['positive_perc'] = self.probability_positive(trace)




        return summary_bayes

###
    def probability_positive(self, trace):
        '''Calculate the percentage of posterior samples with positive values'''
        mu_samples = trace['mu']

        percentage_positive = np.mean(mu_samples > 0) * 100
        print(percentage_positive, "% of posterior samples are positive values")

        return percentage_positive
