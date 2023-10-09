import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import xarray
import datetime

module_path = Path(__file__).parent.parent

def read_temps(file, cols=['date', 'world']):
    """
    Read monthly global mean temperatures from a datafile.
    
    Arguments:
        file: CSV to read from.
        cols: Columns to keep.
        
    Returns: Pandas dataframe with temperatures by month.
    """

    project = pd.read_table(file, sep=':', nrows=1, skiprows=2, header=None).loc[0,1].replace(' ','')
    experiment = pd.read_table(file, sep=':', nrows=1, skiprows=3, header=None).loc[0,1].replace(' ','')
    model = pd.read_table(file, sep=':', nrows=1, skiprows=4, header=None).loc[0,1].replace(' ','')

    dat = pd.read_csv(file, comment='#')[cols]
    dat['project'] = project
    dat['model'] = model
    dat['experiment'] = experiment

    dat['date'] = pd.to_datetime(dat.date)
    return dat

def read_all_temps(project, experiment):
    """
    Read all temperature files for a given project/experiment (e.g. CMIP6/ssp585).
    
    Arguments:
        project: The project (e.g. CMIP6).
        experiment: The experiment (e.g. ssp585).
        
    Returns: All records for the given project/experiment in a Pandas DataFrame.
    """
    
    out = pd.DataFrame()
    files = glob.glob(f'{module_path}/data/{project}_tas_landsea*/*{experiment}*.csv')
    for file in files:
        dat = read_temps(file=file)
        assert np.all(dat.project == project), 'Project/dataset mismatch'
        assert np.all(dat.experiment == experiment), 'Experiment/scenario mismatch.'
        out = pd.concat([out, dat])

    return out

def warming_amount(project, baseline_experiment, future_experiment, 
                   baseline_range, future_range, figsize=(10,5), plot=True):
    """
    Determine warming amount for a given year range.
    
    Arguments:
        project: The project (e.g. CMIP6).
        baseline_experiment: The experiment to use for baseline temperatures (e.g. historical).
        future_experiment: The experiment to use for future temperatures (e.g. ssp585). 
        baseline_range: [start, end] years inclusive.
        future_range: [start, end] year inclusive.
        figsize: Figure size.
        plot: Plot a figure?
        
    Returns: Mean warming for the given year range.
    """

    baseline = read_all_temps(project=project, experiment=baseline_experiment)
    future = read_all_temps(project=project, experiment=future_experiment)

    baseline['year'] = baseline.date.dt.year
    future['year'] = future.date.dt.year

    # Concatenate baseline and future temps.
    temps = pd.concat([baseline, future])
    
    baseline_subset = temps[np.logical_and(temps.year >= baseline_range[0], 
                                           temps.year <= baseline_range[1])]
    future_subset = temps[np.logical_and(temps.year >= future_range[0],
                                         temps.year <= future_range[1])]

    temps_annual = temps.groupby(['year', 'model']).mean(numeric_only=True).reset_index()
    
    mean_temps = baseline_subset.groupby('model').mean(numeric_only=True)[['world']]
    mean_temps = mean_temps.rename(columns={'world': 'baseline'})
    mean_temps['future'] = future_subset.groupby('model').mean(numeric_only=True)[['world']]
    mean_temps['change'] = mean_temps.future - mean_temps.baseline
    
    ## Drop NAs in case there are different numbers of models for future vs. baseline experiments.
    mean_temps = mean_temps.dropna()
    assert not np.any(np.isnan(mean_temps.future)), 'Mismatch in number of models.'
    assert not np.any(np.isnan(mean_temps.baseline)), 'Mismatch in number of models.'
    
    temp_change = mean_temps.change.mean()
    num_models = len(mean_temps)

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(temps_annual, x='year', y='world', units='model', estimator=None, c='#ccc', linewidth=1, ax=ax)
        sns.lineplot(temps_annual.groupby('year').mean(numeric_only=True), x='year', y='world', c='red', 
                     linewidth=2, ax=ax)

        ax.set_xlabel('Year')
        ax.set_ylabel('Global mean temperature [deg. C]')
        ax.set_title((f'Global mean temperature from {project} ({num_models} models).\n' + 
                      f'Mean warming from {baseline_range[0]}-{baseline_range[1]} ({baseline_experiment}) ' + 
                      f'to {future_range[0]}-{future_range[1]} ({future_experiment}) ' + 
                      f'is {np.round(temp_change, 2)} deg. C.'))

        ax.axvspan(xmin=baseline_range[0], xmax=baseline_range[1], color='green', alpha=0.1)
        ax.axvspan(xmin=future_range[0], xmax=future_range[1], color='yellow', alpha=0.3)

        plt.show()
        
    return temp_change

def obs_warming(from_range, to_range, best_file='data/BEST_observed_temperatures/Land_and_Ocean_complete.txt', plot=True, figsize=(10,5)):
    """
    Determine warming from period A to period B in the BEST archive
    (https://doi.org/10.5194/essd-12-3469-2020)
    
    Arguments:
        from_range: The range of years to find the anomaly for.
        to_range: The range of years to find differences for.
        best_file: The file for BEST temperature data.
        plot: Make a plot to show the results?
        
    Returns:
        The mean of monthly anomalies for the selected years.
    """
    
    temp = pd.read_csv(f'{module_path}/{best_file}', comment='%', header=None, delimiter=r"\s+",
                       names=['year', 'month', 'monthly_anom', 'monthly_unc', 'annual_anom', 'annual_unc', 
                              '5yr_anom', '5yr_unc', '10yr_anom', '10yr_unc', '20yr_anom', '20yr_unc']) 
    
    temp['day'] = 1
    temp['time'] = pd.to_datetime(temp[['year', 'month', 'day']])
    
    from_anom = temp[np.logical_and(temp.year >= from_range[0],
                                    temp.year <= from_range[1])].monthly_anom.mean()
    to_anom = temp[np.logical_and(temp.year >= to_range[0],
                                  temp.year <= to_range[1])].monthly_anom.mean()
    
    res = to_anom - from_anom
    
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(temp, x='time', y='monthly_anom', estimator=None, c='black', linewidth=1, ax=ax)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Global mean temperature [deg. C]')
        ax.set_title((f'Global mean temperature from BEST observations.\n' + 
                      f'Mean warming from ' +
                      f'{from_range[0]}-{from_range[1]} ' + 
                      f'to {to_range[0]}-{to_range[1]} ' + 
                      f'is {np.round(res, 2)} deg. C.'))

        ax.axvspan(xmin=pd.to_datetime(f'{from_range[0]}-1-1'), 
                   xmax=pd.to_datetime(f'{from_range[1]}-12-31'), color='green', alpha=0.1)
        ax.axvspan(xmin=pd.to_datetime(f'{to_range[0]}-1-1'), 
                   xmax=pd.to_datetime(f'{to_range[1]}-12-31'), color='yellow', alpha=0.3)

        plt.show()
        
    return res

def obs_anomaly(year_range, best_file='data/BEST_observed_temperatures/Land_and_Ocean_complete.txt', plot=True, figsize=(10,5)):
    """
    Determine temperature anomalies over Jan 1951-Dec 1980 for a given period, based on Berkeley Earth surface temperature observations
    (https://doi.org/10.5194/essd-12-3469-2020)
    
    Arguments:
        year_range: The range of years to find the anomaly for.
        best_file: The file for BEST temperature data.
        plot: Make a plot to show the results?
        
    Returns:
        The mean of monthly anomalies for the selected years.
    """
    
    temp = pd.read_csv(f'{module_path}/{best_file}', comment='%', header=None, delimiter=r"\s+",
                       names=['year', 'month', 'monthly_anom', 'monthly_unc', 'annual_anom', 'annual_unc', 
                              '5yr_anom', '5yr_unc', '10yr_anom', '10yr_unc', '20yr_anom', '20yr_unc']) 
    
    temp['day'] = 1
    temp['time'] = pd.to_datetime(temp[['year', 'month', 'day']])
    
    res = temp[np.logical_and(temp.year >= year_range[0],
                              temp.year <= year_range[1])].monthly_anom.mean()

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(temp, x='time', y='monthly_anom', estimator=None, c='black', linewidth=1, ax=ax)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Global mean temperature [deg. C]')
        ax.set_title((f'Global mean temperature from BEST observations.\n' + 
                      f'Mean warming from 1951-1980 ' + 
                      f'to {year_range[0]}-{year_range[1]} ' + 
                      f'is {np.round(res, 2)} deg. C.'))

        ax.axvspan(xmin=pd.to_datetime('1850-1-1'), xmax=pd.to_datetime('1900-12-31'), color='green', alpha=0.1)
        ax.axvspan(xmin=pd.to_datetime(f'{year_range[0]}-1-1'), 
                   xmax=pd.to_datetime(f'{year_range[1]}-12-31'), color='yellow', alpha=0.3)

        plt.show()
        
    return res

def warming_window(warming_amount, project, baseline_range, baseline_experiment, future_experiment):
    """
    Determine the 20 year window over which a model first reaches a certain number of degrees warming.
    Based on the method outlined by Hauser, Engelbrecht, and Fischer (DOI: 10.5281/zenodo.7390473, 
    https://github.com/mathause/cmip_warming_levels/blob/main/README.md, accessed 28.09.2023).

    Arguments:
        warming_amount: Amount in degrees C to look for.
        project: The project (CMIP6 or CMIP5).
        baseline_range: The [start, end] years for the baseline.
        baseline_experiment: Experiment for the baseline range (nearly always 'historical').
        future_experiment: Experiment for the future years (e.g. 'ssp585').

    Returns: start_year and end_year for 20 year ranges when the warming is first reached.
    """

    baseline = read_all_temps(project=project, experiment=baseline_experiment)
    future = read_all_temps(project=project, experiment=future_experiment)
    
    baseline['year'] = baseline.date.dt.year.astype('int')
    future['year'] = future.date.dt.year.astype('int')
    assert baseline.year.max() == future.year.min() - 1
    
    # Concatenate historic to future simulations.
    temps = pd.concat([baseline, future]).rename(columns={'world': 'temp'})
    
    # Calculate annual means.
    temps = temps.groupby(['model', 'year']).mean(numeric_only=True).reset_index()
    temps = temps.set_index('model')

    # Find baseline average.
    baseline_mean = temps[np.logical_and(temps.year >= baseline_range[0], 
                                         temps.year <= baseline_range[1])]
    baseline_mean = baseline_mean.groupby('model').mean()[['temp']]
    
    # Subtract baseline average from annual temperatures.
    temps['change'] = temps.temp - baseline_mean.temp
    temps = temps.sort_values(['model', 'year'])
    
    # Find rolling window average of temperature change.
    temps = temps.reset_index().set_index('year')
    avg_change = temps.groupby('model').rolling(20, center=True).mean()
    avg_change = avg_change.reset_index().dropna()
    
    # The center year of the warming window is the first year that the rolling mean change
    # exceeds the required warming amount.
    years = avg_change.loc[avg_change.change > warming_amount].groupby('model').year.min()
    years = years.reset_index()
    
    # Find the 20 year period around the centre year.
    years['start_year'] = (years.year - 20 / 2).astype('int')
    years['end_year'] = (years.year + (20 / 2 - 1)).astype('int')
    years = years.drop(columns='year').set_index('model')

    return years

def monthly_mean_temps(desc, CMIP6_dir='/g/data/oi10/replicas', out_dir='data/CMIP6_tas_landsea_local/'):
    """
    Calculate and write to disk global monthly mean temperatures for a given model instance.
    
    Arguments:
        desc: CMIP6 model descriptor.
        CMIP6_dir: CMIP6 data directory.
        out_dir: Output directory.
    """
    
    project, _, _, model, exp, ens = desc.split('.')
    path = f'{CMIP6_dir}/{desc.replace(".", "/")}/Amon/tas/*/'
    version = [os.path.basename(x) for x in sorted(glob.glob(f'{path}/v*'))][-1] # Use latest version.
    path = path + '/' + version + '/*.nc'

    dat = xarray.open_mfdataset(path, parallel=True).load()
    
    weights_path = f'{CMIP6_dir}/{desc.replace(".", "/")}/fx/areacella/*/v*/*.nc'
    weights_files = glob.glob(weights_path)
    if len(weights_files) == 0:
        w, _ = xarray.broadcast(dat.lat, dat.tas)
        dat['weight'] = np.cos(w.isel(time=0) * np.pi/180)
        weights_def = 'cosine of latitude'
    else: 
        dat['weight'] = xarray.open_mfdataset(weights_path, parallel=True).areacella.load()
        weights_def = 'model variable areacella'
    
    global_mean_tas = (dat.tas * dat.weight).sum(['lat', 'lon']) / dat.weight.sum()
    global_mean_tas = global_mean_tas - 273.15
    global_mean_tas = global_mean_tas.reset_coords(drop=True)
    global_mean_tas.name = 'world'
    global_mean_tas['date'] = global_mean_tas.time.dt.strftime('%Y-%m')
    global_mean_tas = global_mean_tas.to_dataframe().reset_index().drop(columns='time')
    
    # Write the file header.
    out_file = f'{out_dir}/CMIP6_{model}_{exp}_{ens}.csv'
    f = open(out_file, 'w')
    f.writelines(['#Dataset: Global monthly mean temperatures\n',
                  '#Reference: NA\n',
                  f'#Project: {project}\n',
                  f'#Experiment: {exp}\n',
                  f'#Model: {model}_{ens}\n',
                  '#Variable: tas\n', 
                  '#Variable_longname: mean near-surface air temperature\n',
                  '#Units: degC\n', 
                  '#Time_frequency: month\n', 
                  '#Feature_type: regional mean time series\n', 
                  '#Regions: global\n', 
                  '#Area: land and sea\n', 
                  '#Spatial_resolution: as per source model\n', 
                  f'#Interpolation_method: weighted mean, weights defined by {weights_def}\n',
                  f'#Creation_Date: {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}\n'])
    f.close()
    
    global_mean_tas.to_csv(out_file, index=False, mode='a')