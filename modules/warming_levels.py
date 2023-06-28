import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt

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
    files = glob.glob(f'data/{project}_tas_landsea/*{experiment}*.csv')
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

    baseline_subset = baseline[np.logical_and(baseline.year >= baseline_range[0], 
                                              baseline.year <= baseline_range[1])]
    future_subset = future[np.logical_and(future.year >= future_range[0],
                                          future.year <= future_range[1])]

    future_annual = future.groupby(['year', 'model']).mean(numeric_only=True).reset_index()
    baseline_subset_annual = baseline_subset.groupby(['year', 'model']).mean(numeric_only=True).reset_index()

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
        sns.lineplot(future_annual, x='year', y='world', units='model', estimator=None, c='#ccc', linewidth=1, ax=ax)
        sns.lineplot(baseline_subset_annual, x='year', y='world', units='model', estimator=None, c='#aaa', linewidth=1, ax=ax)
        sns.lineplot(baseline_subset_annual.groupby('year').mean(numeric_only=True), 
                     x='year', y='world', c='red', linewidth=2, ax=ax, label='Annual multimodel mean')
        sns.lineplot(future_annual.groupby('year').mean(numeric_only=True), x='year', y='world', c='red', 
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