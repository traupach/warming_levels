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

def warming_for_period(project, experiment, year_range, baseline_temp=13.6, figsize=(10,5), plot=True):
    """
    Read all temperature files for a given project/experiment (e.g. CMIP6/ssp585).
    
    Arguments:
        project: The project (e.g. CMIP6).
        experiment: The experiment (e.g. ssp585).
        year_range: [stard, end] years inclusive.
        baseline_temp: Baseline global mean temperture to calculate changes from.
                       By default this is 13.6 C, based on the 1850-1900 in the IPCC Atlas
                       (https://interactive-atlas.ipcc.ch/permalink/Gzfztqvg).
        figsize: Figure size.
        plot: Plot a figure?
        
    Returns: Mean warming for the given year range.
    """

    tas = read_all_temps(project=project, experiment=experiment)
    tas['year'] = tas.date.dt.year
    tas['world'] = tas['world'] - baseline_temp
    tas = tas.groupby(['year', 'model']).mean(numeric_only=True).reset_index()
    tas_mean = tas.groupby('year').mean(numeric_only=True).reset_index()

    temp_change = tas[np.logical_and(tas.year >= year_range[0],
                                     tas.year <= year_range[1])].world.mean()

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(tas, x='year', y='world', units='model', estimator=None, c='#ccc', linewidth=1, ax=ax)
        sns.lineplot(tas_mean, x='year', y='world', c='red', linewidth=2, ax=ax, label='Annual multimodel mean')
        ax.set_xlabel('Year')
        ax.set_ylabel('Warming [deg. C]')
        ax.set_title((f'Warming over 1850-1900 baseline in {project} {experiment}.\n' + 
                      f'Mean warming for {year_range[0]}-{year_range[1]} is {np.round(temp_change, 2)} deg. C.'))

        ax.fill_between(tas.year, y1=0, y2=1, where=np.logical_and(tas.year >= year_range[0],
                                                                   tas.year <= year_range[1]),
                        color='yellow', alpha=0.3, transform=ax.get_xaxis_transform())

        plt.show()
        
    return temp_change