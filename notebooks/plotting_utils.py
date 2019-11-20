import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
import corner

def plot_fig3_topleft(df, area=1.0, duty_cycle_on=True):
    """Plot the top left panel of Fig 3 showing the distribution of rest-frame tau
    
    Parameters
    ----------
    df : Pandas DataFrame
        catalog containing the keys `log_rf_tau` and `M_i`
    
    """
    if duty_cycle_on:
        df = df[df['duty_cycle']==True].copy()
    filled_log_rf_tau = df[(df['M_i'] > -27.0) & (df['M_i'] < -26.0)]['log_rf_tau']

    #_, binning, _ = plt.hist(df['log_rf_tau'], 
    #                         weights=np.ones_like(df['log_rf_tau'])/area,
    #                         bins=np.arange(0.0, 5.0, 0.1),
    #                         color='tab:blue', edgecolor='k', alpha=0.5, label='all')
    _, binning, _ = plt.hist(df['log_rf_tau'], 
                 weights=np.ones_like(df['log_rf_tau'].values)/area,
                 bins=np.arange(0.0, 5.0, 0.1),
                 color='tab:blue', edgecolor='k', alpha=0.5, label='all')
    _ = plt.hist(filled_log_rf_tau, bins=binning, weights=np.ones_like(filled_log_rf_tau)/area, color='tab:blue', edgecolor='k', alpha=1.0, label='-27 < $M_i$ < -26')
    plt.ylabel('count/sq deg')
    plt.yscale('symlog')
    plt.ylim([0, 50])
    plt.xlabel('log(tau/days)')
    plt.legend()
    
def plot_fig3_topright(df, area=1.0, duty_cycle_on=True):
    """Plot the top right panel of Fig 3 showing the distribution of SF_inf
    
    Parameters
    ----------
    df : Pandas DataFrame
        catalog containing the keys `log_sf_inf` and `M_i`
        
    """
    if duty_cycle_on:
        df = df[df['duty_cycle']==True].copy()
    
    filled_sf = 10.0**(df[(df['M_i'] > -27.0) & (df['M_i'] < -26.0)]['log_sf_inf'].values)

    _, binning, _ = plt.hist(10.0**df['log_sf_inf'].values, weights=np.ones_like(df['log_sf_inf'].values)/area, bins=np.arange(0.0, 1.0, 0.02),
                             color='tab:blue', edgecolor='k', alpha=0.5, label='all')
    _ = plt.hist(filled_sf, bins=binning, weights=np.ones_like(filled_sf)/area, color='tab:blue', edgecolor='k', alpha=1.0, label='-27 < $M_i$ < -26')
    plt.ylabel('count/sq deg')
    plt.yscale('symlog', nonposy='clip')
    plt.ylim([1, 50])
    plt.xlabel('SF_inf (mag)')
    plt.legend()
    
def plot_fig3_bottom(df, bandpass=None, duty_cycle_on=True):
    """Plot the bottom panel of Fig 3 showing the correlation between SF_inf and rest-frame tau
    
    Parameters
    ----------
    df : Pandas DataFrame
        catalog containing the keys `log_sf_inf`, `log_rf_tau` and and `M_i`
        
    """
    if duty_cycle_on:
        df = df[df['duty_cycle']==True].copy()
    
    if bandpass is not None:
        data_2d = df[df['bandpass']==bandpass][['log_sf_inf', 'log_rf_tau']].values
    else:
        data_2d = df[['log_sf_inf', 'log_rf_tau']].values
    figure = corner.corner(data_2d,
                          color='tab:blue',
                          smooth=1.0,
                          labels=['log(SF_inf/mag)', 'log(tau/days)'],
                          fill_contours=True,
                          show_titles=True,
                          levels=[0.2, 0.5, 0.7, 0.9],
                          range=[[-1.5, 0.0], [0, 4.3]],
                          hist_kwargs=dict(density=True))
    
    if False:
        log_tau = all_bands['log_tau'].values
        log_sf = all_bands['log_sf'].values
        _ = plt.hist2d(log_sf, log_tau, bins=50, range=[[-1.4, 0], [0, 4.1]], cmap = plt.cm.jet)
        plt.ylabel('log(tau/days)')
        plt.xlabel('log(SF_inf/mag)')
        
def plot_2d_hist_stats(df, colnames_2d, bins_2d, invert_y_axis, colname_color, ticks_color, invert_color, x_label, y_label, color_label, cmap_name, statistic='mean'):
    """Plot a 2D histogram where the color represents the binned average of a third quantity,
    rather than the number count of that bin
    
    Parameters
    ----------
    df : Pandas DataFrame
        table from which to index the column values
    colnames_2d : list of str
        names of columns used for binning the 2D histogram, in order of [horizontal, vertical]
    bins_2d : list of array-like
        binning to use, in order of [horizontal, vertical]
    invert_y_axis : bool
        whether to invert the y-axis scale
    colname_color : str
        name of column used for bin-averaging and coloring
    ticks_color : list of float
        ticks on the colormap spanning the minimum and maximum colormap values
    invert_color : bool
        whether to invert the color scale
    x_label : str
    y_label : str
    color_label : str
        label of the third quantity that's colored
    cmap_name : str
        string identifier for the matplotlib.cm object to use
    
    """
    hist_info = stats.binned_statistic_dd(sample=df[colnames_2d].values, 
                                          values=df[colname_color].values,
                                          bins=bins_2d,
                                          statistic=statistic)
    # Zero the nan values
    bin_stats = np.nan_to_num(hist_info.statistic)
    bin_edges = hist_info.bin_edges
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=(6, 6), gridspec_kw={'height_ratios': [0.05, 1]})

    im = ax.imshow(bin_stats.T,
                   interpolation=None, 
                   origin='low', 
                   extent=(bin_edges[0][0], 
                           bin_edges[0][-1],
                           bin_edges[1][0],
                           bin_edges[1][-1]),
                   cmap=cmap_name, vmin=ticks_color[0], vmax=ticks_color[-1], aspect='auto')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if invert_y_axis:
        ax.invert_yaxis()
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cax.set_xlabel(color_label)
    if invert_color:
        cax.invert_xaxis()
    cbar.set_ticks(ticks_color)
    cbar.set_ticklabels(ticks_color)
    plt.tight_layout()
    
def plot_2d_hist(x_values, y_values, x_bins, y_bins, invert_y_axis, ticks_color, invert_color, log_color, x_label, y_label, color_label, cmap_name, weight=1.0):
    """Plot a normal 2D histogram where the color represents the number count of that bin
    
    Parameters
    ----------
    x_values : array-like
        values to go on the horizontal axis
    y_values : array-like
        values to go on the vertical axis
    x_bins : array-like
        horizontal binning
    y_bins : array-like
        vertical binning
    invert_y_axis : bool
        whether to invert the y-axis scale
    colname_color : str
        name of column used for bin-averaging and coloring
    ticks_color : list of float
        ticks on the colormap spanning the minimum and maximum colormap values
    invert_color : bool
        whether to invert the color scale
    log_color : bool
        whether to put the color on a log scale
    x_label : str
    y_label : str
    color_label : str
        label of the third quantity that's colored
    cmap_name : str
        string identifier for the matplotlib.cm object to use
    weight : float
        weight for each pair
    
    """
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=(6, 6), gridspec_kw={'height_ratios': [0.05, 1]})
    if log_color:
        custom_norm = colors.LogNorm(clip=True)
    else:
        custom_norm = None
    hist, _, _, im = ax.hist2d(x_values, y_values, bins=[x_bins, y_bins], cmap=cmap_name, norm=custom_norm, weights=np.ones_like(x_values)/weight)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if invert_y_axis:
        ax.invert_yaxis()
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cax.set_xlabel(color_label)
    if invert_color:
        cax.invert_xaxis()
    if ticks_color is not None:
        cbar.set_ticks(ticks_color)
        cbar.set_ticklabels(ticks_color)
    plt.tight_layout()
    
def plot_fig13(df, y_colname, y_range, y_label):
    """Plot Fig 13, which shows the trend of the given quantity with rest-frame wavelength for each band
    
    Parameters
    ----------
    df : Pandas DataFrame
        table from which to index the column values, must have keys `log_rf_wavelength`, `bandpass`, and `y_colname`
    y_colname : str
        name of column for the y axis
    y_range : array-like
        range of y values to plot
    y_label : str
    
    """
    color_dict = dict(zip(list('ugriz'), ['m', 'b', 'green', 'orange', 'red']))
    fig = corner.corner(df[df['bandpass']=='u'][['log_rf_wavelength', y_colname]].values,
                          color='m',
                          smooth=1.0,
                          labels=['log(rest-frame wavelength/4000A)', y_label],
                          #fill_contours=False,
                        no_fill_contours=True,
                        plot_datapoints=False,
                        plot_contours=True,
                          show_titles=True,
                          levels=[0.3, 0.7],
                          range=[[-0.7, 0.3], y_range],
                          hist_kwargs=dict(density=True),
                       contour_kwargs=dict(linestyles='solid'))
    for bp in 'griz':
        fig = corner.corner(df[df['bandpass']==bp][['log_rf_wavelength', y_colname]].values,
                          color=color_dict[bp],
                          smooth=1.0,
                          labels=['log(rest-frame wavelength/4000A)', y_label],
                          #fill_contours=False,
                        no_fill_contours=True,
                        plot_datapoints=False,
                        plot_contours=True,
                          show_titles=True,
                      fig=fig,
                          levels=[0.3, 0.7],
                          range=[[-0.7, 0.3], y_range],
                          hist_kwargs=dict(density=True),
                       contour_kwargs=dict(linestyles='solid'))
    fig.set_size_inches(18, 10, forward=True)