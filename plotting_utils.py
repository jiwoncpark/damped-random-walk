import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_log_rf_tau_hist(drw_df):
    filled_log_rf_tau = drw_df[(drw_df['M_i_kcorr'] > -27.0) & (drw_df['M_i_kcorr'] < -26.0)]['log_rf_tau']

    _, binning, _ = plt.hist(drw_df['log_rf_tau'], bins=np.arange(0.0, 5.0, 0.2),
                             color='tab:blue', edgecolor='k', alpha=0.5, label='all')
    _ = plt.hist(filled_log_rf_tau, bins=binning, color='tab:blue', edgecolor='k', alpha=1.0, label='-27 < $M_i$ < -26')
    plt.ylabel('count')
    plt.xlabel('log(tau/days)')
    plt.legend()
    
def plot_sf_inf_hist(drw_df):
    filled_sf = 10.0**drw_df[(drw_df['M_i_kcorr'] > -27.0) & (drw_df['M_i_kcorr'] < -26.0)]['log_sf_inf'].values

    _, binning, _ = plt.hist(10.0**drw_df['log_sf_inf'].values, bins=np.arange(0.0, 1.0, 0.02),
                             color='tab:blue', edgecolor='k', alpha=0.5, label='all')
    _ = plt.hist(filled_sf, bins=binning, color='tab:blue', edgecolor='k', alpha=1.0, label='-27 < $M_i$ < -26')
    plt.ylabel('count')
    plt.xlabel('SF_inf (mag)')
    plt.legend()