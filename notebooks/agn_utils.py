import os, sys
import sqlite3
import numpy as np
import pandas as pd
from pandas_utils import downcast_numeric
from lsst.utils import getPackageDir
from lsst.sims.photUtils import Sed, BandpassDict, Bandpass, CosmologyObject
from lsst.sims.utils import findHtmid
os.environ['SIMS_GCRCATSIMINTERFACE_DIR'] = '/global/u1/j/jwp/DC2/sims_GCRCatSimInterface'
sys.path.append('/global/u1/j/jwp/DC2/sims_GCRCatSimInterface/python')
from desc.sims.GCRCatSimInterface import M_i_from_L_Mass, log_Eddington_ratio, k_correction, tau_from_params, SF_from_params
import GCRCatalogs

def create_k_corr_grid(redshift):
    """
    (Edited from original because of error)
    Returns a grid of redshifts and K corrections on the
    LSST Simulations AGN SED that can be used for K correction
    interpolation.
    """
    bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
    bp_i = bp_dict['i']
    sed_dir = os.path.join(getPackageDir('sims_sed_library'),
                           'agnSED')
    sed_name = os.path.join(sed_dir, 'agn.spec.gz')
    if not os.path.exists(sed_name):
        raise RuntimeError('\n\n%s\n\nndoes not exist\n\n' % sed_name)
    base_sed = Sed()
    base_sed.readSED_flambda(sed_name)
    z_grid = np.arange(0.0, redshift.max(), 0.01)
    k_grid = np.zeros(len(z_grid),dtype=float)
    for i_z, zz in enumerate(z_grid):
        ss = Sed(flambda=base_sed.flambda, wavelen=base_sed.wavelen)
        ss.redshiftSED(zz, dimming=True)
        k = k_correction(ss, bp_i, zz)
        k_grid[i_z] = k
    return z_grid, k_grid

def get_m_i(abs_mag_i, redshift):
    """
    (Edited from original because of error)
    Take numpy arrays of absolute i-band magnitude and
    cosmological redshift.  
    
    Returns
    -------
    array-like
        observed i-band magnitudes
    """
    z_grid, k_grid = create_k_corr_grid(redshift)
    k_corr = np.interp(redshift, z_grid, k_grid)

    dc2_cosmo = CosmologyObject(H0=71.0, Om0=0.265)
    distance_modulus = dc2_cosmo.distanceModulus(redshift=redshift)
    obs_mag_i = abs_mag_i + distance_modulus + k_corr
    return obs_mag_i

def add_columns(joined):
    black_hole_mass = joined['blackHoleMass'].values
    edd_ratio = joined['blackHoleEddingtonRatio'].values
    redshift = joined['redshift'].values
    z_corr = 1.0/(1.0 + redshift)
    wavelength_norm = 4000.0 # Angstroms

    joined['M_i'] = M_i_from_L_Mass(np.log10(edd_ratio), np.log10(black_hole_mass))
    joined['rf_u'] = 3520.0*z_corr/wavelength_norm
    joined['rf_g'] = 4800.0*z_corr/wavelength_norm
    joined['rf_r'] = 6250.0*z_corr/wavelength_norm
    joined['rf_i'] = 7690.0*z_corr/wavelength_norm
    joined['rf_z'] = 9110.0*z_corr/wavelength_norm
    joined['m_i'] = get_m_i(joined['M_i'].values, redshift)
    return joined