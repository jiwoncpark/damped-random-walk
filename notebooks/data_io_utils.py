import os, sys
import sqlite3
import numpy as np
import pandas as pd
from pandas_utils import downcast_numeric
import GCRCatalogs

def join_agn_with_cosmodc2_in_chunks(agn_db_path='/global/projecta/projectdirs/lsst/groups/SSim/DC2/cosmoDC2_v1.1.4/agn_db_mbh7_mi30_sf4.db', save_dir='data'):
    """Save a join between cosmoDC2 and the AGN db in chunks
    
    Parameters
    ----------
    agn_db_path : str
        path to the AGN db. Default: '/global/projecta/projectdirs/lsst/groups/SSim/DC2/cosmoDC2_v1.1.4/agn_db_mbh7_mi30_sf4.db'
    save_dir : str
        directory into which to save the data. Default: 'data'
    
    """
    agn_chunks = read_agn_params_in_chunks(agn_db_path)
    num_agn_chunks = len(agn_chunks)
    cosmodc2 = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_image') # 35s in Jupyter-dev
    for chunk_id in [50,]:#range(num_agn_chunks):
        joined_df = join_agn_with_cosmodc2(agn_chunks[chunk_id], cosmodc2)
        # Optimize memory usage
        joined_df = downcast_numeric(joined_df)
        save_file = os.path.join(save_dir, 'joined_%d.csv' %chunk_id)
        joined_df.to_csv(save_file)
    return None

def read_agn_params_in_chunks(agn_db_path):
    """Read in the AGN db in chunks of size 200K and instantiate Pandas DataFrames with them
    
    Parameters
    ----------
    agn_db_path : str
        path to the AGN db.
    
    Returns
    -------
    list
        list of Pandas DataFrames corresponding to each chunk
    
    """
    conn = sqlite3.connect(agn_db_path)
    # See which tables the db file has <-- only has 'agn_params'
    #cursor = conn.cursor()
    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print(cursor.fetchall())
    # Create list of agn_params table in Pandas DataFrame format
    agn_df_gen = pd.read_sql(sql='SELECT * from agn_params', con=conn, chunksize=200000)
    agn_chunks = list(agn_df_gen)
    return agn_chunks
        
def join_agn_with_cosmodc2(agn_df, loaded_cosmodc2):
    """Join the AGN db with cosmoDC2
    
    Parameters
    ----------
    agn_df : Pandas DataFrame
        table of AGN parameters
    loaded_cosmodc2 : dict
        cosmoDC2 loaded with GCRCatalogs
    
    Returns
    -------
    Pandas DataFrame
        the joined table
        
    """
    
    agn_galaxy_ids = agn_df['galaxy_id'].values
    agn_df = unravel_dictcol(agn_df)
    quantities = ['galaxy_id', 'redshift',
                  'blackHoleAccretionRate', 'blackHoleEddingtonRatio', 'blackHoleMass',]
    #quantities += ['mag_true_%s_sdss' %bp for bp in 'ugriz']
    galaxy_id_min = np.min(agn_galaxy_ids)
    galaxy_id_max = np.max(agn_galaxy_ids)
    filters = ['galaxy_id >= %d' %(galaxy_id_min),
               'galaxy_id <= %d' %(galaxy_id_max)]
    cosmodc2_obj = loaded_cosmodc2.get_quantities(quantities, filters=filters)
    cosmodc2_df = pd.DataFrame(cosmodc2_obj)
    joined = pd.merge(cosmodc2_df, agn_df, on='galaxy_id')
    return joined

def unravel_dictcol(agn_df):
    """Unravel the json-type column of agn params (time-consuming and much memory overhead!)
    
    Parameters
    ----------
    agn_df : Pandas DataFrame
        raw AGN db read into a Pandas DataFrame
    
    Returns
    -------
    Pandas DataFrame
        version of `agn_df` with the `agn_params` column unraveled
    
    """
    # Unravel the string into dictionary
    agn_df['varParamStr'] = agn_df['varParamStr'].apply(eval)
    # Convert the dictionary inside key 'p' of the 'varParamStr' dictionary into columns
    agn_params_df = (agn_df['varParamStr'].apply(pd.Series))['p'].apply(pd.Series)
    # Combine the agn parameters with the original df containing galaxy_id and magNorm
    agn_df = pd.concat([agn_df.drop(['varParamStr'], axis=1), agn_params_df], axis=1)
    return agn_df
