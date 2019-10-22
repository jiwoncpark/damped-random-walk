import pandas as pd
import numpy as np

def mem_usage(pandas_obj):
    # https://www.dataquest.io/blog/pandas-big-data/
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def downcast_numeric(big_df):
    # Downcast integer columns
    df_int = big_df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')
    print("Integer downcasting")
    print("Before: ", mem_usage(df_int))
    print("After: ", mem_usage(converted_int))
    
    # Downcast float columns
    df_float = big_df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric, downcast='float')
    print("Float downcasting")
    print("Before: ", mem_usage(df_float))
    print("After: ", mem_usage(converted_float))
    
    small_df = big_df.copy()
    small_df[converted_int.columns] = converted_int
    small_df[converted_float.columns] = converted_float
    print("Overall summary")
    print("Before: ", mem_usage(big_df))
    print("After: ", mem_usage(small_df))
    return small_df