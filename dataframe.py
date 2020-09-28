import pandas as pd
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
import glob
import numpy as np


def df_dict_to_col(df_column):
    # Takes a DataFrame column where the values are dictionaries. Returns a new DataFrame for concatenation to original DF.
    # Converts DF column to a dict.
    temp = df_column.to_dict()
    # Converts back to a new DF. Indexing by value.
    new_df = pd.DataFrame.from_dict(temp, orient='index')
    return new_df

def get_sample_replace(data, n):
    #Pulls a random sample from data in a dataframe or series.
    #n=size of the sample.
    #This is with replacement.
    #Random state seeds the sample generation for reproducibility.
    sample = data.sample(n, replace=True, random_state=1)
    return sample

df_master['zone_total_trips'] = df_master.groupby('zones')['count'].transform('sum')

# Concatenate csvs into a dataframe.
globs = glob.glob(r'DIRECTORY_WITH_CSVS)

data = []
for csv in globs:
    frame = pd.read_csv(csv,index_col=None,header=0)
    frame['filename'] = os.path.basename(csv)
    data.append(frame)
df = pd.concat(data,axis=0,ignore_index=True)


# Mask the first observation in each group.
def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    return result

mask = df.groupby(['COL_TO_MASK_BY'])['COL_TO_MASK_BY'].transform(mask_first).astype(bool)

# Return all rows from columns starting with a prefix.
df.loc[:,df.columns.str.startswith('COLUMN_PREFIX')]

# Find any columns where all values are null.
df.columns[df.isnull().all()]
