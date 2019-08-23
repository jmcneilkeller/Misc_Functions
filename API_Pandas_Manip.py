import pandas as pd
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)


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
