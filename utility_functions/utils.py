import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import random
from scipy.stats import norm 
from scipy.stats import truncnorm

def rounder_threshhold(df, cols):
  """Rounds values in specified columns of a DataFrame to 0 or 1 based on a threshold of 0.5.
  Args:
      df: The DataFrame to modify .
      cols: A list of column names to round.
  """

  df_rounded = df.copy()

  for col in cols:
    df_rounded[col] = df_rounded[col].apply(lambda val: 1 if val > 0.5 else 0)

  return df_rounded

def rounder(df, cols, dp):
  """Rounds values in specified columns of a DataFrame.
  Args:
      df: The DataFrame to modify .
      cols: A list of column names to round.
      dp:decimal points
  """

  df_rounded = df.copy()

  for col in cols:
    df_rounded[col] = df_rounded[col].apply(lambda val: np.round(val,dp))

  return df_rounded

def make_iid(df, identifier, seq_len):
    """Make the dataset i.i.d.
    Args:
      df: The DataFrame to modify .
      identifier: ID column.
      seq_len:length of sequences
    """
    
    temp_data = []
    grouped = df.groupby(identifier)
    for _, group in grouped:
        for i in range(0, len(group), seq_len):
            _x = group.iloc[i:i + seq_len]
            temp_data.append(_x)
    idx = np.random.permutation(len(temp_data))  
    shuffled_data = [temp_data[i] for i in idx]  
 
    result_df = pd.concat(shuffled_data).reset_index(drop=True)
    return result_df



def convert_to_trun(df, cols, seq_len, cumulative_probs):
    for col in cols:
        for i in range(0, len(df), seq_len):
            group = df.loc[i:i+seq_len-1, col]

            for j in range(len(group)):
                value = group.iloc[j]
                if value == 0:
                    lower_bound = 0
                    upper_bound = cumulative_probs[col][0]
                elif value == 1:
                    lower_bound = cumulative_probs[col][0]
                    upper_bound = 1
                else:
                    continue  

                mean = (lower_bound + upper_bound) / 2
                std = (upper_bound - lower_bound) / 2 
                a = (lower_bound - mean) / std
                b = (upper_bound - mean) / std

                value_sampled = truncnorm.rvs(a, b, loc=mean, scale=std)

                df.loc[i + j, col] = value_sampled
    
    return df

def convert_to_truncated(df, cols):
    intervals_dict = {}
    for col in cols:
        value_counts = df[col].value_counts(normalize=True).sort_values(ascending=False)

        cumulative_prob = value_counts.cumsum()
     
        intervals = {}
        current_lower = 0

        for category, prob in cumulative_prob.items():
            current_upper = prob
            intervals[category] = (current_lower, current_upper)
            current_lower = current_upper
            
        intervals_dict[col] = intervals

        df[col] = df[col].apply(lambda x: sample_from_truncated_gaussian(intervals[x]))

    return df, intervals_dict
    
def sample_from_truncated_gaussian(interval):
    lower_bound, upper_bound = interval
    mean = (lower_bound + upper_bound) / 2
    std = (upper_bound - lower_bound) / 2
    a = (lower_bound - mean) / std
    b = (upper_bound - mean) / std
    
    
    value_sampled = truncnorm.rvs(a, b, loc=mean, scale=std)
    
    return value_sampled   


def reverse_truncation(df, cols, intervals_dict):
    for col in cols:
        intervals = intervals_dict[col]

        def map_value_to_category(value):
            for category, (lower_bound, upper_bound) in intervals.items():
                if lower_bound <= value <= upper_bound:
                    return category
            return None  

        df[col] = df[col].apply(map_value_to_category)
    
    return df

def MinMaxScaler(data):
    
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    

    norm_data = (data - min_val) / (max_val - min_val + 1e-7)
      
    return norm_data, min_val, max_val


def standardize(data, to_exc):
    cols_to_exclude = to_exc
    cols_to_keep = data[cols_to_exclude]
    data_to_standardize = data.drop(columns=cols_to_exclude)

    standardized_data, min_val, max_val = MinMaxScaler(data_to_standardize)

    standardized_df = pd.DataFrame(standardized_data, columns=data_to_standardize.columns)

    result = pd.concat([cols_to_keep, standardized_df], axis=1)

    return result, min_val, max_val


def reverse_min_max_scaling(data, to_exclude,max_val, min_val):
    col_to_keep = data[to_exclude]
    data_to_reverse = data.drop(columns=to_exclude)
    
    original_data = data_to_reverse * (max_val - min_val) + min_val
    
    result = pd.concat([col_to_keep, original_data], axis=1)
    
    return result
