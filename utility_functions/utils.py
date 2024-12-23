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

