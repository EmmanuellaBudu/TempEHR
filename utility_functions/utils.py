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

def sample_from_vae(vae_model, num_samples, batch_size, seq_len, device):
    vae_model.to(device)
    vae_model.eval()

    with torch.no_grad():
        synthetic_samples = torch.zeros(num_samples, seq_len, vae_model.input_size + 1).to(device)
        
        for i in range((num_samples + batch_size - 1) // batch_size):
    
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
          
            z = torch.randn(current_batch_size, vae_model.latent_size).to(device)
            noise = torch.randn_like(z) * 0.2 
         
            h_ = vae_model.fc3(z)
            h_s = h_.unsqueeze(0).repeat(vae_model.lstm_dec.num_layers, 1, 1)
            hidden = (h_s.contiguous(), h_s.contiguous())

           
            time = torch.distributions.Exponential(1 / mean_time).sample((current_batch_size, seq_len)).to(device)

           
            latent = z.unsqueeze(1).expand(-1, seq_len, -1)

            
            synthetic_sample, time_out, _ = vae_model.lstm_dec(latent, time, hidden, autoregressive=True)

            combined_output = torch.cat((synthetic_sample, time_out), dim=2)

            start_idx = i * batch_size
            end_idx = start_idx + current_batch_size
            synthetic_samples[start_idx:end_idx] = combined_output

    return synthetic_samples.cpu().numpy()

