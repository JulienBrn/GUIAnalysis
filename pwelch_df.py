from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2, RessourceHandle
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
import input_df
import clean_df

logger=logging.getLogger(__name__)

def parse_param(param_str: str):
   if param_str.lower() == "true" or param_str.lower()=="false":
    return param_str.lower() == "true"
   try:
        return float(param_str)
   except ValueError:
        return param_str

class pwelchDataDF:
   
  def __init__(self, computation_m, step_signal, lfpDF, buaDF):
     self.computation_m = computation_m
     self.lfpDF = lfpDF
     self.buaDF = buaDF
     self.metadata = {
      "pwelch.window_duration":"3",
      **self.lfpDF.metadata,
      **self.buaDF.metadata,
      } 
     self.invalidated = True
     self.step_signal = step_signal

  
  name = "pwelch"
  result_columns = ["welch_f", "welch_pow"]

  def get_df(self):
    if self.invalidated:
      self.lfpDF.get_df()
      self.buaDF.get_df()
      signal_df = self.step_signal["bua"]
      pwelch_params = {key[7:].replace(".", "_"):parse_param(val) for key,val in self.metadata.items() if "pwelch." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, pwelch_params)
      self.invalidated = False
    return self._dataframe



  def view_item(self, canvas, row):
    
    canvas.ax = canvas.fig.subplots(1)
    y = row["welch_pow"].get_result()
    x = row["welch_f"].get_result()
    canvas.ax.plot(x, y)
    canvas.ax.set_xlabel("Frequency (Hz)")
    canvas.ax.set_ylabel("Amplitude (?)")
    

      
  
  def view_items(self, canvas, row_indices):
     pass
  


def _get_df(computation_m, signal_df, pwelch_params):
  pwelch_df = signal_df[signal_df["signal_type"].isin(["lfp_cleaned", "bua_cleaned"])].copy()

  for key,val in pwelch_params.items():
    pwelch_df[key] = val

  def pwelch(signal, signal_fs, window_duration):
    return scipy.signal.welch(signal, signal_fs, nperseg=window_duration*signal_fs)

  pwelch_df = mk_block(pwelch_df, ["signal", "signal_fs", "window_duration"], pwelch, 
                      {0: (np_loader, "welch_f", True), 1: (np_loader, "welch_pow", True)}, computation_m)
  return pwelch_df

def create_summary_figure(raw, cleaned, fs, bounds, down_sampling, deviation_factor, f):
  summary = bounds
  df=pd.DataFrame()

  df["raw"] = raw
  df["cleaned"] = cleaned
  df["t"] = df.index/fs

  mean = raw.mean()
  rawmin= raw.min()
  rawmax= raw.max()

  cleaned_mean = cleaned.mean()
  cleaned_std = cleaned.std()
  deviation = deviation_factor
  std = raw.std()
  ds= down_sampling


  ax = f.subplots(3)

  ax[0].plot(df["t"][::ds], df["raw"][::ds], label="raw", color="C0")
  ax[0].legend(loc='upper right')
  ax[0].set_xlabel("time (s)")
  ax[0].set_ylabel("signal value")

  ax[1].plot(df["t"][::ds], df["raw"][::ds], label="raw", color="C0")
  ax[1].hlines([mean-std*deviation, mean+std*deviation], 0, df["t"].iat[-1], color="C1", label="raw mean +/- {}*std".format(deviation))
  ax[1].hlines([cleaned_mean-cleaned_std*deviation, cleaned_mean+cleaned_std*deviation], 0, df["t"].iat[-1], color="C4", label="cleaned mean +/- {}*std".format(deviation))

  ax[1].vlines(summary["start"]/fs, rawmin, rawmax, color="C2", label="start artefact")
  ax[1].vlines(summary["end"]/fs, rawmin, rawmax, color="C3", label="end artefact")
  ax[1].legend(loc='upper right')
  ax[1].set_xlabel("time (s)")
  ax[1].set_ylabel("signal value")


  ax[2].plot(df["t"][::ds], df["cleaned"][::ds], color="C1", label="cleaned")
  ax[2].legend(loc='upper right')
  ax[2].set_xlabel("time (s)")
  ax[2].set_ylabel("signal value")
  return ax