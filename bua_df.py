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

class BUADataDF:
   
  def __init__(self, computation_m, step_signal, cleanDF):
     self.computation_m = computation_m
     self.cleanDF = cleanDF
     self.metadata = {
      "bua.bandpass.low_freq":"300",
      "bua.bandpass.high_freq":"6000",
      "bua.lowpass.freq": "1000",
      "bua.passes.order": "3",
      "bua.out_fs": "1000",
      **self.cleanDF.metadata,
      } 
     self.invalidated = True
     self.step_signal = step_signal

  
  name = "bua"
  result_columns = ["bua_sig", "bua_fs"]

  def get_df(self):
    if self.invalidated:
      self.cleanDF.get_df()
      signal_df = self.step_signal["clean"]
      bua_params = {key[4:].replace(".", "_"):parse_param(val) for key,val in self.metadata.items() if "bua." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, bua_params)
      signal_append = self._dataframe.copy().drop(columns=["bua_sig", "bua_fs", "signal", "signal_type", "signal_fs"]+list(bua_params.keys()))
      signal_append["signal"] = self._dataframe["bua_sig"]
      signal_append["signal_type"] = self._dataframe["signal_type"].str.replace("raw_", "bua_").str.replace("mua_", "bua_")
      signal_append["signal_fs"] = self._dataframe["bua_fs"]
      self.step_signal["bua"] = pd.concat([signal_df, signal_append], ignore_index=True)
      self.invalidated = False
    return self._dataframe



  def view_item(self, canvas, row):
    
    canvas.ax = canvas.fig.subplots(2, 1, sharex="all", sharey="all")
    y_source = row["signal"].get_result()
    x_source = np.arange(0, y_source.shape[0]/row["signal_fs"], 1.0/row["signal_fs"])
    canvas.ax[0].plot(x_source, y_source)
    canvas.ax[0].set_xlabel("Time (s)")
    canvas.ax[0].set_ylabel("Amplitude (?)")

    y_bua = row["bua_sig"].get_result()
    x_bua = np.arange(0, y_bua.shape[0]/row["bua_fs"].get_result(), 1.0/row["bua_fs"].get_result())
    canvas.ax[1].plot(x_bua, y_bua)
    canvas.ax[1].set_xlabel("Time (s)")
    canvas.ax[1].set_ylabel("Amplitude (?)")
    

      
  
  def view_items(self, canvas, row_indices):
     pass
  


def _get_df(computation_m, signal_df, bua_params):
  bua_df = signal_df[signal_df["signal_type"].isin(["raw_cleaned", "mua_cleaned"])].copy()

  for key,val in bua_params.items():
    bua_df[key] = val

  def extract_bua(signal, signal_fs, bandpass_low_freq, bandpass_high_freq, lowpass_freq, out_fs, passes_order):
    bua, out_fs = toolbox.extract_mu(signal, signal_fs, bandpass_low_freq, bandpass_high_freq, lowpass_freq, out_fs, passes_order)
    return (bua, out_fs)

  bua_df = mk_block(bua_df, ["signal", "signal_fs"] + list(bua_params.keys()), extract_bua, 
             {0: (np_loader, "bua_sig", True), 1: (float_loader, "bua_fs", True)}, computation_m) 

  return bua_df

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