from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2, RessourceHandle
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
import analysisGUI.input_df as input_df
import analysisGUI.clean_df as clean_df

logger=logging.getLogger(__name__)

def parse_param(param_str: str):
   if param_str.lower() == "true" or param_str.lower()=="false":
    return param_str.lower() == "true"
   try:
        return float(param_str)
   except ValueError:
        return param_str

class SpikeContinuousDataDF:
   
  def __init__(self, computation_m, step_signal, inputDF):
     self.computation_m = computation_m
     self.inputDF = inputDF
     self.metadata = {
      "spike_bin.size":str(1.0/1000),
      **self.inputDF.metadata,
      } 
     self.invalidated = True
     self.step_signal = step_signal

  
  name = "spike_bin"
  result_columns = ["spike_sig"]

  def get_df(self):
    if self.invalidated:
      self.inputDF.get_df()
      signal_df = self.step_signal["input"]
      spike_params = {key[10:].replace(".", "_"):parse_param(val) for key,val in self.metadata.items() if "spike_bin." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, spike_params)
      signal_append = self._dataframe.copy().drop(columns=["spike_sig", "signal", "signal_type", "signal_fs"]+list(spike_params.keys()))
      signal_append["signal"] = self._dataframe["spike_sig"]
      signal_append["signal_type"] = "spike_continuous"
      signal_append["signal_fs"] = 1.0/self._dataframe["size"]
      self.step_signal["spike_continuous"] = signal_append
      self.invalidated = False
    return self._dataframe



  def view_item(self, canvas, row):
    
    canvas.ax = canvas.fig.subplots(2, 1, sharex="all")
    y_source = row["signal"].get_result()
    
    canvas.ax[0].eventplot(y_source/row["signal_fs"])
    canvas.ax[0].set_xlabel("Time (s)")
    # canvas.ax[0].set_ylabel("Has Spike")

    y_spike = row["spike_sig"].get_result()
    x_spike = np.arange(0, y_spike.shape[0]*row["size"], row["size"])
    canvas.ax[1].plot(x_spike, y_spike)
    canvas.ax[1].set_xlabel("Time (s)")
    # canvas.ax[1].set_ylabel("Amplitude (?)")
    

      
  
  # def view_items(self, canvas, row_indices):
  #    pass
  


def _get_df(computation_m, signal_df, spike_params):
  spike_df = signal_df[signal_df["signal_type"].isin(["spike_times"])].copy()

  for key,val in spike_params.items():
   spike_df[key] = val

  def make_continuous(signal: np.array, signal_fs, size):
    out_fs = 1.0/size
    if signal.size <5:
       return np.nan
    new_size = int(np.max(signal)*out_fs/signal_fs)+2
    if new_size < 100:
       logger.warning("Make continuous, new size <100, got {}, with new_fs = {}, in_fs ={}, max = {}".format(new_size, out_fs, signal_fs, np.max(signal)))
    zeros = np.zeros(new_size)
    indexes = (signal *out_fs/signal_fs).astype(int)
    np.add.at(zeros, indexes, 1)
    return zeros

  tqdm.pandas(desc="Declaring continuous spike signals")
  spike_df = mk_block(spike_df, ["signal", "signal_fs"] + list(spike_params.keys()), make_continuous, 
              (np_loader, "spike_sig", False), computation_m) 

  return spike_df

