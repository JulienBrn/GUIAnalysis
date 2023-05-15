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

class LFPDataDF:
   
  def __init__(self, computation_m, step_signal, cleanDF):
     self.computation_m = computation_m
     self.cleanDF = cleanDF
     self.metadata = {
      "lfp.lowpass.filter_freq":"200",
      "lfp.out_fs": "500",
      "lfp.lowpass.order": "3",
      **self.cleanDF.metadata,
      } 
     self.invalidated = True
     self.step_signal = step_signal

  
  name = "lfp"
  result_columns = ["lfp_sig", "lfp_fs"]

  def get_df(self):
    if self.invalidated:
      self.cleanDF.get_df()
      signal_df = self.step_signal["clean"]
      lfp_params = {key[4:].replace(".", "_"):parse_param(val) for key,val in self.metadata.items() if "lfp." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, lfp_params)
      if len(self._dataframe.index) !=0:
        signal_append = self._dataframe.copy().drop(columns=["lfp_sig", "lfp_fs", "signal", "signal_type", "signal_fs"]+list(lfp_params.keys()))
        signal_append["signal"] = self._dataframe["lfp_sig"]
        signal_append["signal_type"] = self._dataframe["signal_type"].str.replace("raw_", "lfp_", regex=False)
        signal_append["signal_fs"] = self._dataframe["lfp_fs"]
        self.step_signal["lfp"] = signal_append
      else:
        self.step_signal["lfp"] = pd.DataFrame()
      self.invalidated = False
    return self._dataframe



  def view_item(self, canvas, row):
    
    canvas.ax = canvas.fig.subplots(2, 1, sharex="all", sharey="all")
    y_source = row["signal"].get_result()
    x_source = np.arange(0, y_source.shape[0]/row["signal_fs"], 1.0/row["signal_fs"])
    canvas.ax[0].plot(x_source, y_source)
    canvas.ax[0].set_xlabel("Time (s)")
    canvas.ax[0].set_ylabel("Amplitude (?)")

    y_lfp = row["lfp_sig"].get_result()
    x_lfp = np.arange(0, y_lfp.shape[0]/row["lfp_fs"].get_result(), 1.0/row["lfp_fs"].get_result())
    canvas.ax[1].plot(x_lfp, y_lfp)
    canvas.ax[1].set_xlabel("Time (s)")
    canvas.ax[1].set_ylabel("Amplitude (?)")
    

      
  
  # def view_items(self, canvas, row_indices):
  #    pass
  


def _get_df(computation_m, signal_df, lfp_params):

  lfp_df = signal_df[(signal_df["signal_type"].isin(["raw_cleaned"]))].copy()

  for key,val in lfp_params.items():
    lfp_df[key] = val

  def extract_lfp(signal, signal_fs, lowpass_filter_freq, out_fs, lowpass_order):
    lfp, out_fs = toolbox.extract_lfp(signal, signal_fs, lowpass_filter_freq, out_fs, lowpass_order)
    return (lfp, out_fs)
  
  tqdm.pandas(desc="Declaring lfp signals")
  lfp_df = mk_block(lfp_df, ["signal", "signal_fs", "lowpass_filter_freq","out_fs", "lowpass_order"], extract_lfp, 
             {0: (np_loader, "lfp_sig", True), 1: (float_loader, "lfp_fs", True)}, computation_m)
  return lfp_df


