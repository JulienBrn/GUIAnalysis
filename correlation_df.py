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

class correlationDataDF:
   
  def __init__(self, computation_m, step_signal, spikeDF):
     self.computation_m = computation_m
     self.spikeDF = spikeDF
     self.metadata = {
       "correlation.max":"1",
      **self.spikeDF.metadata,
      } 
     self.invalidated = True
     self.step_signal = step_signal

  
  name = "correlation"
  result_columns = ["correlation_fs", "correlation_pow"]

  def get_df(self):
    if self.invalidated:
      self.spikeDF.get_df()
      signal_df = pd.concat([self.step_signal["spike_continuous"]], ignore_index=True)
      correlation_params = {key[12:].replace(".", "_"):parse_param(val) for key,val in self.metadata.items() if "correlation." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, correlation_params)
      self.invalidated = False
    return self._dataframe



  def view_item(self, canvas, row):
    canvas.ax = canvas.fig.subplots(1)
    canvas.ax.set_xlabel("Time (s)")
    canvas.ax.set_ylabel("Amplitude (?)")
    y = row["correlation_pow"].get_result()
    fs = row["correlation_fs"].get_result()
    if not fs is np.nan and hasattr(y, "shape"):
        x = np.arange(-0.5*y.size/fs, 0.5*y.size/fs, 1.0/fs)
        canvas.ax.plot(x, y)

      
  
  def view_items(self, canvas, rows):
    canvas.ax = canvas.fig.subplots(1)
    canvas.ax.set_xlabel("Time (s)")
    canvas.ax.set_ylabel("Amplitude (?)")
    for i in range(len(rows.index)):
      y = rows["correlation_pow"].iat[i].get_result()
      fs = rows["correlation_fs"].iat[i].get_result()
      if not fs is np.nan and hasattr(y, "shape"):
        x = np.arange(-0.5*y.size/fs, 0.5*y.size/fs, 1.0/fs)
        canvas.ax.plot(x, y)
        
  def compute(self):
    self.get_df()
    tqdm.pandas(desc="Compute coherence_df results") 
    def compute_elem(row):
      for col in self.result_columns:
        if isinstance(row[col], RessourceHandle):
          row[col].get_result()
    self._dataframe.progress_apply(compute_elem, axis=1)



def _get_df(computation_m, signal_df, correlation_params):
  tqdm.pandas(desc="Computing shape of correlation_df") 
  correlation_df = toolbox.group_and_combine(signal_df[signal_df["signal_type"].isin(["spike_continuous"])], ["Condition", "Subject", "Species", "Session", "Date", "SubSessionInfo", "SubSessionInfoType"], include_eq=True)

  for key,val in correlation_params.items():
    correlation_df[key] = val
   
  def compute_correlation(signal_1, signal_fs_1, signal_2, signal_fs_2, max):
    if signal_fs_1 != signal_fs_2 or signal_1.size != signal_2.size:
      logger.warning("The two signals do not have the same fs in compute correlation")
      return np.nan, np.nan
    else:
      mlen = min(signal_1.size, signal_2.size)
      sig1 = signal_1[0:mlen]
      sig2 = signal_2[0:mlen]
      correlation = np.zeros(2*int(max*signal_fs_1))
      if int(max*signal_fs_1) <3:
        logger.warning("strange")
      for i in tqdm(range(correlation.shape[0])):
        if i == max*signal_fs_1:
          correlation[i] = np.nan
        else:
          correlation[i] = np.dot(sig1, np.roll(sig2, int(i-max*signal_fs_1)))
      return signal_fs_1, correlation
   
  tqdm.pandas(desc="Declaring ressources for correlation_df") 
  correlation_df = mk_block(correlation_df, ["signal_1", "signal_fs_1", "signal_2", "signal_fs_2", "max"], compute_correlation, 
                      {0: (np_loader, "correlation_fs", True), 1: (np_loader, "correlation_pow", True)}, computation_m)

  return correlation_df
