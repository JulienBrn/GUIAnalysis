from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2, RessourceHandle
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
import analysisGUI.input_df as input_df

logger=logging.getLogger(__name__)

def parse_param(param_str: str):
   if param_str.lower() == "true" or param_str.lower()=="false":
    return param_str.lower() == "true"
   try:
        return float(param_str)
   except ValueError:
        return param_str

class CleanDataDF:
   
  def __init__(self, computation_m, step_signal, inputDF):
     self.computation_m = computation_m
     self.inputDF = inputDF
     self.metadata = {
      "clean.deviation_factor": "5",
      "clean.min_length": "0.003",
      "clean.join_width": "3",
      "clean.shoulder_width": "1",
      "clean.recursive": "True",
      "clean.invalidate_all": "False",
      "clean.replace_type": "affine",
      **self.inputDF.metadata,
      } 
     self.invalidated = True
     self.step_signal = step_signal
    #  self._dataframe = self._get_df(dataframe_manager, computation_m)

  
  name = "clean"
  result_columns = ["clean_bounds", "cleaned_signal"]

  def get_df(self):
    if self.invalidated:
      self.inputDF.get_df()
      signal_df = self.step_signal["input"]
      clean_params = {key[6:]:parse_param(val) for key,val in self.metadata.items() if "clean." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, clean_params)
      signal_append = self._dataframe.copy().drop(columns=["clean_bounds", "cleaned_signal", "signal", "signal_type"]+list(clean_params.keys()))
      signal_append["signal"] = self._dataframe["cleaned_signal"]
      signal_append["signal_type"] = self._dataframe["signal_type"]+"_cleaned"
      self.step_signal["clean"] = signal_append
      self.invalidated = False
    return self._dataframe

  def compute(self):
    self.get_df()
    tqdm.pandas(desc="Compute coherence_df results") 
    def compute_elem(row):
      for col in self.result_columns:
        if isinstance(row[col], RessourceHandle):
          row[col].get_result()
    self._dataframe.progress_apply(lambda row: compute_elem, axis=1, result_type="reduce")

  def view_item(self, canvas, row):
    params = [row["signal"], row["cleaned_signal"], row["signal_fs"], row["clean_bounds"].get_result(), 100, row["deviation_factor"], canvas.fig]
    canvas.ax = create_summary_figure(*[p.get_result() if isinstance(p, RessourceHandle) else p for p in params])

      
  
  # def view_items(self, canvas, row_indices):
  #    pass
  


def _get_df(computation_m, signal_df, clean_params):

  cleaned_df = signal_df[signal_df["signal_type"].isin(["raw", "mua"])].copy()
  
  for key,val in clean_params.items():
    cleaned_df[key] = val

  def clean(signal, signal_fs, deviation_factor, min_length, join_width, recursive, shoulder_width):
      bounds = toolbox.compute_artefact_bounds(signal, signal_fs, deviation_factor, min_length, join_width, recursive, shoulder_width)
      return pd.DataFrame(bounds, columns=["start", "end"])

  tqdm.pandas(desc="Declaring clean bounds")
  cleaned_df = mk_block(cleaned_df, ["signal", "signal_fs", "deviation_factor", "min_length", "join_width", "recursive", "shoulder_width", "clean_version"], clean,
                              (df_loader, "clean_bounds", True), computation_m)


  def generate_clean(signal, clean_bounds, replace_type):
    filtered= signal.copy().astype(float)
    for _,artefact in clean_bounds.iterrows():
      s = artefact["start"]
      e = artefact["end"]
      filtered[s:e] = np.nan
    if replace_type == "affine":
      return toolbox.affine_nan_replace(filtered)
    elif replace_type == "nan":
      return filtered
    else:
      raise BaseException("Invalid replace type")
    
  tqdm.pandas(desc="Declaring clean signal")
  cleaned_df = mk_block(cleaned_df, ["signal", "clean_bounds", "replace_type"], generate_clean, (np_loader, "cleaned_signal", False), computation_m)  
  return cleaned_df



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


  ax = f.subplots(3, sharex="all", sharey="all")

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
