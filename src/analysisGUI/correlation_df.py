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



  def get_nb_figs(self, rows):
    if len(rows.index) < 6:
      return 1
    else:
      df = rows.copy()
      df["Condition_b"] = df.apply(lambda row: "Park" if row["Condition"] in ["pd", "Park", "mptp"] else "healthy", axis=1)
      return  df.groupby(["Condition_b", "signal_type_1", "signal_type_2", "is_auto"]).ngroups*2 + df.groupby(["Species", "signal_type_1", "signal_type_2", "Condition_b", "is_auto"]).ngroups

  def show_figs(self, rows, canvas_list):
    if len(rows.index) < 6:
      canvas = canvas_list[0]
      canvas.ax = canvas.fig.subplots(1)
      canvas.ax.set_xlabel("Time (s)")
      canvas.ax.set_ylabel("Amplitude (?)")
      for i in range(len(rows.index)):
        y = rows["correlation_pow"].iat[i].get_result()
        fs = rows["correlation_fs"].iat[i].get_result()
        if not fs is np.nan and hasattr(y, "shape") and y.size>10:
          x = np.arange(-0.5*y.size/fs, 0.5*y.size/fs, 1.0/fs)
          canvas.ax.plot(x, y)
      canvas.ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
      canvas.fig.tight_layout()
      yield([0])
    else:
      curr_canvas = 0
      df = rows.copy().reset_index()
      tqdm.pandas(desc="Computing ressources for correlation_df") 
      df["correlation_pow"] = df.progress_apply(lambda row: row["correlation_pow"].get_result(), axis=1, result_type="reduce")
      def get_times(row):
        fs = row["correlation_fs"].get_result()
        import math
        if (not fs) or fs is math.nan or np.isnan(fs) or isinstance(fs, str):
          # logger.info("ignored fs = {}".format(fs))
          return np.nan
        # else:
        #   logger.info("fs = {}, type={}, size={}".format(fs, type(fs), fs.size))
        n = row["correlation_pow"].size
        return np.arange(-0.5*n/fs, 0.5*n/fs, 1.0/fs)
      df["correlation_t"] = df.progress_apply(get_times, axis=1, result_type="reduce")
      df["Condition_b"] = df.apply(lambda row: "Park" if row["Condition"] in ["pd", "Park", "mptp"] else "healthy", axis=1)
      df["aggregation_type"] = "plot"
      na_mask = df.apply(lambda row: not hasattr(row["correlation_t"], "size") or row["correlation_t"].size < 20, axis=1)
      df = df[~na_mask].reset_index(drop=True)
      if na_mask.any():
        logger.warning("Ignored {} results due to na".format(na_mask.sum()))
      else:
        logger.info("all results good")

      df["correlation_pow"] = df.apply(lambda row: scipy.stats.zscore(row["correlation_pow"], nan_policy="omit"), axis=1, result_type="reduce")

      tqdm.pandas(desc="Adding group information for correlation_df") 
      r = df.groupby(by=["Species",  "Condition_b", "Structure_1", "signal_type_1", "Structure_2", "signal_type_2", "is_auto"]).progress_apply(compute_group_info).reset_index()
      
      
 
      def by_structure():
        nonlocal curr_canvas, r
        df = r[~r["aggregation_type"].str.contains("best")].reset_index(drop=True)
        toolbox.add_draw_metadata(df, fig_group = ["aggregation_type", "Condition_b", "signal_type_1", "signal_type_2", "is_auto"], col_group=["Structure_1"], row_group=["Structure_2"], color_group=["Species", "n"])
        nb_figs = df.groupby(["aggregation_type", "Condition_b", "signal_type_1", "signal_type_2", "is_auto"]).ngroups
        p = toolbox.prepare_figures2(df, [canvas.fig for canvas in canvas_list[curr_canvas:curr_canvas+nb_figs]])
        curr_canvas+=nb_figs
        p.plot2(df, x="correlation_t", y="correlation_pow", use_zscore=False)
        return list(range(curr_canvas-nb_figs, curr_canvas))
      

      def details_pow():
        nonlocal curr_canvas, df, r
        df = pd.concat([df, r[~r["aggregation_type"].str.contains("best")]], ignore_index=True)
        toolbox.add_draw_metadata(df, fig_group = ["Species", "signal_type_1", "signal_type_2", "Condition_b", "is_auto"], col_group=["Structure_1"], row_group=["Structure_2"], color_group=["aggregation_type"])
        nb_figs = df.groupby(["Species", "signal_type_1", "signal_type_2", "Condition_b", "is_auto"]).ngroups
        p = toolbox.prepare_figures2(df, [canvas.fig for canvas in canvas_list[curr_canvas:curr_canvas+nb_figs]])
        curr_canvas+=nb_figs
        p.plot2(df, x="correlation_t", y="correlation_pow", use_zscore=False)
        return list(range(curr_canvas-nb_figs, curr_canvas))
      
      # def details_phase():
      #   nonlocal curr_canvas, r
      #   df = r[r["aggregation_type"].str.contains("best")].reset_index(drop=True)
      #   df["f"] = df["coherence_f"]
      #   toolbox.add_draw_metadata(df, fig_group = ["Species", "signal_type_1", "signal_type_2", "Condition_b"], col_group=["Structure_1"], row_group=["Structure_2"], color_group=["aggregation_type", "f", "n"])
      #   nb_figs = df.groupby(["Species", "signal_type_1", "signal_type_2", "Condition_b"]).ngroups
      #   p = toolbox.prepare_figures2(df, [canvas.fig for canvas in canvas_list[curr_canvas:curr_canvas+nb_figs]], projection="polar", ylim=[0, 1], ignore_legend=True)
      #   curr_canvas+=nb_figs
      #   p.scatter_plot_vlines(df, x="coherence_phase", y="coherence_pow", use_zscore=False)
      #   return list(range(curr_canvas-nb_figs, curr_canvas))
      
      yield(by_structure())
      yield(details_pow())
      # yield(details_phase())






















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
      if not fs is np.nan and hasattr(y, "shape") and y.size>10:
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
  correlation_df["is_auto"] = correlation_df["signal_1"] == correlation_df["signal_2"]
  for key,val in correlation_params.items():
    correlation_df[key] = val
   
  def compute_correlation(signal_1, signal_fs_1, signal_2, signal_fs_2, max, is_auto):
    if not hasattr(signal_1, "size") or not hasattr(signal_2, "size"):
      return np.nan, np.nan
    if signal_fs_1 != signal_fs_2 or signal_1.size != signal_2.size:
      if signal_fs_1 > signal_fs_2:
        new_x = np.arange(0, signal_2.size/signal_fs_2, step = 1.0/signal_fs_1)
        signal_2 = np.interp(new_x, np.arange(0, signal_2.size/signal_fs_2, step = 1.0/signal_fs_2), signal_2)
        signal_fs_2 = signal_fs_1
      else:
        new_x = np.arange(0, signal_1.size/signal_fs_1, step = 1.0/signal_fs_2)
        signal_1 = np.interp(new_x, np.arange(0, signal_1.size/signal_fs_1, step = 1.0/signal_fs_1), signal_1)
        signal_fs_1 = signal_fs_2

    #   logger.warning("The two signals do not have the same fs in compute correlation")
    #   return np.nan, np.nan
    # else:
    mlen = min(signal_1.size, signal_2.size)
    sig1 = signal_1[0:mlen]
    sig2 = signal_2[0:mlen]
    correlation = np.zeros(2*int(max*signal_fs_1))
    if int(max*signal_fs_1) <3:
      logger.warning("strange")
    for i in tqdm(range(correlation.shape[0])):
      if is_auto and i == max*signal_fs_1:
        correlation[i] = np.nan
      else:
        correlation[i] = np.dot(sig1, np.roll(sig2, int(i-max*signal_fs_1)))
    return signal_fs_1, correlation
   
  tqdm.pandas(desc="Declaring ressources for correlation_df") 
  correlation_df = mk_block(correlation_df, ["signal_1", "signal_fs_1", "signal_2", "signal_fs_2", "max", "is_auto"], compute_correlation, 
                      {0: (np_loader, "correlation_fs", True), 1: (np_loader, "correlation_pow", True)}, computation_m)

  return correlation_df


def resample_if_needed(df):
    def nan_with_warning(str):
      logger.warning(str)
      return np.nan
    
    x_coords = df["correlation_t"].to_list()
    if all(np.array_equal(x, x_coords[0]) for x in x_coords):
      x_coords = np.array(x_coords[0])
      y_pow = df["correlation_pow"]
    else:
      try:
        minx, maxx = min(np.amin(x_coord) for x_coord in x_coords), max(np.amax(x_coord) for x_coord in x_coords)
      except:
        logger.warning("Problem Strange xcoords: {}".format(x_coords))
        input()
      step = (maxx-minx)/max(v.size for v in x_coords)
      x_coords = np.arange(minx, maxx, step)
      y_pow = df.apply(lambda row: 
                       np.interp(x_coords, row["correlation_t"], row["correlation_pow"]) if np.all(np.diff(row["correlation_t"]) > 0) 
                       else nan_with_warning("Not increasing x values for resampling. Ignoring."), 
      axis=1)
    return x_coords, np.array(y_pow.to_list())


def compute_group_info(df):
  x_coords, y_pow = resample_if_needed(df)
  pow_avg = np.mean(y_pow, axis=0)
  pow_median = np.median(y_pow, axis=0)

  res = pd.DataFrame(
    [
      [x_coords,  pow_avg, "avg", len(df.index)], 
      [x_coords,  pow_median, "median", len(df.index)], 
    ], 
    columns=["correlation_t", "correlation_pow", "aggregation_type", "n"]
  )
  return res
