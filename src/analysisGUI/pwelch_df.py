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

class pwelchDataDF:
   
  def __init__(self, computation_m, step_signal, lfpDF, buaDF):
     self.computation_m = computation_m
     self.lfpDF = lfpDF
     self.buaDF = buaDF
     self.metadata = {
      "pwelch.window_duration":"3",
      "pwelch.preprocess.normalization":"z-score",
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
      signal_df = pd.concat([self.step_signal["bua"], self.step_signal["lfp"]], ignore_index=True)
      pwelch_params = {key[7:].replace(".", "_"):parse_param(val) for key,val in self.metadata.items() if "pwelch." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, pwelch_params)
      self.invalidated = False
    return self._dataframe

  def compute(self):
    self.get_df()
    tqdm.pandas(desc="Compute coherence_df results") 
    def compute_elem(row):
      for col in self.result_columns:
        if isinstance(row[col], RessourceHandle):
          row[col].get_result()
    self._dataframe.progress_apply(compute_elem, axis=1)

  def view_item(self, canvas, row):
    
    canvas.ax = canvas.fig.subplots(1)
    y = row["welch_pow"].get_result()
    x = row["welch_f"].get_result()
    canvas.ax.plot(x, y)
    canvas.ax.set_xlabel("Frequency (Hz)")
    canvas.ax.set_ylabel("Amplitude (?)")
    canvas.ax.set_xlim(3, 60)

  def get_nb_figs(self, rows):
    if len(rows.index) < 6:
      return 1
    else:
      return rows["Species"].nunique() + 4

  def show_figs(self, rows, canvas_list):
    if len(rows.index) < 6:
      canvas = canvas_list[0]
      canvas.ax = canvas.fig.subplots(1)
      for i in range(len(rows.index)):
        label_dict = {k:v for k,v in rows.iloc[i, :].to_dict().items() if not "__" in k and not "SubSession" in k and not isinstance(v, RessourceHandle)}
        label = ",".join(["{}={}".format(k, v) for k,v in label_dict.items()])
        y = rows["welch_pow"].iat[i].get_result()
        x = rows["welch_f"].iat[i].get_result()
        canvas.ax.plot(x, y)
      canvas.ax.set_xlabel("Frequency (Hz)")
      canvas.ax.set_ylabel("Amplitude (?)")
      canvas.ax.set_xlim(3, 60)
      canvas.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
      canvas.fig.tight_layout()
    else:
      curr_canvas = 0
      df = rows.copy().reset_index()
      tqdm.pandas(desc="Computing ressources for pwelch_df") 
      df["welch_pow"] = df.progress_apply(lambda row: row["welch_pow"].get_result(), axis=1, result_type="reduce")
      df["aggregation_type"] = "plot"
      def compute_info(df):
        x_coords = [v.get_result() for v in df["welch_f"]]
        if all(np.array_equal(x, x_coords[0]) for x in x_coords):
          y_vals = [v for v in df["welch_pow"]]
          y_avg = np.mean(y_vals, axis=0)
          y_med = np.median(y_vals, axis=0)
          res = pd.DataFrame([[x_coords[0],  y_avg, "avg"], [x_coords[0],  y_med, "median"]], columns=["welch_f", "welch_pow", "aggregation_type"])
          return res
        else:
          mdf = pd.DataFrame()
          mdf["welch_f"] = df["welch_f"]
          mdf["welch_pow"] = df["welch_pow"]
          mdf["min"] = mdf.apply(lambda row: np.amin(row["welch_f"].get_result()), axis=1)
          mdf["max"] = mdf.apply(lambda row: np.amax(row["welch_f"].get_result()), axis=1)

          max_x = mdf["max"].min()
          min_x = mdf["min"].max()
          logger.info("minx = {}, max_x={}".format(min_x, max_x))
          new_x = np.arange(min_x, max_x, (max_x-min_x)/10000)
          mdf["resampled"] = mdf.apply(lambda row: np.interp(new_x, row["welch_f"].get_result(), row["welch_pow"]) if np.all(np.diff(row["welch_f"].get_result()) > 0) else np.nan, axis=1)
          y_vals = [v for v in mdf["resampled"]]
          y_avg = np.mean(y_vals, axis=0)
          y_med = np.median(y_vals, axis=0)
          res = pd.DataFrame([[new_x,  y_avg, "avg"], [new_x,  y_med, "median"]], columns=["welch_f", "welch_pow", "aggregation_type"])
          return res
      r = df.groupby(["Species", "Structure", "signal_type", "Condition"]).apply(compute_info).reset_index()
      def by_species():
        nonlocal curr_canvas
        df = r.reset_index(drop=True)
        toolbox.add_draw_metadata(df, fig_group = ["aggregation_type"], col_group=["Species"], row_group=["signal_type"], color_group=[ "Structure", "Condition"])
        nb_figs = df.groupby(by = ["aggregation_type"]).ngroups
        logger.info("nb_figs = {}".format(nb_figs))
        # logger.info("df is\n{}".format(df))
        p = toolbox.prepare_figures2(df, [canvas.fig for canvas in canvas_list[curr_canvas:curr_canvas+nb_figs]], xlim=[3, 60])
        curr_canvas+=nb_figs
        p.plot2(df, x="welch_f", y="welch_pow", use_zscore=False)
        logger.info("by species done")
        return list(range(curr_canvas-nb_figs, curr_canvas))
      def by_structure():
        nonlocal curr_canvas
        df = r.reset_index(drop=True)
        df["Condition_b"] = df.apply(lambda row: "Park" if row["Condition"] in ["pd", "Park", "mptp"] else "healthy", axis=1)
        toolbox.add_draw_metadata(df, fig_group = ["aggregation_type"], col_group=["Structure"], row_group=["Condition_b", "signal_type"], color_group=["Species"])
        nb_figs = df.groupby(["aggregation_type"]).ngroups
        logger.info("nb_figs = {}".format(nb_figs))
        p = toolbox.prepare_figures2(df, [canvas.fig for canvas in canvas_list[curr_canvas:curr_canvas+nb_figs]], xlim=[3, 60])
        curr_canvas+=nb_figs
        p.plot2(df, x="welch_f", y="welch_pow", use_zscore=False)
        return list(range(curr_canvas-nb_figs, curr_canvas))
      def details():
        nonlocal curr_canvas, df
        df = pd.concat([df, r], ignore_index=True)
        toolbox.add_draw_metadata(df, fig_group = ["Species"], col_group=["Structure"], row_group=["signal_type", "Condition"], color_group=["aggregation_type"])
        nb_figs = df.groupby(["Species"]).ngroups
        p = toolbox.prepare_figures2(df, [canvas.fig for canvas in canvas_list[curr_canvas:curr_canvas+nb_figs]], xlim=[3, 60])
        curr_canvas+=nb_figs
        p.plot2(df, x="welch_f", y="welch_pow", use_zscore=False)
        return list(range(curr_canvas-nb_figs, curr_canvas))
      yield(by_species())
      yield(by_structure())
      yield(details())

  def view_items(self, canvas, rows):
    if len(rows.index) < 6:
      mode=0
    elif len(rows["Species"].unique())==1:
      mode =1
    else:
      mode=2
    if mode == 0:
      canvas.ax = canvas.fig.subplots(1)
      for i in range(len(rows.index)):
        label_dict = {k:v for k,v in rows.iloc[i, :].to_dict().items() if not "__" in k and not "SubSession" in k and not isinstance(v, RessourceHandle)}
        label = ",".join(["{}={}".format(k, v) for k,v in label_dict.items()])
        y = rows["welch_pow"].iat[i].get_result()
        x = rows["welch_f"].iat[i].get_result()
        canvas.ax.plot(x, y)
      canvas.ax.set_xlabel("Frequency (Hz)")
      canvas.ax.set_ylabel("Amplitude (?)")
      canvas.ax.set_xlim(3, 60)
      canvas.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
      canvas.fig.tight_layout()
    elif mode ==1 or mode==2:
      df = rows.copy().reset_index()
      tqdm.pandas(desc="Computing ressources for pwelch_df") 
      df["welch_pow"] = df.progress_apply(lambda row: row["welch_pow"].get_result(), axis=1, result_type="reduce")
      df["aggregation_type"] = "plot"
      def compute_info(df):
        x_coords = [v.get_result() for v in df["welch_f"]]
        if all(np.array_equal(x, x_coords[0]) for x in x_coords):
          y_vals = [v for v in df["welch_pow"]]
          y_avg = np.mean(y_vals, axis=0)
          y_med = np.median(y_vals, axis=0)
          res = pd.DataFrame([[x_coords[0],  y_avg, "avg"], [x_coords[0],  y_med, "median"]], columns=["welch_f", "welch_pow", "aggregation_type"])
          # res["welch_f"] = x_coords[0]
          # res["welch_pow"] = y_avg
          # res["aggregation_type"] = "avg"
          return res
          # print(x_coords[0].shape, y_avg.shape)
        else:
          mdf = pd.DataFrame()
          mdf["welch_f"] = df["welch_f"]
          mdf["welch_pow"] = df["welch_pow"]
          mdf["min"] = mdf.apply(lambda row: np.amin(row["welch_f"].get_result()), axis=1)
          mdf["max"] = mdf.apply(lambda row: np.amax(row["welch_f"].get_result()), axis=1)

          max_x = mdf["max"].min()
          min_x = mdf["min"].max()
          logger.info("minx = {}, max_x={}".format(min_x, max_x))
          new_x = np.arange(min_x, max_x, (max_x-min_x)/10000)
          mdf["resampled"] = mdf.apply(lambda row: np.interp(new_x, row["welch_f"].get_result(), row["welch_pow"]) if np.all(np.diff(row["welch_f"].get_result()) > 0) else np.nan, axis=1)
          y_vals = [v for v in mdf["resampled"]]
          y_avg = np.mean(y_vals, axis=0)
          y_med = np.median(y_vals, axis=0)
          # debugdf = pd.DataFrame()
          # debugdf["f"] = new_x
          # for i in range(len(mdf.index)):
          #   debugdf["pow" + str(i)] = mdf["resampled"].iat[i]
          # debugdf["yavg"] = y_avg
          # debugdf["ymed"] = y_med
          # toolbox.df_loader.save("debug.tsv", debugdf)
          res = pd.DataFrame([[new_x,  y_avg, "avg"], [new_x,  y_med, "median"]], columns=["welch_f", "welch_pow", "aggregation_type"])
          return res
      r = df.groupby(["Species", "Structure", "signal_type", "Condition"]).apply(compute_info).reset_index()
      if mode ==1:
        df = pd.concat([df, r], ignore_index=True)
        # logger.info(df.to_string())
        toolbox.add_draw_metadata(df, col_group=["Species", "Structure"], row_group=["signal_type", "Condition"], color_group=["aggregation_type"])
        # logger.info(df.to_string())
        p = toolbox.prepare_figures2(df, [canvas.fig], xlim=[3, 60])
        # p = toolbox.prepare_figures2(df, [canvas.fig])
        # df=df[(df["Structure"].isin(["STN", "STR", "ECoG"])) & (df["Condition"]=="CTL") ]
        # bugdf = df[["Structure", "signal_type", "Condition", "Row_label", "Column_label", "Column"]]
        # toolbox.df_loader.save("bugdf.tsv", bugdf)
        p.plot2(df, x="welch_f", y="welch_pow", use_zscore=False)
      elif mode ==2:
        df = r[r["aggregation_type"]=="median"].reset_index(drop=True)
        # logger.info(df.to_string())
        toolbox.add_draw_metadata(df, col_group=["Species","aggregation_type"], row_group=["signal_type"], color_group=[ "Structure", "Condition"])
        # logger.info(df.to_string())
        p = toolbox.prepare_figures2(df, [canvas.fig], xlim=[3, 60])
        # p = toolbox.prepare_figures2(df, [canvas.fig])
        # df=df[(df["Structure"].isin(["STN", "STR", "ECoG"])) & (df["Condition"]=="CTL") ]
        # bugdf = df[["Structure", "signal_type", "Condition", "Row_label", "Column_label", "Column"]]
        # toolbox.df_loader.save("bugdf.tsv", bugdf)
        p.plot2(df, x="welch_f", y="welch_pow", use_zscore=False)
      


def _get_df(computation_m, signal_df, pwelch_params):
  pwelch_df = signal_df[signal_df["signal_type"].isin(["lfp_cleaned", "bua_cleaned"])].copy()

  for key,val in pwelch_params.items():
    pwelch_df[key] = val

  def pwelch(signal, signal_fs, window_duration, preprocess_normalization):
    if preprocess_normalization=="z-score":
      normalized = scipy.stats.zscore(signal)
    elif preprocess_normalization=="none":
      normalized = signal
    else:
      raise BaseException("Unknown value {} for preprocess_normalization".format(preprocess_normalization))
    return scipy.signal.welch(normalized, signal_fs, nperseg=window_duration*signal_fs)

  tqdm.pandas(desc="Declaring pwelch")
  pwelch_df = mk_block(pwelch_df, ["signal", "signal_fs", "window_duration", "preprocess_normalization"], pwelch, 
                      {0: (np_loader, "welch_f", True), 1: (np_loader, "welch_pow", True)}, computation_m)
  return pwelch_df
