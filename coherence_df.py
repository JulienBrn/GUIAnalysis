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

class coherenceDataDF:
   
  def __init__(self, computation_m, step_signal, lfpDF, buaDF, spikeDF):
     self.computation_m = computation_m
     self.lfpDF = lfpDF
     self.buaDF = buaDF
     self.spikeDF = spikeDF
     self.metadata = {
      "coherence.window_duration":"3",
      **self.lfpDF.metadata,
      **self.buaDF.metadata,
      **self.spikeDF.metadata,
      } 
     self.invalidated = True
     self.step_signal = step_signal

  
  name = "coherence"
  result_columns = ["coherence_f", "coherence_pow", "coherence_phase"]

  def get_df(self):
    if self.invalidated:
      self.lfpDF.get_df()
      self.buaDF.get_df()
      self.spikeDF.get_df()
      signal_df = pd.concat([self.step_signal["bua"], self.step_signal["lfp"], self.step_signal["spike_continuous"]], ignore_index=True)
      coherence_params = {key[10:].replace(".", "_"):parse_param(val) for key,val in self.metadata.items() if "coherence." in key}
      self._dataframe = _get_df(self.computation_m, signal_df, coherence_params)
      self.invalidated = False
    return self._dataframe



  def view_item(self, canvas, row):
    canvas.ax = canvas.fig.subplots(2, sharex="all")
    canvas.ax[0].set_xlabel("Frequency (Hz)")
    canvas.ax[0].set_ylabel("Amplitude (?)")
    canvas.ax[1].set_xlabel("Frequency (Hz)")
    canvas.ax[1].set_ylabel("Phase (?)")
    canvas.ax[0].set_xlim(3, 60)
    y = row["coherence_pow"].get_result()
    x = row["coherence_f"].get_result()
    phase = row["coherence_phase"].get_result()
    if not x is np.nan:
      canvas.ax[0].plot(x, y)
      canvas.ax[1].plot(x, phase)

      
  
  def view_items(self, canvas, rows: pd.DataFrame):
    if len(rows.index) < 6:
      mode=0
    elif len(rows["Species"].unique())==1:
      mode =1
    else:
      mode=2
    if mode == 0:
      canvas.ax = canvas.fig.subplots(2, sharex="all")
      canvas.ax[0].set_xlabel("Frequency (Hz)")
      canvas.ax[0].set_ylabel("Amplitude (?)")
      canvas.ax[1].set_xlabel("Frequency (Hz)")
      canvas.ax[1].set_ylabel("Phase (?)")
      canvas.ax[0].set_xlim(3, 60)
      for i in range(len(rows.index)):
        y = rows["coherence_pow"].iat[i].get_result()
        phase = rows["coherence_phase"].iat[i].get_result()
        x = rows["coherence_f"].iat[i].get_result()
        if not x is np.nan:
          label_dict = {k:v for k,v in rows.iloc[i, :].to_dict().items() if not "__" in k and not "SubSession" in k and not isinstance(v, RessourceHandle)}
          label = ",".join(["{}={}".format(k, v) for k,v in label_dict.items()])
          canvas.ax[0].plot(x, y, label="{}".format(label))
          canvas.ax[1].plot(x, phase, label="{}".format(label))
      canvas.ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
      canvas.fig.tight_layout()
    elif mode ==1 or mode==2:
      df = rows[(rows["signal_type_1"]=="bua_cleaned") & (rows["signal_type_2"]==rows["signal_type_1"])].copy().reset_index()
      # df["coherence_pow"] = df.apply(lambda row: scipy.stats.zscore(row["coherence_pow"].get_result()), axis=1, result_type="reduce")
      tqdm.pandas(desc="Computing ressources for coherence_df") 
      df["coherence_pow"] = df.progress_apply(lambda row: row["coherence_pow"].get_result(), axis=1, result_type="reduce")
      df["aggregation_type"] = "plot"
      def compute_info(df):
        x_coords = [v.get_result() for v in df["coherence_f"]]
        if all(np.array_equal(x, x_coords[0]) for x in x_coords):
          y_vals = [v for v in df["coherence_pow"]]
          y_avg = np.mean(y_vals, axis=0)
          y_med = np.median(y_vals, axis=0)
          res = pd.DataFrame([[x_coords[0],  y_avg, "avg"], [x_coords[0],  y_med, "median"]], columns=["coherence_f", "coherence_pow", "aggregation_type"])
          # res["welch_f"] = x_coords[0]
          # res["welch_pow"] = y_avg
          # res["aggregation_type"] = "avg"
          return res
          # print(x_coords[0].shape, y_avg.shape)
        else:
          mdf = pd.DataFrame()
          mdf["coherence_f"] = df["coherence_f"]
          mdf["coherence_pow"] = df["coherence_pow"]
          mdf["min"] = mdf.apply(lambda row: np.amin(row["coherence_f"].get_result()), axis=1)
          mdf["max"] = mdf.apply(lambda row: np.amax(row["coherence_f"].get_result()), axis=1)

          max_x = mdf["max"].min()
          min_x = mdf["min"].max()
          logger.info("minx = {}, max_x={}".format(min_x, max_x))
          new_x = np.arange(min_x, max_x, (max_x-min_x)/10000)
          mdf["resampled"] = mdf.apply(lambda row: np.interp(new_x, row["coherence_f"].get_result(), row["coherence_pow"]) if np.all(np.diff(row["coherence_f"].get_result()) > 0) else np.nan, axis=1)
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
          res = pd.DataFrame([[new_x,  y_avg, "avg"], [new_x,  y_med, "median"]], columns=["coherence_f", "coherence_pow", "aggregation_type"])
          return res
      # print(df)
      r = df.groupby(by=["Species", "Condition", "Structure_1", "signal_type_1", "Structure_2", "signal_type_2"]).apply(compute_info)
      # print(r)
      r=r.reset_index()
      # if mode ==1:
      if True:
        df = pd.concat([df, r], ignore_index=True)
        # logger.info(df.to_string())
        toolbox.add_draw_metadata(df, col_group=["Species", "Structure_1", "Condition"], row_group=["signal_type_1" , "Structure_2"], color_group=["aggregation_type"])
        # logger.info(df.to_string())
        p = toolbox.prepare_figures2(df, [canvas.fig], xlim=[3, 60])
        # p = toolbox.prepare_figures2(df, [canvas.fig])
        # df=df[(df["Structure"].isin(["STN", "STR", "ECoG"])) & (df["Condition"]=="CTL") ]
        # bugdf = df[["Structure", "signal_type", "Condition", "Row_label", "Column_label", "Column"]]
        # toolbox.df_loader.save("bugdf.tsv", bugdf)
        p.plot2(df, x="coherence_f", y="coherence_pow", use_zscore=False)
      elif mode ==2:
        df = r[r["aggregation_type"]=="median"].reset_index(drop=True)
        # logger.info(df.to_string())
        toolbox.add_draw_metadata(df, col_group=["Species","aggregation_type"], row_group=["signal_type"], color_group=[ "Structure", "Condition"])
        # logger.info(df.to_string())
        p = toolbox.prepare_figures2(df, [canvas.fig], xlim=[3, 60], ylim=[-2.1, 8])
        # p = toolbox.prepare_figures2(df, [canvas.fig])
        # df=df[(df["Structure"].isin(["STN", "STR", "ECoG"])) & (df["Condition"]=="CTL") ]
        # bugdf = df[["Structure", "signal_type", "Condition", "Row_label", "Column_label", "Column"]]
        # toolbox.df_loader.save("bugdf.tsv", bugdf)
        p.plot2(df, x="welch_f", y="welch_pow", use_zscore=False)
        
  


def _get_df(computation_m, signal_df, coherence_params):
  tqdm.pandas(desc="Computing shape of coherence_df") 
  # coherence_df = toolbox.group_and_combine(signal_df[signal_df["signal_type"].isin(["lfp_cleaned", "bua_cleaned", "spike_continuous"])], ["Condition", "Subject", "Species", "Session", "Date", "SubSessionInfo", "SubSessionInfoType"])
  coherence_df = toolbox.group_and_combine(signal_df[signal_df["signal_type"].isin(["bua_cleaned"])], ["Condition", "Subject", "Species", "Session", "Date", "SubSessionInfo", "SubSessionInfoType"])

  for key,val in coherence_params.items():
    coherence_df[key] = val

  def compute_coherence(signal_1, signal_fs_1, signal_2, signal_fs_2, window_duration):
    if signal_fs_1 != signal_fs_2 or signal_1.shape != signal_2.shape:
        mduration = min(signal_1.size/signal_fs_1, signal_2.size/signal_fs_2)
        if signal_fs_1 < signal_fs_2:
          sig1 = signal_1[0:int(mduration*signal_fs_1)]
          sig2 = scipy.signal.resample(signal_2[0:int(mduration*signal_fs_2)], int(mduration*signal_fs_1))
          fs = signal_fs_1
        else:
          sig1 = scipy.signal.resample(signal_1[0:int(mduration*signal_fs_1)], int(mduration*signal_fs_2))
          sig2 = signal_2[0:int(mduration*signal_fs_2)]
          fs = signal_fs_2
        # logger.warning("The two signals do not have the same fs")
        # return {"frequencies":np.nan, "pow": np.nan, "phase": np.nan}
    else:
        sig1 = signal_1
        sig2 = signal_2
        fs = signal_fs_1

    f11, csd11=scipy.signal.csd(sig1, sig1, fs, nperseg=window_duration*fs)
    f22, csd22=scipy.signal.csd(sig2, sig2, fs, nperseg=window_duration*fs)
    f12, csd12=scipy.signal.csd(sig1, sig2, fs, nperseg=window_duration*fs)
    coherence_pow = abs(csd12)**2/(csd22*csd11)
    coherence_phase = np.angle(csd12, deg=True)
    return {"frequencies": f11, "pow": coherence_pow, "phase": coherence_phase}

  tqdm.pandas(desc="Declaring ressources for coherence_df") 
  coherence_df = mk_block(coherence_df, ["version", "signal_1", "signal_fs_1", "signal_2", "signal_fs_2", "window_duration"], compute_coherence, 
                      {"frequencies": (np_loader, "coherence_f", True), "pow": (np_loader, "coherence_pow", True), "phase": (np_loader, "coherence_phase", True)}, computation_m)

  return coherence_df
