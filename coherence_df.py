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
    mode = 0
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
    elif mode == 1:
       df = rows.copy()
       toolbox.add_draw_metadata(row_group=["Structure_1"], col_group=["Structure_2"])
        
  


def _get_df(computation_m, signal_df, coherence_params):
  tqdm.pandas(desc="Computing shape of coherence_df") 
  coherence_df = toolbox.group_and_combine(signal_df[signal_df["signal_type"].isin(["lfp_cleaned", "bua_cleaned", "spike_continuous"])], ["Condition", "Subject", "Species", "Session", "Date", "SubSessionInfo", "SubSessionInfoType"])

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
