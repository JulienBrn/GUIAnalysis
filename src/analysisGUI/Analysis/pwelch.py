
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging

logger = logging.getLogger(__name__)

class PWelch(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("analysis.pwelch", 
            {
                "pwelch.window_duration":"1",
                "pwelch.preprocess":"z-score",
                "pwelch.nb_min_sig_time":"10",
                "pwelch.best_f.min":"8",
                "pwelch.best_f.max":"35",

            }
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        self.tqdm.pandas(desc="Computing pwelch")
        df = db.copy()

        for key,val in self.metadata.items():
            if "pwelch." in key:
                df[str(key[len("pwelch."):])] = val

        df.insert(0, "pwelch_nb_f", df.progress_apply(lambda row:  1+int(float(row["window_duration"]) * float(row["signal_resampled_fs"])/2), axis=1))
        df.insert(0, "pwelch_max_f", df.progress_apply(lambda row:  float(row["signal_resampled_fs"])/2, axis=1))
        df.insert(0, "pwelch_fs", df.progress_apply(lambda row:  (row["pwelch_nb_f"]-1) / row["pwelch_max_f"], axis=1))

        df.insert(0, "pwelch", df.progress_apply(lambda row: 
            self.computation_m.declare_computable_ressource(pwelch,
                dict(signal=row["signal_resampled"], signal_fs = row["signal_resampled_fs"], 
                     window_duration=float(row["window_duration"]),
                     preprocess=row["preprocess"],
                     out_fs=row["pwelch_fs"],
                     nb_min_sig_time=float(row["nb_min_sig_time"]),
                ), toolbox.np_loader, "pwelch", True
            ),
            axis=1)
        )

        df.insert(0, "best_f_ind", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
            lambda a, min, max, fs: np.argmax(a[int(min*fs):int(max*fs)]) + int(min*fs), {"a":row["pwelch"], "min":float(row["best_f.min"]), "max":float(row["best_f.max"]), "fs": row["pwelch_fs"]}, 
            toolbox.float_loader, "pwelch_best_f_ind", True), axis=1))
        df.insert(0, "best_f", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
            lambda a, min, max, fs: (np.argmax(a[int(min*fs):int(max*fs)]) + int(min*fs))/float(fs), {"a":row["pwelch"], "min":float(row["best_f.min"]), "max":float(row["best_f.max"]), "fs": row["pwelch_fs"]}, 
            toolbox.float_loader, "pwelch_best_f", True), axis=1))
        df.insert(0, "best_amp", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
            lambda a, min, max, fs: float(np.amax(a[int(min*fs):int(max*fs)])), {"a":row["pwelch"], "min":float(row["best_f.min"]), "max":float(row["best_f.max"]), "fs": row["pwelch_fs"]}, 
            toolbox.float_loader, "pwelch_best_amp", True), axis=1))
        # df.insert(0, "best_amp_check", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
        #     lambda a, ind: float(a[ind]), {"a":row["pwelch"], "ind": row["best_f_ind"]}, 
        #     toolbox.float_loader, "pwelch_best_amp_check", True), axis=1))
        


        return df




def pwelch(signal, signal_fs, window_duration, preprocess, out_fs, nb_min_sig_time):
    if preprocess=="z-score":
      normalized = scipy.stats.zscore(signal)
    elif preprocess=="none":
      normalized = signal
    else:
      raise BaseException("Unknown value {} for preprocess_normalization".format(preprocess))
    if normalized.size < nb_min_sig_time*signal_fs:
       return toolbox.Error("Discarded: signal too short")
    freqs, vals = scipy.signal.welch(normalized, signal_fs, nperseg=window_duration*signal_fs)
    real_out_fs = float(1.0/np.mean(freqs[1:] - freqs[0:-1]))
    if real_out_fs - out_fs>0.00001:
       raise BaseException("Wrong output fs. Expected {}. Got {}".format(real_out_fs, out_fs))
    if freqs[0]!= 0:
        raise BaseException("Wrong freqs[0]")
    return vals