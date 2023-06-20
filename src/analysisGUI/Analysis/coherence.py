
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging

logger = logging.getLogger(__name__)

class Coherence(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("analysis.coherence", 
            {
                "coherence.window_duration":"0.5",
                "coherence.preprocess":"z-score",
                "coherence.nb_min_sig_time":"2",
            }
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):

        self.tqdm.pandas(desc="Computing shape of coherence_df") 
        df = toolbox.group_and_combine(db, ["Condition", "Subject", "Species", "Session", "Date"])
        def as_set(x):
            if isinstance(x, set):
                return x
            else:
                return {x}
        df.insert(0, "Same_Electrode", df.apply(lambda row: (row["Structure_1"] == row["Structure_2"]) and len(as_set(row["Electrode_1"]).intersection(as_set(row["Electrode_2"]))) > 0, axis=1))
        df["fs_same"] = df["signal_resampled_fs_1"] == df["signal_resampled_fs_2"]
        df.insert(0, "meta_error", ~df["fs_same"])
        df["signal_resampled_fs"] = df["signal_resampled_fs_1"]
        
        
        for key,val in self.metadata.items():
            if "coherence." in key:
                df[str(key[len("coherence."):])] = val

        df.insert(0, "coherence_nb_f", df.apply(lambda row:  1+int(float(row["window_duration"]) * float(row["signal_resampled_fs"])/2), axis=1))
        df.insert(0, "coherence_max_f", df.apply(lambda row:  float(row["signal_resampled_fs"])/2, axis=1))
        df.insert(0, "coherence_fs", df.apply(lambda row:  (row["coherence_nb_f"]-1) / row["coherence_max_f"], axis=1))

        df.insert(0, "coherence", df.apply(lambda row: 
            self.computation_m.declare_computable_ressource(coherence,
                dict(s1=row["signal_resampled_1"], s2=row["signal_resampled_2"],
                     fs = row["signal_resampled_fs"], 
                     window_duration=float(row["window_duration"]),
                     preprocess=row["preprocess"],
                     out_fs=row["coherence_fs"],
                     nb_min_sig_time=float(row["nb_min_sig_time"]),
                ), toolbox.np_loader, "coherence", True
            ) if not row["meta_error"] else "Ignored",
            axis=1)
        )


        return df


def coherence(s1, s2, fs, window_duration, preprocess, out_fs, nb_min_sig_time):
    if s1.size < nb_min_sig_time*fs:
       return toolbox.Error("Discarded: signal1 too short")
    if s2.size < nb_min_sig_time*fs:
       return toolbox.Error("Discarded: signal2 too short")
    if preprocess=="z-score":
        s1 = scipy.stats.zscore(s1)
        s2 = scipy.stats.zscore(s2)
    else:
        raise BaseException("Unknown value {} for preprocess_normalization".format(preprocess))
    
    f11, csd11=scipy.signal.csd(s1, s1, fs, nperseg=window_duration*fs)
    f22, csd22=scipy.signal.csd(s2, s2, fs, nperseg=window_duration*fs)
    f12, csd12=scipy.signal.csd(s1, s2, fs, nperseg=window_duration*fs)

    for freqs in [f11, f22, f12]:
        real_out_fs = float(1.0/np.mean(freqs[1:] - freqs[0:-1]))
        if real_out_fs - out_fs>0.00001:
            raise BaseException("Wrong output fs. Expected {}. Got {}".format(real_out_fs, out_fs))
        if freqs[0]!= 0:
            raise BaseException("Wrong freqs[0]")

    coherence = csd12 * abs(csd12)/(csd22*csd11)
    return coherence
