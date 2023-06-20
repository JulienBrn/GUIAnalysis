
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging

logger = logging.getLogger(__name__)

class LFP(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("signals.lfp", 
            {
                "signals.fs":"1000",
                "lfp.lowpass_filter_freq":"200",
                "lfp.lowpass_order": "3",
            }
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        df = db[db["input_signal_type"]!="mua"].copy()

        for key,val in self.metadata.items():
            if "lfp." in key:
                df[str(key[len("lfp."):])] = val


        df.insert(0, "lfp_signal", df.apply(lambda row: 
            self.computation_m.declare_computable_ressource(extract_lfp,
                dict(signal=row["cleaned_signal"], signal_fs = row["input_signal_fs"], 
                     lowpass_filter_freq=float(row["lowpass_filter_freq"]),
                     lowpass_order=float(row["lowpass_order"]),
                ), toolbox.np_loader, "lfp_signal", False
            ),
            axis=1)
        )
        
        df.insert(0, "signal_resampled_fs", float(self.metadata["signals.fs"]))
        df.insert(0, "signal_resampled", df.apply(lambda row:
            self.computation_m.declare_computable_ressource(
                lambda sig, fs, out_fs: scipy.signal.resample(sig, math.ceil(sig.size* out_fs/float(fs))),
                dict(sig=row["lfp_signal"], fs = row["input_signal_fs"], 
                     out_fs=row["signal_resampled_fs"],
                ), toolbox.np_loader, "lfp_signal_resampled", True
            ),
            axis=1)
        )


        return df

def extract_lfp(signal, signal_fs, lowpass_filter_freq, lowpass_order):
    filtera_lfp, filterb_lfp = scipy.signal.butter(lowpass_order, lowpass_filter_freq, fs=signal_fs, btype='low', analog=False)
    lfp = scipy.signal.lfilter(filtera_lfp, filterb_lfp, signal)
    return lfp