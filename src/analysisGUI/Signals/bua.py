
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging

logger = logging.getLogger(__name__)

class BUA(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("signals.bua", 
            {
                "signals.fs":"1000",
                "bua.bandpass.low_freq":"300",
                "bua.bandpass.high_freq":"6000",
                "bua.lowpass.freq": "1000",
                "bua.passes.order": "3",
            }
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        self.tqdm.pandas(desc="Computing bua signals")
        df = db.copy()

        for key,val in self.metadata.items():
            if "bua." in key:
                df[str(key[len("bua."):])] = val

        

        df.insert(0, "bua_signal", df.progress_apply(lambda row: 
            self.computation_m.declare_computable_ressource(extract_bua,
                dict(sig=row["cleaned_signal"], fs = row["input_signal_fs"], 
                     filter_low_freq=float(row["bandpass.low_freq"]),
                     filter_high_freq=float(row["bandpass.high_freq"]),
                     filter_refreq=float(row["lowpass.freq"]),
                     order=float(row["passes.order"]),
                ), toolbox.np_loader, "bua_signal", False
            ),
            axis=1)
        )

        df.insert(0, "signal_resampled_fs", float(self.metadata["signals.fs"]))
        df.insert(0, "signal_resampled", df.progress_apply(lambda row:
            self.computation_m.declare_computable_ressource(
                lambda sig, fs, out_fs: scipy.signal.resample(sig, math.ceil(sig.size* out_fs/float(fs))),
                dict(sig=row["bua_signal"], fs = row["input_signal_fs"], 
                     out_fs=row["signal_resampled_fs"],
                ), toolbox.np_loader, "bua_signal_resampled", True
            ),
            axis=1)
        )
        
        return df
    
def extract_bua(sig, fs, filter_low_freq, filter_high_freq, filter_refreq, order):
    filtera_mu_tmp, filterb_mu_tmp = scipy.signal.butter(order, [filter_low_freq, filter_high_freq], fs=fs, btype='band', analog=False)
    mu_tmp = scipy.signal.lfilter(filtera_mu_tmp, filterb_mu_tmp, sig)

    mu_abs=np.abs(mu_tmp)

    filtera_mu, filterb_mu = scipy.signal.butter(order, filter_refreq, fs=fs, btype='low', analog=False)
    bua = scipy.signal.lfilter(filtera_mu, filterb_mu, mu_abs)
    return bua