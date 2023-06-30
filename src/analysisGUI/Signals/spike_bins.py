
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging

logger = logging.getLogger(__name__)

class SpikeBins(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("signals.spike_bins", 
            {
                "signals.fs":"1000",
                "spike_bins.min_nb": "100",
                "spike_bins.diff_duration_error": "1",
            }
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        self.tqdm.pandas(desc="Computing spike bins")
        df = db[db["input_signal_type"]=="spike_times"].copy()
        
        for key,val in self.metadata.items():
            if "spike_bins." in key:
                df[str(key[len("spike_bins."):])] = val
        df["spike_bins_fs"] = self.metadata["signals.fs"]

        df.insert(0, "spike_bins_signal", df.progress_apply(lambda row: 
            self.computation_m.declare_computable_ressource(make_continuous,
                dict(signal=row["input_signal"], signal_fs = row["input_signal_fs"], 
                     out_fs=float(row["spike_bins_fs"]),
                     duration=row["Duration"],
                     min_nb = int(row["min_nb"]),
                     diff_duration_error = float(row["diff_duration_error"]),
                ), toolbox.np_loader, "spike_bins_signal", True
            ),
            axis=1)
        )
        
        df.insert(0, "signal_resampled_fs", float(self.metadata["signals.fs"]))
        df.insert(0, "signal_resampled", df["spike_bins_signal"])
        
        return df

def make_continuous(signal: np.array, signal_fs, out_fs, duration, min_nb, diff_duration_error):
    signal_fs = float(signal_fs)
    if signal.size <min_nb:
        return toolbox.Error("Removed: insufficient number of spikes ({})".format(signal.size))
    
    indices = (signal *out_fs/signal_fs).astype(int)
    new_expected_size = int(duration*out_fs)+1
    minimum_fit_size = int(np.amax(indices))+1
    if minimum_fit_size > new_expected_size + diff_duration_error*signal_fs:
        raise BaseException("Duration and sizes do not match")
    zeros = np.zeros(max(new_expected_size, minimum_fit_size))
    np.add.at(zeros, (signal *out_fs/signal_fs).astype(int), 1)
    return zeros