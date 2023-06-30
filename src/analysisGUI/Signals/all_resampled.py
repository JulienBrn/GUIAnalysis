
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class AllResampled(GUIDataFrame):
    def __init__(self, lfp, bua, spikes, computation_m: toolbox.Manager):
        super().__init__("signals.all_resampled", {}
            , computation_m, {"lfp":lfp, "bua":bua, "spikes":spikes})
        self.computation_m = computation_m
    
    def compute_df(self, lfp: pd.DataFrame, bua: pd.DataFrame, spikes: pd.DataFrame):
        df = pd.concat([lfp,bua, spikes], join="inner", keys=["lfp", "bua", "spike_bins"], names=["signal_resampled_type"]).reset_index("signal_resampled_type").reset_index(drop=True)
        return df
    
