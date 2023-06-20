
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class MergeHumanSignals(GUIDataFrame):
    def __init__(self, stn, other, computation_m: toolbox.Manager):
        super().__init__("inputs.human", {}, computation_m, {"stn":stn, "other":other})
        self.computation_m = computation_m
    
    def compute_df(self, stn: pd.DataFrame, other: pd.DataFrame):
        stncp = stn.copy()
        stncp["Unit"] = stncp.pop("neuron_num")
        df = pd.concat([stncp, other[~other["Structure"].str.contains("STN")]], join="outer", ignore_index=True)
        return df.drop(columns=["file length (s)", "number of units", "Rate", "Isolation"])


