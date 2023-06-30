
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class MergeMonkeySignals(GUIDataFrame):
    def __init__(self, monkey_raw, monkey_spikes, computation_m: toolbox.Manager):
        super().__init__("inputs.monkey.signals.merged", {}, computation_m, {"raw":monkey_raw, "spikes":monkey_spikes}, alternative_names=["inputs.monkey.signals", "inputs.monkey"])
        self.computation_m = computation_m
    
    def compute_df(self, raw: pd.DataFrame, spikes: pd.DataFrame):
        df = pd.concat([raw, spikes], join="inner", ignore_index=True)
        return df


