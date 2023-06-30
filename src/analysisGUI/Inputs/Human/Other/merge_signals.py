
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class MergeHumanOtherSignals(GUIDataFrame):
    def __init__(self, raw, spikes, computation_m: toolbox.Manager):
        super().__init__("inputs.human.other.signals.merged", {}, computation_m, {"raw":raw, "spikes":spikes}, alternative_names=["inputs.human.other.signals", "inputs.human.other"])
        self.computation_m = computation_m
    
    def compute_df(self, raw: pd.DataFrame, spikes: pd.DataFrame):
        df = pd.concat([raw, spikes], join="outer", ignore_index=True)
        return df.drop(columns=["DateH", "ElectrodeDepth", "file_path"])


