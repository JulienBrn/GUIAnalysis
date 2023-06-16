
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class Inputs(GUIDataFrame):
    def __init__(self, human, rat, monkey, computation_m: toolbox.Manager):
        super().__init__("inputs", {}, computation_m, {"human":human, "rat":rat, "monkey": monkey})
        self.computation_m = computation_m
    
    def compute_df(self, human: pd.DataFrame, rat: pd.DataFrame, monkey: pd.DataFrame):
        df = pd.concat([human, rat, monkey], join="outer", ignore_index=True)
        df.insert(0, "input_signal_path", df.pop("signal_path"))
        df.insert(0, "input_signal_fs", df.pop("signal_fs"))
        df.insert(0, "input_signal", df.pop("signal"))
        df.insert(0, "input_signal_type", df.pop("signal_type"))
        
        return df.drop(columns=[])

