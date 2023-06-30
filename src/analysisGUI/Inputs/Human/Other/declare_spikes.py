
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class DeclareHumanOtherSpikes(GUIDataFrame):
    def __init__(self, md, computation_m: toolbox.Manager):
        super().__init__("inputs.human.other.signals.spikes", {}, computation_m, {"db":md})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        neuron_df = db.copy()
        neuron_df["signal_type"] = "spike_times"
        neuron_df["signal_fs"] = 1
        neuron_df["signal_path"] = neuron_df["file_path"].apply(lambda fp: toolbox.DataPath(fp, ["SUA"]))
        def get_neuron_sig(fp):
            mat = scipy.io.loadmat(self.metadata["inputs.human.other.files.base_folder"] + "/" + fp)
            sig = np.reshape(mat["SUA"], -1)
            return sig
        neuron_df["signal"] = neuron_df["file_path"].apply(lambda fp: self.computation_m.declare_computable_ressource(get_neuron_sig, {"fp": fp}, toolbox.np_loader, "human_other_spike_input", True))
        
        neuron_df["max_spike_time"] = neuron_df.apply(lambda row: self.computation_m.declare_computable_ressource(lambda sig, fs: float(sig[-1])/fs, {"sig": row["signal"], "fs": row["signal_fs"]}, toolbox.float_loader, "human_other_spike_input_max", False), axis=1)
        
        return neuron_df.drop(columns=["Source"])

