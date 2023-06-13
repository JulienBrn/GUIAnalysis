
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class DeclareMonkeySpikes(GUIDataFrame):
    def __init__(self, monkey_md, computation_m: toolbox.Manager):
        super().__init__("inputs.monkey.signals.spikes", {}, computation_m, {"db":monkey_md})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        neuron_df = db.copy()
        neuron_df.insert(0, "signal_fs", 40000)
        neuron_df.insert(0, "signal_type", "spike_times")
        neuron_df.insert(len(neuron_df.columns),"signal_path", neuron_df["Files"].apply(lambda fp: toolbox.DataPath(fp, ["SUA"])))
        def declare_spikes(dp):
            mat =  scipy.io.loadmat(str(self.metadata["inputs.monkey.files.base_folder"])+"/"+dp.file_path, variable_names=dp.keys[0])
            spikes = np.squeeze(mat[dp.keys[0]])
            return spikes
        
        neuron_df.insert(0, "signal", neuron_df.apply(lambda row: self.computation_m.declare_computable_ressource(declare_spikes, {"dp": row["signal_path"]}, toolbox.np_loader, "monkey_input_spikes", True), axis=1))
        return neuron_df.drop(columns=["Source", "Files"])


