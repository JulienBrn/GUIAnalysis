
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, h5py
import toolbox

class DeclareRatSignals(GUIDataFrame):
    def __init__(self, db, computation_m: toolbox.Manager):
        super().__init__("inputs.rat.signals.declare", {}, computation_m, {"db":db}, alternative_names=["inputs.rat.signals", "inputs.rat"])
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        self.tqdm.pandas(desc="Declaring rat signals")
        df = db.copy()
        def declare_sig(d: toolbox.DataPath):
            if not isinstance(d, toolbox.DataPath):
                d = toolbox.DataPath.from_str(d)
            m = toolbox.matlab_loader.load(self.metadata["inputs.rat.files.base_folder"] + "/" +d.file_path)
            for k in d.keys:
                m = m[k]
            return np.squeeze(m)

        df["signal"] = df["signal_path"].progress_apply(lambda d: self.computation_m.declare_computable_ressource(declare_sig, {"d":d}, toolbox.np_loader, "rat input signals"))
        return df
    

        
    
