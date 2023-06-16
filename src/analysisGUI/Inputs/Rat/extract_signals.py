
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, h5py
import toolbox

class ExtractRatSignals(GUIDataFrame):
    def __init__(self, db, computation_m):
        super().__init__("inputs.rat.signals.extract", {}, computation_m, {"db":db}, save=True)
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        self.tqdm.pandas(desc="Extracting (channels, unit) mapping")
        df = db.copy()
        group_cols = [col for col in df.columns if col not in ["raw_file_path", "units_file_path"]]
        df = df.groupby(by=group_cols).progress_apply(
                lambda row: self.extract_channels_and_units(row["raw_file_path"].iat[0], row["units_file_path"].iat[0])
            ).reset_index(level=group_cols).reset_index(drop=True)
        return df
    
    def extract_channels_and_units(self, raw_file_path, spike_file_path):
        if raw_file_path and not type(raw_file_path) is float:
            with h5py.File(self.metadata["inputs.rat.files.base_folder"] + "/"+ raw_file_path, 'r') as file:
                raw_keys = [k for k in file.keys()]
                fs = [1.0/float(file[key]["interval"][0,0]) for key in raw_keys]
                duration_raw = [np.size(file[key]["values"])*float(file[key]["interval"][0,0]) for key in raw_keys]
        else:
            raw_keys=[]
            fs=[]
            duration_raw=[]
        if spike_file_path and not type(spike_file_path) is float:
            with h5py.File(self.metadata["inputs.rat.files.base_folder"] + "/"+ spike_file_path, 'r') as file:
                spike_keys = [k for k in file.keys()]
                duration_spike=[float(np.squeeze(file[key]["length"])) for key in spike_keys]
        else:
            spike_keys=[]
            duration_spike=[]
        return pd.DataFrame(
            [[k, None, "raw", fs, toolbox.DataPath(raw_file_path, [k, "values"]), dur] for k,fs, dur in zip(raw_keys, fs, duration_raw)] + 
            [[None, k, "spike_times", 1, toolbox.DataPath(spike_file_path, [k, "times"]), dur] for k,dur in zip(spike_keys, duration_spike)], 
            columns=["Electrode", "Unit","signal_type", "signal_fs", "signal_path", "Duration"])


        
    
