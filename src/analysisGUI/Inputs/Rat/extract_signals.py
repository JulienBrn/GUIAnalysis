
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, h5py, re
import toolbox, logging

logger = logging.getLogger(__name__)


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
                r =re.compile("Probe(?P<probe>[0-9]+)")
                probes_raw=[int(r.search(k).groupdict()["probe"]) if not r.search(k) is None else None for k in raw_keys]
        else:
            raw_keys=[]
            fs=[]
            duration_raw=[]
            probes_raw=[]
        if spike_file_path and not type(spike_file_path) is float:
            with h5py.File(self.metadata["inputs.rat.files.base_folder"] + "/"+ spike_file_path, 'r') as file:
                spike_keys = [k for k in file.keys()]
                duration_spike=[float(np.squeeze(file[key]["length"])) for key in spike_keys]
                rs = [re.compile("Pr_([0-9]+)"), re.compile("Pr([0-9]+)"), re.compile("P(([0-9]+_)+)")]
                probes_spikes=[{int(pr) for r in rs if not r.search(k) is None for g in r.search(k).groups() for pr in re.findall("[0-9]+", g)} for k in spike_keys]
                electrodes=[{raw_keys[probes_raw.index(pr)] if pr in probes_raw else None for pr in l } for l in probes_spikes]
        else:
            spike_keys=[]
            duration_spike=[]
            probes_spikes=[]
            electrodes=[]
        res= pd.DataFrame(
            [[k, None, "raw", fs, toolbox.DataPath(raw_file_path, [k, "values"]), dur, pr] for k,fs, dur, pr in zip(raw_keys, fs, duration_raw, probes_raw)] + 
            [[e, k, "spike_times", 1, toolbox.DataPath(spike_file_path, [k, "times"]), dur, pr] for e, k,dur,pr in zip(electrodes, spike_keys, duration_spike, probes_spikes)], 
            columns=["Electrode", "Unit","signal_type", "signal_fs", "signal_path", "Duration", "Probes"], dtype=object)
        # print(res.dtypes)
        # input()
        return res


        
    
