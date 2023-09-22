
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox

class AddHumanOtherMetadata(GUIDataFrame):
    def __init__(self, merged_db, computation_m):
        super().__init__("inputs.human.other.merged.metadata", {"inputs.human.other.files.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review/"}, computation_m, {"db":merged_db}, alternative_names=["inputs.human.other.merged"])
        self.computation_m = computation_m
    
    def compute_df(self, db, inputs_human_other_files_base_folder):
        df =  db[db["Source"]=="Files + DB"].copy()
        df["Species"] = "Human"
        df["Healthy"] = df["Condition"] != "PD"
        df["Hemisphere"] = df['DateH'].str.extract(r'_([RL])(\d?)($|_)')[0]
        df["Date"] = df['DateH'].str.slice(0, 10)
        df["Electrode"] = df["ElectrodeDepth"].str.slice(0,2)
        df["Depth"] = df["ElectrodeDepth"].str.slice(3)
        df["Subject"] = np.nan
        df["file_path"] = df.pop("Files")
        df["Start"] = 0
        df["Session"] = "HO#" + df.groupby(by=["Date", "Hemisphere", "Electrode", "Depth"]).ngroup().astype(str)
        df["raw_fs"] = df["Date"].apply(lambda d: 48000 if d < "2015_01_01" else 44000)
        def get_duration(fp, raw_fs):
            mat = scipy.io.loadmat(inputs_human_other_files_base_folder + "/" + fp)
            dur = np.squeeze(mat["MUA"]).size / raw_fs
            return dur
        df["Duration"] = df.apply(lambda row: self.computation_m.declare_computable_ressource(get_duration, {"fp": row["file_path"], "raw_fs": row["raw_fs"]}, toolbox.float_loader, "human_other_input_duration", True), axis=1)
    
        return df.drop(columns=["raw_fs"])



        
    
