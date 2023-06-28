
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox

class AddRatMetadata(GUIDataFrame):
    def __init__(self, db, computation_m):
        super().__init__("inputs.rat.signals.metadata", {}, computation_m, {"db":db})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        df =  db.loc[~(db["Structure"].isna() | db["Structure"].astype(str).str.contains("None")), :].copy().reset_index(drop=True)
        df["Species"] = "Rat"
        df["Healthy"] = ~df["Condition"].str.contains("Park")
        df["Structure"] = df.apply(lambda row: row["Structure"] if row["Structure"]!="Striatum" else "STR", axis=1)
        df["signal_type"] = df["signal_type"].str.lower()
        df["Session"] = "RS"+ df["Session"]+"#" + df.groupby(by=["Date", "Subject", "Condition"]).ngroup().astype(str)
        df = pd.pivot(df, index=[col for col in df.columns if not col in ["signal_type", "Files"]], columns=["signal_type"], values="Files")
        for c in df.columns:
            df["{}_file_path".format(c)] = df.pop(c)
        df.reset_index(inplace=True)
        df["Start"] = 0
        def get_duration(fp, raw_fs):
            mat = scipy.io.loadmat(self.metadata["inputs.rat.files.base_folder"] + "/" + fp)
            dur = np.squeeze(mat["MUA"]).size / raw_fs
            return dur
        
        return df



        
    
