
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class MergeHumanSTNData(GUIDataFrame):
    def __init__(self, human_parsed, mua_db, spike_db, computation_m):
        super().__init__("inputs.human.stn.signals.merged", 
            {
            }, computation_m, {"files": human_parsed, "mua":mua_db, "spikes": spike_db})
        self.computation_m = computation_m
        
    
    def compute_df(self, files: pd.DataFrame, mua: pd.DataFrame, spikes: pd.DataFrame):
        filtered = files.loc[~(files["Structure"].isna() | files["Structure"].astype(str).str.contains("None")), :].reset_index(drop=True)
        filtered["is_from_files"] = True
        db_df = pd.concat([mua, spikes], ignore_index=True)
        db_df["is_from_db"] = True
        db_df["Files"] = db_df["signal_path"].apply(lambda x: x.file_path)
        df = pd.merge(filtered, db_df, how="outer", on=["Files", "Structure"])
        df.insert(0, "Source", df.apply(lambda row: "Files + DB" if row["is_from_files"] is True and row["is_from_db"] is True else "Files" if row["is_from_files"] is True else "DB", axis=1))
        df = df.sort_values("Source", ignore_index=True)
        df["Discarded"] = df["Source"] != "Files + DB"
        return df.drop(columns=["is_from_files", "is_from_db", "Files"])



        
    
