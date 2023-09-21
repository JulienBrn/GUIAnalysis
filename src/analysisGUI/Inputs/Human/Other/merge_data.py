
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class MergeHumanOtherData(GUIDataFrame):
    def __init__(self, parsed, db, computation_m):
        super().__init__("inputs.human.other.merged.merged", 
            {
            }, computation_m, {"files": parsed, "db":db})
        self.computation_m = computation_m
        
    
    def compute_df(self, files: pd.DataFrame, db: pd.DataFrame):
        filtered = files.loc[~(files["Structure"].isna() | files["Structure"].astype(str).str.contains("None")), :].reset_index(drop=True)
        filtered["is_from_files"] = True
        db_df = db.copy()
        db_df["is_from_db"] = True
        df = pd.merge(filtered, db_df, how="outer")
        df["Source"] = df.apply(lambda row: "Files + DB" if row["is_from_files"] is True and row["is_from_db"] is True else "Files" if row["is_from_files"] is True else "DB", axis=1)
        df = df.sort_values("Source", ignore_index=True)
        df["_Discarded"] = df["Source"] != "Files + DB"
        return df.drop(columns=["is_from_files", "is_from_db"])



        
    
