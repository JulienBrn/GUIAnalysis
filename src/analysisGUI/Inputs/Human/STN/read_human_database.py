
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np
import toolbox
import json

class ReadHumanSTNDataBase(GUIDataFrame):
    def __init__(self, list_db, computation_m):
        super().__init__("inputs.human.stn.db.read", 
            {
                "inputs.human.stn.db.separator":",",
            }, computation_m, {"list_db": list_db}, save=True)
        self.computation_m = computation_m
        
    
    def compute_df(self, list_db):
        dfs=[]
        for entry in list_db["Db_File_Path"]:
            entry: toolbox.DataPath
            mat = toolbox.matlab_loader.load(entry.file_path)
            df = pd.DataFrame(mat[entry.keys[0]])
            for col in df.columns:
                df[col] = df.apply(lambda row: np.reshape(row[col], -1)[0] if row[col].size == 1 else None if row[col].size == 0 else row[col], axis=1)
                if df[col].isnull().all():
                    df.drop(columns=[col], inplace=True)
            df.columns = [str(s) for s in df.iloc[0, :].to_list()]
            df = df.iloc[1:, :]
            df.columns = ["StructDateH"] + list(df.columns[1:])
            df.insert(0, "Entry", entry)
            dfs.append(df)
        df = pd.concat(dfs)
        return df



        
    
