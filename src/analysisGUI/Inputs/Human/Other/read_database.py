
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np
import toolbox

class ReadHumanOtherDataBase(GUIDataFrame):
    def __init__(self, computation_m):
        super().__init__("inputs.human.other.db.read", 
            {
                "inputs.human.other.db.path":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review/AllHumanData.mat",
            }, computation_m)
        self.computation_m = computation_m
        
    
    def compute_df(self):
        m = toolbox.matlab_loader.load(self.metadata["inputs.human.other.db.path"])
        df=pd.DataFrame(m["AllHumanData"])
        for col in df.columns:
            df[col] = df.apply(lambda row: np.reshape(row[col], -1)[0] if row[col].size == 1 else None if row[col].size == 0 else row[col], axis=1).astype(str)
        df.columns=["Condition", "Structure", "DateH", "ElectrodeDepth", "Unit"]
        return df



        
    
