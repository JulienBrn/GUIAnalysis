
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class ReadMonkeyDataBase(GUIDataFrame):
    def __init__(self, computation_m):
        super().__init__("inputs.monkey.db.read", 
            {
                "inputs.monkey.db.path":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review/BothMonkData_withTime.csv",
                "inputs.monkey.db.separator":",",
            }, computation_m)
        self.computation_m = computation_m
        
    
    def compute_df(self):
        df = pd.read_csv(str(self.metadata["inputs.monkey.db.path"]), sep=self.metadata["inputs.monkey.db.separator"])
        return df



        
    
