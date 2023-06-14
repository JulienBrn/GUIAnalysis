
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class ParseHumanOtherFiles(GUIDataFrame):
    def __init__(self, rd_human_files, computation_m):
        super().__init__("inputs.human.other.files.parsed", {}, computation_m, {"files":rd_human_files}, alternative_names=["inputs.human.files"])
        self.computation_m = computation_m
    
    def compute_df(self, files):
        df =  toolbox.extract_wildcards(files["Files"].to_list(), "{Structure}/{DateH}/{ElectrodeDepth}/{Unit}.mat", tqdm = self.tqdm)
        df["Unit"] = df["Unit"].str.extract("(\d+)")
        return df


        
    
