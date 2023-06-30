
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class ParseRatFiles(GUIDataFrame):
    def __init__(self, rd_files, computation_m):
        super().__init__("inputs.rat.files.parsed", {}, computation_m, {"files":rd_files}, alternative_names=["inputs.rat.files"])
        self.computation_m = computation_m
    
    def compute_df(self, files):
        ["Condition", "Subject", "Date", "Session", "Structure"]
        df =  toolbox.extract_wildcards(files["Files"].to_list(), "{Condition}/{Subject}/{Date}/{Session}/{Structure}/{signal_type}.mat", tqdm = self.tqdm)
        return df


        
    
