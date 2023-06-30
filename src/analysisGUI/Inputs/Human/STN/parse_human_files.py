
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox

class ParseHumanSTNFiles(GUIDataFrame):
    def __init__(self, rd_human_files, computation_m):
        super().__init__("inputs.human.stn.files.parsed", {}, computation_m, {"files":rd_human_files}, alternative_names=["inputs.human.stn.files"])
        self.computation_m = computation_m
    
    def compute_df(self, files):
        df =  toolbox.extract_wildcards(files["Files"].to_list(), "{Structure}/{DateH}/{file}.mat", tqdm = self.tqdm)
        return df


    def get_filtered_df(self):
        df = self.get_df()
        return df[df["Condition"]!=None].reset_index(drop=True)


        
    
