
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class ParseMonkeyFiles(GUIDataFrame):
    def __init__(self, rd_monkey_files, computation_m):
        super().__init__("inputs.monkey.files.parsed", {}, computation_m, {"files":rd_monkey_files}, alternative_names=["inputs.monkey.files"])
        self.computation_m = computation_m
    
    def compute_df(self, files):
        df =  toolbox.extract_wildcards(files["Files"].to_list(), "{Condition}/{Subject}/{Structure}/{Date}/{Unit}.mat", tqdm = self.tqdm)
        df["Unit"] = df["Unit"].str.slice(len("unit"))
        return df


    def get_filtered_df(self):
        df = self.get_df()
        return df[df["Condition"]!=None].reset_index(drop=True)


        
    
