
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np
import toolbox

class AddMonkeyMetadata(GUIDataFrame):
    def __init__(self, merged_monkey_db, computation_m):
        super().__init__("inputs.monkey.merged.metadata", {}, computation_m, {"db":merged_monkey_db}, alternative_names=["inputs.monkey.merged"])
        self.computation_m = computation_m
    
    def compute_df(self, db):
        df =  db[db["Source"]=="Files + DB"].copy()
        df["Species"] = "Monkey"
        df["Healthy"] = df["Condition"] == "healthy"
        df_depth = df.sort_values(by=["Date","Species","Condition", "Subject", "Electrode",  "Structure", "Start", "End"])
        df_depth["diff"] = (df_depth.shift(1, fill_value=-np.inf)["End"] <= df_depth["Start"]).astype(int)
        df_depth["Depth_num"] = df_depth.groupby(by=["Date","Species", "Condition", "Subject", "Electrode",  "Structure"])["diff"].cumsum()
        df_depth["Depth"] = "#"+ df_depth["Depth_num"].astype(str)
        df_depth["Session"] = "MS#" + df_depth.groupby(by=["Date", "Subject"]).ngroup().astype(str)
        return df_depth.drop(columns=["diff", "Depth_num"])
        # return df_depth



        
    
