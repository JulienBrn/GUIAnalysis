
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class ParseMonkeyDataBase(GUIDataFrame):
    def __init__(self, rd_monkey_db, computation_m):
        super().__init__("inputs.monkey.db.parsed", {}, computation_m, {"db":rd_monkey_db}, save=True, alternative_names=["inputs.monkey.db"])
        self.computation_m = computation_m
        self.rd_monkey_db = rd_monkey_db
    
    def compute_df(self, db):
        df =  db.copy()
        df.columns=["Condition", "Subject", "Structure", "Date", "Electrode", "Unit", "Start", "End"]
        df = df.iloc[1:, :]
        df["Structure"] = df["Structure"].str.slice(0, 3)
        df["Start"] = df["Start"].astype(float)
        df["End"] = df["End"].astype(float)
        return df



        
    
