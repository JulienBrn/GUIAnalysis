
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class FilterHumanSTNData(GUIDataFrame):
    def __init__(self, merged_db, computation_m):
        super().__init__("inputs.human.stn.signals.filtered", 
            {"inputs.human.stn.filter.isolation":"0.6"}, computation_m, {"db": merged_db}, alternative_names=["inputs.human.stn", "inputs.human.stn.signals"])
        self.computation_m = computation_m
        
    
    def compute_df(self, db: pd.DataFrame, inputs_human_stn_filter_isolation):
        df =  db[db["Source"]=="Files + DB"].copy()
        df["Discarded_Isolation"] =~((df["signal_type"]=="mua") | (df["Isolation"].astype(float)>=float(inputs_human_stn_filter_isolation)))
        df["DiscardedInt"] = df["Discarded_Isolation"].astype(int)
        df["nb_units_discarded"] = df.groupby(["Session", "Electrode"], group_keys=False)["DiscardedInt"].transform("sum")
        df["Discarded"] = df["Discarded_Isolation"].astype(bool) | (pd.to_numeric(df["number of units"], errors="coerce") - df["nb_units_discarded"] <=0)
        # df = df[(df["signal_type"]=="mua") | (df["Isolation"].astype(float)>=float(self.metadata["inputs.human.stn.filter.isolation"]))].reset_index(drop=True)
        return df.drop(columns=["Source", "file", "file_path", "Entry", "DateH", "StructDateH", "DiscardedInt", "Discarded_Isolation"])


        
    
