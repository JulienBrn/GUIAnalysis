
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox

class ParseHumanSTNDataBase(GUIDataFrame):
    def __init__(self, rd_human_stn_db, computation_m):
        super().__init__("inputs.human.stn.db.parsed", {"inputs.human.stn.db.base_folder":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All",}, computation_m, {"db":rd_human_stn_db}, alternative_names=["inputs.human.stn.db"])
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame, inputs_human_stn_db_base_folder):
        df =  db.copy()
        meta = df['StructDateH'].astype(str).str.isdigit()
        df.loc[meta, 'StructDateH'] = np.nan
        df['StructDateH'].ffill(inplace=True)
        df['StructDateH'] = df['StructDateH'].str.replace("\\", "/")
        df= df.loc[meta, :].reset_index(drop=True)
        df["Species"] = "Human"
        df["Condition"] = "Park"
        df["Healthy"] = df["Condition"] != "Park"
        df["Structure"] = "STN_"+ df['StructDateH'].str.slice(0,4)
        df["Date"] = df['StructDateH'].str.slice(5,15)
        df["Hemisphere"] = df['StructDateH'].str.slice(16)
        df["Electrode"] = df.pop("channel").astype(int)
        df["Depth"] = df["file"].str.extract('(\d+)').astype(str)
        df["Subject"] = np.nan
        df["Session"] = "HS#" + df.groupby(by=["Date", "Hemisphere", "Depth"]).ngroup().astype(str)
        df["file_path"] = df["Structure"] + "/" + df['StructDateH'].str.slice(5) + "/"+ df["file"].str.replace("map", "mat")
        df["Start"] = 0
        def get_duration(fp):
            mat = scipy.io.loadmat(inputs_human_stn_db_base_folder+"/" +fp, variable_names=['CElectrode1_TimeBegin', 'CElectrode1_TimeEnd'])
            dur= np.squeeze(mat['CElectrode1_TimeEnd']) - np.squeeze(mat['CElectrode1_TimeBegin'])
            return dur
        self.tqdm.pandas(desc="Declaring durations")
        df["Duration"] = df["file_path"].apply(lambda fp: self.computation_m.declare_computable_ressource(get_duration, {"fp":fp}, toolbox.float_loader, "human_input_durations", True))
        df.insert(0, "_Discarded", df[(~df["Depth"].isna())].duplicated(subset=["Session", "Depth","Electrode"], keep=False))
        df = df.sort_values(["_Discarded", "Session", "Depth", "Electrode"], ignore_index=True, ascending=False)
        # df = df[df["Maybe Duplicate"]!=True].reset_index(drop=True)
        return df



        
    
