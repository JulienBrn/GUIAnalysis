
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox

class ParseHumanSTNFiles(GUIDataFrame):
    def __init__(self, rd_human_files, computation_m):
        super().__init__("inputs.human.stn.files.parsed", {}, computation_m, {"files":rd_human_files}, alternative_names=["inputs.human.stn.files"])
        self.computation_m = computation_m
    
    def compute_df(self, files):
        df =  toolbox.extract_wildcards(files["Files"].to_list(), "{Structure}/{DateH}/{file}.mat", tqdm = self.tqdm)
        df['StructDateH'] = df["Structure"].str.slice(4) +"/"+ df["DateH"].astype(str)
        # df["Species"] = "Human"
        # df["Condition"] = "Park"
        # df["Structure"] = "STN_"+ df['StructDateH'].str.slice(0,4)
        # df["Date"] = df['DateH'].str.slice(0,10)
        # df["Hemisphere"] = df['StructDateH'].str.slice(11)
        # df["Electrode"] = df["channel"]
        # df["Depth"] = df["Depth"].str.extract('(\d+)').astype(str)
        # df["Subject"] = np.nan
        # df["Start"] = 0
        # def get_duration(fp):
        #     mat = scipy.io.loadmat("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+fp, variable_names=['CElectrode1_TimeBegin', 'CElectrode1_TimeEnd'])
        #     dur= np.squeeze(mat['CElectrode1_TimeEnd']) - np.squeeze(mat['CElectrode1_TimeBegin'])
        #     return dur
        # self.tqdm.pandas(desc="Declaring durations")
        # df["End"] = df["file_path"].apply(lambda fp: self.computation_m.declare_computable_ressource(get_duration, {"fp":fp}, toolbox.float_loader, "human_input_durations", True))
        return df


    def get_filtered_df(self):
        df = self.get_df()
        return df[df["Condition"]!=None].reset_index(drop=True)


        
    
