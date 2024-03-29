
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class ReadHumanSTNFiles(GUIDataFrame):
    def __init__(self, computation_m):
        super().__init__("inputs.human.stn.files.list", 
            {
                "inputs.human.stn.files.base_folder":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All"
            }, computation_m, save=True, alternative_names=["inputs.human.stn.files"])
        self.computation_m = computation_m
        

    def compute_df(self, inputs_human_stn_files_base_folder):
        base_folder = pathlib.Path(inputs_human_stn_files_base_folder)
        files = toolbox.find_files_recursively(str(base_folder), tqdm=self.tqdm)
        files = [pathlib.Path(f) for f in files]
        return pd.DataFrame([[str(f.relative_to(base_folder))] for f in files if f.is_file()], columns = ["Files"])
    
