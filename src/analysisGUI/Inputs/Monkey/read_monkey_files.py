
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd
import toolbox

class ReadMonkeyFiles(GUIDataFrame):
    def __init__(self, computation_m):
        super().__init__("inputs.monkey.files.list", 
            {
                "inputs.monkey.files.base_folder":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review"
            }, computation_m, save=True)
        self.computation_m = computation_m
        

    def compute_df(self):
        base_folder = pathlib.Path(self.metadata["inputs.monkey.files.base_folder"])
        files = toolbox.find_files_recursively(self.metadata["inputs.monkey.files.base_folder"], tqdm=self.tqdm)
        files = [pathlib.Path(f) for f in files]
        return pd.DataFrame([[str(f.relative_to(base_folder))] for f in files if f.is_file()], columns = ["Files"])
    
