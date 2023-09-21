
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np
import toolbox
import ast, itertools

class HumanSTNDatabaseFiles(GUIDataFrame):
    def __init__(self, computation_m):
        super().__init__("inputs.human.stn.db.list", 
            {
                "inputs.human.stn.db.base_folder":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All",
                "inputs.human.stn.db.file_pattern":"{base_folder}/all_Sorting_data_{structure}_Part{part}.mat",
                "inputs.human.stn.db.key_pattern":"allSortingResultsDatabase_{structure}_{part}",
                "inputs.human.stn.db.part_range":"(1, 4)",
                "inputs.human.stn.db.structure":"('DLOR', 'VMNR')",
            }, computation_m)
        self.computation_m = computation_m
        
    
    def compute_df(self, inputs_human_stn_db_base_folder, inputs_human_stn_db_file_pattern, inputs_human_stn_db_key_pattern, inputs_human_stn_db_part_range, inputs_human_stn_db_structure):
        file_pattern = inputs_human_stn_db_file_pattern
        key_pattern = inputs_human_stn_db_key_pattern
        base_folder = inputs_human_stn_db_base_folder
        part_range = ast.literal_eval(inputs_human_stn_db_part_range)
        part_range = range(part_range[0], part_range[1])
        structure = ast.literal_eval(inputs_human_stn_db_structure)

        l=[]
        for part, structure in itertools.product(part_range, structure):
            dict_eval={"base_folder":base_folder, "part":part, "structure":structure}
            l.append(toolbox.DataPath(file_pattern.format(**dict_eval), [key_pattern.format(**dict_eval)]))
        df = pd.DataFrame([[f] for f in l], columns=["Db_File_Path"])
        return df



        
    
