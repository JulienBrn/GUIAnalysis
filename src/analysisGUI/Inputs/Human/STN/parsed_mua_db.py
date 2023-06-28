
from analysisGUI.gui import GUIDataFrame
import pathlib, logging
import pandas as pd, numpy as np, scipy
import toolbox

logger=logging.getLogger(__name__)

class ParsedHumanSTNMUADataBase(GUIDataFrame):
    def __init__(self, parsed_human_stn_db, computation_m):
        super().__init__("inputs.human.stn.signals.mua", {"inputs.human.stn.files.base_folder":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All"}, computation_m, {"db":parsed_human_stn_db})
        self.computation_m = computation_m
    
    def compute_df(self, db):
        self.tqdm.pandas(desc="Computing human stn mua")
        raw_df =  db.copy()
        raw_df["file_keys"] = raw_df.progress_apply(lambda row: ["CElectrode{}".format(row["Electrode"])], axis=1) 
        raw_df["signal_path"] = raw_df.progress_apply(lambda row: toolbox.DataPath(row["file_path"], row["file_keys"]), axis=1)
        raw_df["signal_fs_path"] = raw_df.progress_apply(lambda row: toolbox.DataPath(row["file_path"], ["CElectrode{}_KHz".format(row["Electrode"])]), axis=1)
        raw_df["signal_type"] = "mua"
        base_folder = self.metadata["inputs.human.stn.files.base_folder"]
        def declare_raw_sig(dp):
            mat =  scipy.io.loadmat(base_folder+"/"+dp.file_path, variable_names=dp.keys[0])
            raw = np.squeeze(mat[dp.keys[0]])
            # logger.info("Got ressource {}. Shape is {}".format(dp, raw.shape))
            return raw
        def declare_raw_fs(dp):
            mat =  scipy.io.loadmat(base_folder+"/"+dp.file_path, variable_names=dp.keys[0])
            fs = np.squeeze(mat[dp.keys[0]])*1000
            # logger.info("Got ressource {}. Value is {}".format(dp, fs))
            return fs
        
        raw_df["signal"] = raw_df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(declare_raw_sig, {"dp": row["signal_path"]}, toolbox.np_loader, "human_input_mua", True), axis=1)
        raw_df["signal_fs"] = raw_df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(declare_raw_fs, {"dp": row["signal_fs_path"]}, toolbox.float_loader, "human_input_fs", True), axis=1)
    
        for i in range(4):
            raw_df.drop(columns = ["Unit {} Rate".format(i+1), "Unit {} Isolaton".format(i+1)], inplace=True)
        return raw_df.drop(columns=["file", "file_keys", "signal_fs_path"])



        
    
