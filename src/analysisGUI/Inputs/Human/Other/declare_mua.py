
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class DeclareHumanOtherMUA(GUIDataFrame):
    def __init__(self, md, computation_m: toolbox.Manager):
        super().__init__("inputs.human.other.signals.mua", {"inputs.human.other.files.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review/"}, computation_m, {"db":md})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame, inputs_human_other_files_base_folder):
        raw_df = db.copy()
        raw_df = raw_df.groupby(["Condition", "Structure", "DateH", "ElectrodeDepth"]).first().reset_index()
        raw_df["signal_type"] = "mua"
        raw_df["signal_fs"] = raw_df["Date"].apply(lambda d: 48000 if d < "2015_01_01" else 44000)
        raw_df["signal_path"] = raw_df["file_path"].apply(lambda fp: toolbox.DataPath(fp, ["MUA"]))
        def get_sig(fp):
            mat = scipy.io.loadmat(inputs_human_other_files_base_folder + "/" + fp)
            sig = np.reshape(mat["MUA"], -1)
            return sig
        raw_df["signal"] = raw_df["file_path"].apply(lambda fp: self.computation_m.declare_computable_ressource(get_sig, {"fp": fp}, toolbox.np_loader, "human_other_raw_input", True))
        return raw_df.drop(columns=["Source", "Unit"])

