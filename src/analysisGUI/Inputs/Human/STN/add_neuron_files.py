
from analysisGUI.gui import GUIDataFrame
import pathlib, logging
import pandas as pd, numpy as np, scipy
import toolbox

logger=logging.getLogger(__name__)

class ParseHumanSTNDataBase(GUIDataFrame):
    def __init__(self, rd_human_stn_db, computation_m):
        super().__init__("inputs.human.stn.db.parsed", {"inputs.human.stn.isolation.threshold": "0.6"}, computation_m, {"db":rd_human_stn_db}, alternative_names=["inputs.human.stn.db"])
        self.computation_m = computation_m
    
    def compute_df(self, db):
        df =  db.copy()
        for i in range(4):
            df["neuron_data{}".format(i)] = list(zip(df["Unit {} Rate".format(i+1)], df["Unit {} Isolaton".format(i+1)]))
            df.drop(columns = ["Unit {} Rate".format(i+1), "Unit {} Isolaton".format(i+1)], inplace=True)

        neuron_df = pd.wide_to_long(df, stubnames="neuron_data", i = ["Species", "Condition", "Structure", "Date", "Hemisphere", "Electrode", "Depth", "Subject", "file"], j="neuron_num").reset_index()

        neuron_df["Rate"] = neuron_df["neuron_data"].str[0]
        neuron_df["Isolation"] = neuron_df["neuron_data"].str[1]
        neuron_df.drop(columns = ["neuron_data"], inplace=True)
        neuron_df = neuron_df.loc[~neuron_df["Rate"].isna()].reset_index()
        neuron_df = neuron_df.loc[neuron_df["Isolation"] > self.metadata["inputs.human.stn.isolation.threshold"]].reset_index()

        neuron_df["spike_file_path"] = neuron_df["Structure"] + "/" + neuron_df['StructDateH'].str.slice(5) + "/sorting results_"+ neuron_df['StructDateH'].str.slice(5) + "-" + neuron_df["file"].str[0:-4] + "-channel" + neuron_df["Electrode"].astype(str) + "-1.mat"
        neuron_df["signal_path"] = neuron_df.apply(lambda row: row["spike_file_path"], axis=1)
        neuron_df["signal_fs"] = 1
        neuron_df["signal_type"] = "spike_times"

        # def declare_spike_sig(signal_path, neuron):
        #     mat = toolbox.matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+ signal_path, variable_names="sortingResults")
        #     spike_df = np.squeeze(mat["sortingResults"])[()]
        #     spike_arr = np.squeeze(spike_df[6])
        #     res = pd.DataFrame(spike_arr, columns=["neuron_num", "spike_time"] + ["pca_{}".format(i) for i in range(spike_arr.shape[1]-2)])
        #     spike = res.loc[res["neuron_num"]==neuron+1, "spike_time"].to_numpy()
        #     # logger.info("Got ressource {}, neuron = {}. Shape is {}".format(signal_path, neuron, spike.shape))
        #     return spike
        
        # neuron_df["signal"] = neuron_df.apply(lambda row: computation_m.declare_computable_ressource(declare_spike_sig, {"signal_path": row["signal_path"], "neuron": row["neuron_num"]}, toolbox.np_loader, "human_input_spike", True), axis=1)
    



        
    
