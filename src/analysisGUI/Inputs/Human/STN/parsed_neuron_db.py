
from toolbox import GUIDataFrame
import pathlib, logging
import pandas as pd, numpy as np, scipy
import toolbox

logger=logging.getLogger(__name__)

class ParsedHumanSTNNeuronDataBase(GUIDataFrame):
    def __init__(self, parsed_human_stn_db, computation_m):
        super().__init__("inputs.human.stn.signals.neurons",{"inputs.human.stn.files.base_folder":
                    "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All"}, computation_m, {"db":parsed_human_stn_db})
        self.computation_m = computation_m
    
    def compute_df(self, db, inputs_human_stn_files_base_folder):
        self.tqdm.pandas(desc="Computing human stn neurons")
        df =  db.copy()
        for i in range(4):
            df["neuron_data{}".format(i)] = list(zip(df["Unit {} Rate".format(i+1)], df["Unit {} Isolaton".format(i+1)]))
            df.drop(columns = ["Unit {} Rate".format(i+1), "Unit {} Isolaton".format(i+1)], inplace=True)

        logger.info("wide to long")
        neuron_df = pd.wide_to_long(df, stubnames="neuron_data", i = [col for col in df.columns if not "neuron_data" in col], j="neuron_num").reset_index()
        logger.info("wide to long done")

        neuron_df["Rate"] = neuron_df["neuron_data"].str[0]
        neuron_df["Isolation"] = neuron_df["neuron_data"].str[1]
        neuron_df.drop(columns = ["neuron_data"], inplace=True)
        neuron_df = neuron_df.loc[~neuron_df["Rate"].isna()].reset_index(drop=True)
       
        neuron_df["spike_file_path"] = neuron_df["Structure"] + "/" + neuron_df['StructDateH'].str.slice(5) + "/sorting results_"+ neuron_df['StructDateH'].str.slice(5) + "-" + neuron_df["file"].str[0:-4] + "-channel" + neuron_df["Electrode"].astype(int).astype(str) + "-1.mat"
        neuron_df["signal_path"] = neuron_df.pop("spike_file_path").progress_apply(lambda s:toolbox.DataPath(s, ["ComplexKey"]))
        # neuron_df["signal_fs"] = 1
        neuron_df["signal_type"] = "spike_times"


        neuron_df["signal_fs_path"] = neuron_df.progress_apply(lambda row: toolbox.DataPath(row["file_path"], ["CElectrode{}_KHz".format(row["Electrode"])]), axis=1)
        def declare_fs(dp):
            mat =  scipy.io.loadmat(base_folder+"/"+dp.file_path, variable_names=dp.keys[0])
            fs = np.squeeze(mat[dp.keys[0]])*1000
            # logger.info("Got ressource {}. Value is {}".format(dp, fs))
            return fs
        neuron_df["signal_fs"] = neuron_df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(declare_fs, {"dp": row["signal_fs_path"]}, toolbox.float_loader, "human_input_fs", True), axis=1)
    

        base_folder = inputs_human_stn_files_base_folder
        def declare_spike_sig(signal_path, neuron):
            mat = toolbox.matlab_loader.load(base_folder+"/"+signal_path.file_path)
            spike_df = np.squeeze(mat["sortingResults"])[()]
            spike_arr = np.squeeze(spike_df[6])
            res = pd.DataFrame(spike_arr, columns=["neuron_num", "spike_time"] + ["pca_{}".format(i) for i in range(spike_arr.shape[1]-2)])
            spike = res.loc[res["neuron_num"]==neuron+1, "spike_time"].to_numpy()
            return spike
    
        neuron_df["signal"] = neuron_df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(declare_spike_sig, {"signal_path": row["signal_path"], "neuron": row["neuron_num"]}, toolbox.np_loader, "human_input_spike", True), axis=1)
        neuron_df["max_spike_time"] = neuron_df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(lambda sig, fs: float(sig[-1])/fs, {"sig": row["signal"], "fs": row["signal_fs"]}, toolbox.float_loader, "human_other_spike_input_max", False), axis=1)
        



        return neuron_df.drop(columns=["file_path", "file", "signal_fs_path"])

       

        
    
