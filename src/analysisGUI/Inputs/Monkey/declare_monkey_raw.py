
from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class DeclareMonkeyRaw(GUIDataFrame):
    def __init__(self, monkey_md, computation_m: toolbox.Manager):
        super().__init__("inputs.monkey.signals.raw", {"inputs.monkey.files.base_folder":"/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review"}, computation_m, {"db":monkey_md})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame, inputs_monkey_files_base_folder):
        raw_df = db.groupby(["Date","Species", "Condition", "Healthy", "Subject", "Electrode", "Depth", "Structure", "Session"]).aggregate(lambda x: tuple(x)).reset_index()
        raw_df.insert(0, "signal_fs", 25000)
        raw_df.insert(0, "signal_type", "raw")
        raw_df.insert(len(raw_df.columns), "signal_path", raw_df.apply(lambda row: [(s, e, toolbox.DataPath(f, ["RAW"])) for s,e,f in zip(row["Start"], row["End"], row["Files"])], axis=1))
        raw_df.insert(0, "nb_subsignals", raw_df["signal_path"].apply(len))
        raw_df.insert(0, "Start", raw_df.pop("Start").apply(min))
        raw_df.insert(0, "End", raw_df.pop("End").apply(max))
        raw_df.pop("Duration")
        raw_df.insert(0, "Duration", raw_df["End"] - raw_df["Start"])

        self.tqdm.pandas(desc="Declaring signal matrix")
        raw_df.insert(0, "signal_matrix",raw_df.progress_apply(
            lambda row: self.computation_m.declare_computable_ressource(
                lambda dpl, fs: mk_matrix(dpl, inputs_monkey_files_base_folder, fs),
                {"dpl": row["signal_path"], "fs": row["signal_fs"]},
                toolbox.np_loader, "monkey_input_matrix", False),
            axis=1
        ))

        self.tqdm.pandas(desc="Declaring shifts")
        raw_df.insert(0, "Shift", raw_df.progress_apply(
            lambda row: self.computation_m.declare_computable_ressource(
                lambda matrix, amount: find_shift(matrix, amount),
                {"matrix": row["signal_matrix"], "amount": row["signal_fs"]},
                toolbox.json_loader, "monkey_input_shifts", True),
            axis=1
        ))

        self.tqdm.pandas(desc="Declaring raw signals")
        raw_df.insert(0, "signal", raw_df.progress_apply(
            lambda row: self.computation_m.declare_computable_ressource(
                lambda matrix, shift: mk_signal(shift_matrix(matrix, shift)) if not shift is np.nan else toolbox.Error("No shift found"),
                {"matrix": row["signal_matrix"], "shift": row["Shift"]},
                toolbox.np_loader, "monkey_input_raw_signals", True),
            axis=1
        ))
        raw_df = raw_df.sort_values("nb_subsignals", ascending=False)

        return raw_df.drop(columns=["Source", "Files"])


def mk_matrix(dpl, base_folder, fs):
    # print("Loading")
    new_dpl=[]
    for s,e,dp in dpl:
        mat =  scipy.io.loadmat(str(base_folder)+"/"+dp.file_path, variable_names=dp.keys[0])
        raw = np.squeeze(mat[dp.keys[0]])
        # print("raw", raw.shape)
        new_dpl.append((int(fs*s), int(fs*s+raw.size), raw))
    start = min([s for s, e,d in new_dpl])
    end = max([e for s, e,d in new_dpl])
    res = np.empty(shape=(len(new_dpl), end-start))
    res[:] = np.nan
    for i, (s,e,r) in enumerate(new_dpl):
        res[i,s-start:e-start] = r
    # print("input:", res)
    return res

def shift_matrix(matrix, shift):
    
    nsize = matrix.shape[1] + (max(shift) - min(shift))
    res = np.empty((matrix.shape[0], nsize))
    res[:] = np.nan
    for i in range(matrix.shape[0]):
        res[i, shift[i]-min(shift):shift[i]-min(shift)+matrix.shape[1]] = matrix[i, :]
    return res

def mk_signal(matrix):
    return np.min(matrix, axis=0)

def eval(matrix: np.ndarray):
    if matrix.shape[0] > 2:
        sorted = np.sort(matrix,axis=0)
    else:
        sorted = matrix
    
    nb_identical = ((sorted[1:, :] == sorted[:-1,:])).sum(axis=0).sum()
    nb_different = ((sorted[1:, :] != sorted[:-1,:]) & ~np.isnan(sorted[:-1, :]) & ~np.isnan(sorted[1:, :])).sum(axis=0).sum()
    return nb_identical, nb_different

from tqdm import tqdm

def find_shift(matrix, max_amount):
    import itertools
    tests = list(itertools.product(toolbox.crange(0, max_amount), repeat=matrix.shape[0] -1))
    relevant_indices = np.argwhere(~np.isnan(matrix[0, :])).flatten()
    relevant_indices = relevant_indices[::(int(len(relevant_indices)/100)+1)]
    if len(tests) > 1:
        progress = lambda x: tqdm(x, desc = "Finding shift")
    else:
        progress = lambda x: x
    if len(tests) == 0:
        logger.error("Test size zero in find shift")
    for shift in progress(list(tests)):
        shift = [0] + list(shift)
        nb_identical, nb_different = eval(shift_matrix(matrix[:, relevant_indices], shift))
        if nb_different!=0:
            continue
        else:
            nb_identical, nb_different = eval(shift_matrix(matrix, shift))
            if nb_different==0:
                break
    if nb_different==0:
        return shift
    else:
        return np.nan


