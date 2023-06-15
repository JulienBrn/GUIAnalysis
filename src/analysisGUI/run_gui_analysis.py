# from analysisGUI import correlationDataDF
# from analysisGUI import coherenceDataDF
# from analysisGUI import pwelchDataDF
# from analysisGUI import SpikeContinuousDataDF
# from analysisGUI import BUADataDF
# from analysisGUI import LFPDataDF
# from analysisGUI import CleanDataDF
# from analysisGUI import InputDataDF
from analysisGUI import Window
# from analysisGUI import group_coherenceDataDF
from analysisGUI.Inputs.Monkey.read_monkey_files import ReadMonkeyFiles
from analysisGUI.Inputs.Monkey.parse_monkey_files import ParseMonkeyFiles
from analysisGUI.Inputs.Monkey.read_monkey_database import ReadMonkeyDataBase
from analysisGUI.Inputs.Monkey.merge_monkey_data import MergeMonkeyData
from analysisGUI.Inputs.Monkey.parse_monkey_db import ParseMonkeyDataBase
from analysisGUI.Inputs.Monkey.add_monkey_metadata import AddMonkeyMetadata
from analysisGUI.Inputs.Monkey.declare_monkey_raw import DeclareMonkeyRaw
from analysisGUI.Inputs.Monkey.declare_monkey_spikes import DeclareMonkeySpikes
from analysisGUI.Inputs.Monkey.merge_monkey_signals import MergeMonkeySignals
from analysisGUI.Inputs.Human.STN.read_human_database import ReadHumanSTNDataBase
from analysisGUI.Inputs.Human.STN.human_stn_db_files import HumanSTNDatabaseFiles
from analysisGUI.Inputs.Human.STN.parse_human_db import ParseHumanSTNDataBase
from analysisGUI.Inputs.Human.STN.read_human_files import ReadHumanSTNFiles
from analysisGUI.Inputs.Human.STN.parse_human_files import ParseHumanSTNFiles
from analysisGUI.Inputs.Human.STN.parsed_neuron_db import ParsedHumanSTNNeuronDataBase
from analysisGUI.Inputs.Human.STN.parsed_mua_db import ParsedHumanSTNMUADataBase
from analysisGUI.Inputs.Human.STN.merge_human_data import MergeHumanSTNData
from analysisGUI.Inputs.Human.STN.filter_human_data import FilterHumanSTNData
from analysisGUI.Inputs.Human.Other.parse_files import ParseHumanOtherFiles
from analysisGUI.Inputs.Human.Other.read_database import ReadHumanOtherDataBase
from analysisGUI.Inputs.Human.Other.read_files import ReadHumanOtherFiles
from analysisGUI.Inputs.Human.Other.merge_data import MergeHumanOtherData
from analysisGUI.Inputs.Human.Other.add_metadata import AddHumanOtherMetadata
from analysisGUI.Inputs.Human.Other.declare_mua import DeclareHumanOtherMUA
from analysisGUI.Inputs.Human.Other.declare_spikes import DeclareHumanOtherSpikes
from analysisGUI.Inputs.Human.Other.merge_signals import MergeHumanOtherSignals
from analysisGUI.Inputs.Human.merged import MergeHumanSignals
from analysisGUI.Inputs.Rat.read_files import ReadRatFiles
from analysisGUI.Inputs.Rat.parse_files import ParseRatFiles
from analysisGUI.Inputs.Rat.add_metadata import AddRatMetadata
from analysisGUI.Inputs.Rat.extract_signals import ExtractRatSignals
from analysisGUI.Inputs.Rat.declare_signals import DeclareRatSignals
from analysisGUI.Inputs.all import Inputs
from analysisGUI.Signals.raw_cleaned import Clean
from analysisGUI.Signals.lfp import LFP
from analysisGUI.Signals.bua import BUA


from toolbox import Manager, json_loader, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import toolbox
import logging
import beautifullogger
import pathlib
import pandas as pd
import toolbox
import numpy as np
import scipy
import h5py
import re
import ast
import sys
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QCoreApplication

beautifullogger.setup(logmode="w")
logger = logging.getLogger(__name__)
logging.getLogger("toolbox.ressource_manager").setLevel(logging.WARNING)
logging.getLogger("toolbox.signal_analysis_toolbox").setLevel(logging.WARNING)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info("Keyboard interupt")
        # QCoreApplication.instance().quit()
        sys.exit()
        return
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception

computation_m = Manager("./cache/computation")
dataframe_manager = Manager("./cache/dataframes")


step_signals = {}



def run_gui():
    app = QApplication(sys.argv)
    win = Window()
    read_monkey_files = ReadMonkeyFiles(computation_m)
    parse_monkey_files = ParseMonkeyFiles(read_monkey_files, computation_m)
    read_monkey_db = ReadMonkeyDataBase(computation_m)
    parse_monkey_db = ParseMonkeyDataBase(read_monkey_db, computation_m)
    merge_monkey_data = MergeMonkeyData(parse_monkey_files, parse_monkey_db, computation_m)
    add_monkey_md = AddMonkeyMetadata(merge_monkey_data, computation_m)
    monkey_raw = DeclareMonkeyRaw(add_monkey_md, computation_m)
    monkey_spikes = DeclareMonkeySpikes(add_monkey_md, computation_m)
    monkey_signals = MergeMonkeySignals(monkey_raw, monkey_spikes, computation_m)
    
    read_human_stn_files = ReadHumanSTNFiles(computation_m)
    parse_human_stn_files = ParseHumanSTNFiles(read_human_stn_files, computation_m)
    human_stn_db_files = HumanSTNDatabaseFiles(computation_m)
    read_human_stn_db = ReadHumanSTNDataBase(human_stn_db_files, computation_m)
    parse_human_stn_db = ParseHumanSTNDataBase(read_human_stn_db, computation_m)
    parsed_human_stn_neuron_db = ParsedHumanSTNNeuronDataBase(parse_human_stn_db, computation_m)
    parsed_human_stn_mua_db = ParsedHumanSTNMUADataBase(parse_human_stn_db, computation_m)
    merge_human_stn_data = MergeHumanSTNData(parse_human_stn_files, parsed_human_stn_mua_db, parsed_human_stn_neuron_db, computation_m)
    filter_human_stn_data = FilterHumanSTNData(merge_human_stn_data, computation_m)

    read_human_other_files = ReadHumanOtherFiles(computation_m)
    parse_human_other_files = ParseHumanOtherFiles(read_human_other_files, computation_m)
    read_human_other_db = ReadHumanOtherDataBase(computation_m)
    merge_human_other_data = MergeHumanOtherData(parse_human_other_files, read_human_other_db, computation_m)
    add_human_other_md = AddHumanOtherMetadata(merge_human_other_data, computation_m)
    human_other_mua = DeclareHumanOtherMUA(add_human_other_md, computation_m)
    human_other_spikes = DeclareHumanOtherSpikes(add_human_other_md, computation_m)
    human_other_signals = MergeHumanOtherSignals(human_other_mua, human_other_spikes, computation_m)
    human_signals = MergeHumanSignals(filter_human_stn_data, human_other_signals, computation_m)

    read_rat_files = ReadRatFiles(computation_m)
    parse_rat_files = ParseRatFiles(read_rat_files, computation_m)
    add_rat_md = AddRatMetadata(parse_rat_files, computation_m)
    rat_signals = ExtractRatSignals(add_rat_md, computation_m)
    rat_declared_signals = DeclareRatSignals(rat_signals, computation_m)

    inputs = Inputs(human_signals, rat_declared_signals, monkey_signals, computation_m)
    cleaned = Clean(inputs, computation_m)
    lfp = LFP(cleaned, computation_m)
    bua = BUA(cleaned, computation_m)
    # input_df = InputDataDF(dataframe_manager, computation_m, step_signals)
    # clean_df = CleanDataDF(computation_m, step_signals, input_df)
    # lfp_df = LFPDataDF(computation_m, step_signals, clean_df)
    # bua_df = BUADataDF(computation_m, step_signals, clean_df)
    # spike_continuous_df = SpikeContinuousDataDF(
    #     computation_m, step_signals, input_df)
    # pwelch_df = pwelchDataDF(computation_m, step_signals, lfp_df, bua_df)
    # coherence_df = coherenceDataDF(
    #     computation_m, step_signals, lfp_df, bua_df, spike_continuous_df)
    # group_coherence_df = group_coherenceDataDF(computation_m,coherence_df)
    # correlation_df = correlationDataDF(
    #     computation_m, step_signals, spike_continuous_df)
    win.add_df(inputs)

    win.add_df(read_monkey_files)
    win.add_df(parse_monkey_files)
    win.add_df(read_monkey_db)
    win.add_df(merge_monkey_data)
    win.add_df(parse_monkey_db)
    win.add_df(add_monkey_md)
    win.add_df(monkey_raw)
    win.add_df(monkey_spikes)
    win.add_df(monkey_signals)

    win.add_df(read_human_stn_files)
    win.add_df(parse_human_stn_files)
    win.add_df(human_stn_db_files)
    win.add_df(read_human_stn_db)
    win.add_df(parse_human_stn_db)
    win.add_df(parsed_human_stn_neuron_db)
    win.add_df(parsed_human_stn_mua_db)
    win.add_df(merge_human_stn_data)
    win.add_df(filter_human_stn_data)

    win.add_df(read_human_other_files)
    win.add_df(parse_human_other_files)
    win.add_df(read_human_other_db)
    win.add_df(merge_human_other_data)
    win.add_df(add_human_other_md)
    win.add_df(human_other_mua)
    win.add_df(human_other_spikes)
    win.add_df(human_other_signals)

    win.add_df(human_signals)

    win.add_df(read_rat_files)
    win.add_df(parse_rat_files)
    win.add_df(add_rat_md)
    win.add_df(rat_signals)
    win.add_df(rat_declared_signals)

    win.add_df(cleaned)
    win.add_df(lfp)
    win.add_df(bua)
    
    # win.add_df(input_df)
    # win.add_df(clean_df)
    # win.add_df(lfp_df)
    # win.add_df(bua_df)
    # win.add_df(spike_continuous_df)
    # win.add_df(pwelch_df)
    # win.add_df(coherence_df)
    # win.add_df(group_coherence_df)
    # win.add_df(correlation_df)

    # if pathlib.Path("setup_params.json").exists():
    if False:
        default_params = win.get_setup_params()
        win.set_setup_params(json_loader.load(
            pathlib.Path("setup_params.json")))
        last_params = win.get_setup_params()
        if last_params.keys() != default_params.keys():
            logger.warning("Strange")
        else:
            res = pd.DataFrame(zip(last_params.keys(), default_params.values(
            ), last_params.values()), columns=["key", "default", "last"])
            res["same"] = res["default"] == res["last"]
            # logger.info("Params:\n{}".format(res.to_string()))
        win.on_computation_tab_clicked()
        win.tabWidget.setCurrentWidget(
            win.tabWidget.findChild(QWidget, "computation_tab"))
    else:
        win.menu_tabs.setCurrentWidget(
            win.menu_tabs.findChild(QWidget, "setup_tab"))
    # coherence_df.compute()
    # pwelch_df.compute()
    # correlation_df.compute()
    # if win.process:
    #     win.process.wait()
    # check_df = input_df.get_df().reset_index(drop=True)
    # check_df["file_path"]=check_df["file_path"].str.slice(start=len("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/"))
    # check_df = check_df.loc[check_df["Species"] == "Human", :]
    # check_df=check_df.loc[check_df["signal_type"] == "mua", :]
    # # # tqdm.pandas(desc="Loading signals")
    # # # # check_df["signal"] = check_df.progress_apply(lambda row: row["signal"].get_result(), axis =1)
    # check_df["key"] = 0
    # check_df["old_index"] = np.arange(0, len(check_df.index))
    # check_df = check_df[["signal", "key", "file_path", "file_keys", "old_index"]].reset_index(drop=True)
    # res_df = toolbox.group_and_combine(check_df, ["key"], include_eq=False)
    # res_df["sort"] = res_df["old_index_1"] + res_df["old_index_2"]
    # res_df.sort_values(by = "sort", inplace=True, ignore_index=True)
    # def is_same(row):
    #     s1 = row["signal_1"].get_result()
    #     s2 = row["signal_2"].get_result()
    #     if s1.size != s2.size:
    #         return -1
    #     diff = s1-s2
    #     nb = (np.abs(diff)>0.001).sum()
    #     if nb == 0:
    #         logger.warning("The following arrays are equal \n{}[{}]\n{}[{}]".format(row["file_path_1"], row["file_keys_1"], row["file_path_2"], row["file_keys_2"]))
    #     return nb
    #     #     logger.warning("The following arrays are equal \n{}[{}]\n{}[{}]".format(row["file_path_1"], row["file_keys_1"], row["file_path_2"], row["file_keys_2"]))
    #     #     return True
    #     # return False
    # print(res_df.columns)
    # tqdm.pandas(desc="Checking identical signals")
    # res_df["same"] = res_df.progress_apply(is_same, axis=1)
    # res_df = res_df.loc[res_df["same"]!=-1, :]
    # df_loader.save("duplicates_human_signals.tsv", res_df[["same", "file_path_1", "file_keys_1", "file_path_2", "file_keys_2"]].copy())
    # # duplicates = df_loader.load("duplicates signals.tsv")
    # # duplicates = duplicates.loc[duplicates["same"]==0]

    # # s = set(duplicates["file_path_1"].to_list() + duplicates["file_path_2"].to_list())
    # # s = set(check_df["file_path"].to_list())
    # # # id_dict = {k:i for i,k in enumerate(set(duplicates["file_path_1"].to_list() + duplicates["file_path_2"].to_list()))}
    # # # duplicates["id_1"] = duplicates.apply(lambda row: id_dict[row["file_path_1"]], axis=1)
    # # # duplicates["id_2"] = duplicates.apply(lambda row: id_dict[row["file_path_2"]], axis=1)
    # # # print(duplicates)
    # # from disjoint_union import DisjointUnion
    # # uf = DisjointUnion([])
    # # for e in s:
    # #     uf |= [e]
    # # print(len(uf))
    # # duplicates.apply(lambda row: uf.union(row["file_path_1"], row["file_path_2"]), axis=1)
    # # print([sorted(list(s))[0] for s in uf])

    # raise BaseException("stop")

    win.setup_ready.connect(lambda d: json_loader.save(
        pathlib.Path("setup_params.json"), d))

    win.showMaximized()
    sys.exit(app.exec())
