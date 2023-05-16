from toolbox import Manager, json_loader, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QCoreApplication

beautifullogger.setup(logmode="w")
logger=logging.getLogger(__name__)
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


from analysisGUI import Window

from analysisGUI import InputDataDF
from analysisGUI import CleanDataDF
from analysisGUI import LFPDataDF
from analysisGUI import BUADataDF

from analysisGUI import SpikeContinuousDataDF
from analysisGUI import pwelchDataDF
from analysisGUI import coherenceDataDF
from analysisGUI import correlationDataDF

def run_gui():
    app = QApplication(sys.argv)
    win = Window() 
    input_df = InputDataDF(dataframe_manager, computation_m, step_signals)
    clean_df = CleanDataDF(computation_m, step_signals, input_df)
    lfp_df = LFPDataDF(computation_m, step_signals, clean_df)
    bua_df = BUADataDF(computation_m, step_signals, clean_df)
    spike_continuous_df = SpikeContinuousDataDF(computation_m, step_signals, input_df)
    pwelch_df = pwelchDataDF(computation_m, step_signals, lfp_df, bua_df)
    coherence_df = coherenceDataDF(computation_m, step_signals, lfp_df, bua_df, spike_continuous_df)
    correlation_df = correlationDataDF(computation_m, step_signals, spike_continuous_df)

    

    win.add_df(input_df)
    win.add_df(clean_df)
    win.add_df(lfp_df)
    win.add_df(bua_df)
    win.add_df(spike_continuous_df)
    win.add_df(pwelch_df)
    win.add_df(coherence_df)
    win.add_df(correlation_df)
    

   
    if pathlib.Path("setup_params.json").exists():
        default_params = win.get_setup_params()
        win.set_setup_params(json_loader.load(pathlib.Path("setup_params.json")))
        last_params = win.get_setup_params()
        if last_params.keys() != default_params.keys():
            logger.warning("Strange")
        else:
            res = pd.DataFrame(zip(last_params.keys(), default_params.values(), last_params.values()), columns=["key", "default", "last"])
            res["same"] = res["default"] == res["last"]
            logger.info("Params:\n{}".format(res.to_string()))
        win.on_computation_tab_clicked()
        win.tabWidget.setCurrentWidget(win.tabWidget.findChild(QWidget, "computation_tab"))
    else:
    	win.tabWidget.setCurrentWidget(win.tabWidget.findChild(QWidget, "setup_tab"))
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

    win.setup_ready.connect(lambda d: json_loader.save(pathlib.Path("setup_params.json"), d))

    
    win.showMaximized()
    sys.exit(app.exec())

