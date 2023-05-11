from toolbox import Manager, json_loader, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
from PyQt5.QtWidgets import QApplication
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


from gui import Window

from input_df import InputDataDF
from clean_df import CleanDataDF
from lfp_df import LFPDataDF
from bua_df import BUADataDF

from spike_continuous_df import SpikeContinuousDataDF
from pwelch_df import pwelchDataDF
from coherence_df import coherenceDataDF
from correlation_df import correlationDataDF

if __name__ == "__main__":
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
        win.set_setup_params(json_loader.load(pathlib.Path("full_params.json")))
        last_params = win.get_setup_params()
        if last_params.keys() != default_params.keys():
            logger.warning("Strange")
        else:
            res = pd.DataFrame(zip(last_params.keys(), default_params.values(), last_params.values()), columns=["key", "default", "last"])
            res["same"] = res["default"] == res["last"]
            logger.info("Params:\n{}".format(res.to_string()))
        win.on_computation_tab_clicked()
    # coherence_df.compute()
    # pwelch_df.compute()
    # correlation_df.compute()

    win.setup_ready.connect(lambda d: json_loader.save(pathlib.Path("setup_params.json"), d))

    
    win.showMaximized()
    sys.exit(app.exec())

