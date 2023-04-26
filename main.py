from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm


beautifullogger.setup(logmode="w")
logger=logging.getLogger(__name__)
logging.getLogger("toolbox.ressource_manager").setLevel(logging.WARNING)
logging.getLogger("toolbox.signal_analysis_toolbox").setLevel(logging.WARNING)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info("Keyboard interupt")
        return
    else:
       sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception

computation_m = Manager("./cache/computation")
dataframe_manager = Manager("./cache/dataframes")



from gui import Window
from PyQt5.QtWidgets import QApplication
from input_df import InputDataDF

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window() 
    win.add_df(InputDataDF(dataframe_manager, computation_m))
    win.show()
    sys.exit(app.exec())

