
from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy
import toolbox
import logging

logger = logging.getLogger(__name__)

class Clean(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("signals.cleaned", 
            {
                "clean.deviation_factor": "5",
                "clean.min_length": "0.003",
                "clean.join_width": "3",
                "clean.shoulder_width": "1",
                "clean.recursive": "True",
                "clean.replace_type": "affine",
            }
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        self.tqdm.pandas(desc="Computing cleaned signals")
        df = db[db["input_signal_type"].isin(["raw", "mua"])].copy()

        for key,val in self.metadata.items():
            if "clean." in key:
                df[str(key[len("clean."):])] = val

        df.insert(0, "clean_rm_bounds", df.progress_apply(lambda row: 
            self.computation_m.declare_computable_ressource(
                lambda **kwargs: pd.DataFrame(toolbox.compute_artefact_bounds(**kwargs),columns=["start", "end"]),
                dict(sig=row["input_signal"], fs = row["input_signal_fs"], 
                     deviation_factor=float(self.metadata["clean.deviation_factor"]),
                     min_length=float(self.metadata["clean.min_length"]),
                     join_width=float(self.metadata["clean.join_width"]),
                     recursive=bool(self.metadata["clean.recursive"]),
                     shoulder_width=float(self.metadata["clean.shoulder_width"])
                ), toolbox.df_loader, "clean_bounds", True
            ),
            axis=1)
        )


        df.insert(1, "cleaned_signal", df.progress_apply(lambda row: 
            self.computation_m.declare_computable_ressource(generate_clean,
                dict(signal=row["input_signal"], clean_bounds = row["clean_rm_bounds"], replace_type=self.metadata["clean.replace_type"])
                , toolbox.np_loader, "clean_signal", False
            ),axis=1)
        )

        
        return df

def generate_clean(signal, clean_bounds, replace_type):
    filtered= signal.copy().astype(float)
    for _,artefact in clean_bounds.iterrows():
        s = artefact["start"]
        e = artefact["end"]
        filtered[s:e] = np.nan
    if replace_type == "affine":
        return toolbox.affine_nan_replace(filtered)
    elif replace_type == "nan":
        return filtered
    else:
        raise BaseException("Invalid replace type")