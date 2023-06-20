from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging

logger = logging.getLogger(__name__)

class PWelchGroups(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("analysis.pwelch.groups", 
            {}
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        df = db.groupby(["Species", "Structure", "signal_resampled_type", "Condition"]).agg(lambda x:tuple(x)).reset_index()
        def is_identical(x):
            try:
                return isinstance(x, tuple) and len(set(x)) == 1
            except: 
                False
        
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x if not is_identical(x) else x[0])
        
        df.insert(0, "nplots", df["pwelch"].apply(len))
        df.insert(0, "Mean", df["pwelch"].apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.mean(np.vstack(kwargs.values()),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_pwelch_avg", True, error_method="filter")))
        df.insert(0, "Median", df["pwelch"].apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.median(np.vstack(kwargs.values()),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_pwelch_median", True, error_method="filter")))
        
        return df
    
    def view(self, row, ax, fig):
        x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        err=0
        handles=[]
        if len(row["pwelch"]) >3:
            for p in row["pwelch"]:
                try:
                    ax.plot(x, toolbox.get(p), color="blue")[0]
                except:
                    err+=1
            handles.append(ax.plot([], color="blue", label="plots")[0])
        else:
            for p in row["pwelch"]:
                try:
                    handles.append(ax.plot(x, toolbox.get(p), color="blue", label="plot"+str())[0])
                except:
                    err+=1
        handles.append(ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")[0])
        handles.append(ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")[0])
        

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (?)")
        ax.set_xlim(3, 60)
        ax.set_title("PWELCH\n"+str(row[["Species", "Structure", "signal_resampled_type", "Condition", "nplots"]].to_dict()) + ", nb_error_plots: "+str(err))
        ax.legend(handles=handles, fancybox=True, shadow=True)

    def view_all(self, ax, fig):
        df = self.get_df()
        row=df.iloc[0, :]
        x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        # ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")
        for j in range(len(df.index)):
            row = df.iloc[j, :]
            ax.plot(x, toolbox.get(row["Mean"]), label=str(row[["Species", "Structure", "signal_resampled_type", "Condition"]].to_dict()))
        
        # ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")
        

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (?)")
        ax.set_xlim(3, 60)
        ax.set_title("Pwelch averages")
        fig.legend(fancybox=True, shadow=True)