from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging

logger = logging.getLogger(__name__)

class CoherenceGroups(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("analysis.coherence.groups", 
            {}
            , computation_m, {"db":signals})
        self.computation_m = computation_m
    
    def compute_df(self, db: pd.DataFrame):
        df = db.copy()
        df = df.groupby(["Species",  "Condition", "Structure_1", "signal_resampled_type_1", "Structure_2","signal_resampled_type_2", "Same_Electrode"]).agg(lambda x:tuple(x)).reset_index()
        def is_identical(x):
            try:
                return isinstance(x, tuple) and len(set(x)) == 1
            except: 
                False
        
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x if not is_identical(x) else x[0])
        
        df["coherence"]=df["coherence"].apply(lambda x: x if  isinstance(x, tuple) else tuple([x]))
        df.insert(0, "nplots", df["coherence"].apply(lambda x: len(x) if  isinstance(x, tuple) else 1))
        df.insert(0, "Mean", df["coherence"].apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.mean(np.vstack(kwargs.values()),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_coherence_avg", True, error_method="filter")))
        df.insert(0, "Median", df["coherence"].apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.median(np.vstack(kwargs.values()),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_coherence_median", True, error_method="filter")))
        
        return df
    
    def view(self, row, ax, fig):
        x = np.arange(0, row["coherence_max_f"]+1/row["coherence_fs"], 1/row["coherence_fs"])
        err=0
        handles=[]
        if len(row["coherence"]) >5:
            for i, p in enumerate(row["coherence"]):
                try:
                    r = np.abs(toolbox.get(p))
                    if np.mean(r) > 0.7:
                        nrow = row.apply(lambda x: x if not isinstance(x, tuple) or i>= len(x) else x[i])
                        logger.warning("High coherence (avg = {}) for {}.".format(np.mean(r), nrow.to_dict()))
                    ax.plot(x, np.abs(toolbox.get(p)), color="blue")[0]
                except:
                    err+=1
            handles.append(ax.plot([], color="blue", label="plots")[0])
        else:
            for i, p in enumerate(row["coherence"]):
                try:
                    r = np.abs(toolbox.get(p))
                    if np.mean(r) > 0.7:
                        nrow = row.apply(lambda x: x if not isinstance(x, tuple) or i>= len(x) else x[i])
                        logger.warning("High coherence (avg = {}) for {}.".format(np.mean(r), nrow.to_dict()))
                    handles.append(ax.plot(x, np.abs(toolbox.get(p)), label="plot"+str(i))[0])
                except:
                    err+=1
        handles.append(ax.plot(x, np.abs(toolbox.get(row["Mean"])), color="red", label="avg")[0])
        handles.append(ax.plot(x, np.abs(toolbox.get(row["Median"])), color="yellow", label="median")[0])
        

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (?)")
        ax.set_xlim(3, 60)
        ax.set_title("Coherence Amplitude\n"+str(row[["Species",  "Condition", "Structure_1", "signal_resampled_type_1", "Structure_2", "signal_resampled_type_2", "Same_Electrode", "nplots"]].to_dict()) + ", nb_error_plots: "+str(err))
        ax.legend(handles=handles, fancybox=True, shadow=True)

    def view_all(self, ax, fig):
        df = self.get_df()
        row=df.iloc[0, :]
        x = np.arange(0, row["coherence_max_f"]+1/row["coherence_fs"], 1/row["coherence_fs"])
        # ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")
        for j in range(len(df.index)):
            row = df.iloc[j, :]
            ax.plot(x, np.abs(toolbox.get(row["Mean"])), label=str(row[["Species",  "Condition", "Structure_1", "signal_resampled_type_1", "Structure_2", "signal_resampled_type_2", "Same_Electrode"]].to_dict()))
        
        # ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")
        

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (?)")
        ax.set_xlim(3, 60)
        ax.set_title("Coherence amplitude averages")
        fig.legend(fancybox=True, shadow=True)