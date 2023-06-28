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
        self.tqdm.pandas(desc="Computing group pwelch")
        df = db.groupby(["Species", "Structure", "signal_resampled_type", "Condition"]).agg(lambda x:tuple(x)).reset_index()
        def is_identical(x):
            try:
                return isinstance(x, tuple) and len(set(x)) == 1
            except: 
                False
        
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x if not is_identical(x) else x[0])
        
        df.insert(0, "nplots", df["pwelch"].apply(len))
        df.insert(0, "Mean", df["pwelch"].progress_apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.mean(np.vstack(kwargs.values()),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_pwelch_avg", True, error_method="filter")))
        df.insert(0, "Median", df["pwelch"].progress_apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.median(np.vstack(kwargs.values()),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_pwelch_median", True, error_method="filter")))
        df.insert(0, "nb_non_err", df["pwelch"].progress_apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: len(kwargs), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_pwelch_nb_non_err", True, error_method="filter")))
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

    def view_bis(self, row, rtab):
        title = "PWELCH\n"+str(row[["Species", "Structure", "signal_resampled_type", "Condition", "nplots"]].to_dict()) + ", nb_non_error_plots: "+str(toolbox.get(row["nb_non_err"]))
        from analysisGUI.gui import mk_result_tab
        result_tab,mpls = mk_result_tab(1,1)
        rtab.addTab(result_tab, title.replace("\n", ": "))
        fig = mpls[0,0].canvas.fig
        ax = mpls[0,0].canvas.ax
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
        ax.set_title(title)
        ax.legend(handles=handles, fancybox=True, shadow=True)


        title = "PWELCH best_f\n"+str(row[["Species", "Structure", "signal_resampled_type", "Condition", "nplots"]].to_dict()) + ", nb_non_error_plots: "+str(toolbox.get(row["nb_non_err"]))
        from analysisGUI.gui import mk_result_tab
        result_tab,mpls = mk_result_tab(1,1)
        rtab.addTab(result_tab, title.replace("\n", ": "))
        fig = mpls[0,0].canvas.fig
        ax = mpls[0,0].canvas.ax
        x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        err=0
        handles=[]
        if len(row["pwelch"]) >3:
            for f, m in zip(row["best_f"], row["best_amp"]):
                try:
                    ax.scatter(toolbox.get(f), toolbox.get(m), color="blue")[0]
                except:
                    err+=1
            handles.append(ax.plot([], color="blue", label="best")[0])
        # else:
        #     for p in row["pwelch"]:
        #         try:
        #             handles.append(ax.plot(x, toolbox.get(p), color="blue", label="plot"+str())[0])
        #         except:
        #             err+=1
        # handles.append(ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")[0])
        # handles.append(ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")[0])
        

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (?)")
        ax.set_xlim(5, 40)
        ax.set_title(title)
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
        fig.legend(loc='lower center', ncols=3, fontsize=8,fancybox=True, shadow=True)
        fig.subplots_adjust(bottom=0.5)

    def view_all_bis(self, rtab):
        self.figs = {}
        from analysisGUI.gui import mk_result_tab
        df = self.get_df()
        row=df.iloc[0, :]
        x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        grouped_df = df.groupby(by=["signal_resampled_type", "Healthy"]).agg(lambda x:tuple(x)).reset_index()
        

        for i in range(len(grouped_df.index)):
            result_tab,mpls = mk_result_tab(1,1)
            rtab.addTab(result_tab, "pwelch {}, healthy={}".format(grouped_df["signal_resampled_type"].iat[i], grouped_df["Healthy"].iat[i]))
            fig = mpls[0,0].canvas.fig
            ax = mpls[0,0].canvas.ax
            xmin=3
            xmax = 35
            ymax= 0
            ymin = 1
            for j in range(len(grouped_df["Mean"].iat[i])):
                row_dict = {}
                for col in grouped_df.columns:
                    l = grouped_df[col].iat[i]
                    row_dict.update({col: l if not isinstance(l, tuple) else l[j]})
                if "ecog" not in row_dict["Structure"].lower():
                    ymax = max(ymax, np.amax(toolbox.get(row_dict["Mean"])[(x >= xmin) & (x <= xmax)]))
                    ymin = min(ymin, np.amin(toolbox.get(row_dict["Mean"])[(x >= xmin) & (x <= xmax)]))
                    ax.plot(x, toolbox.get(row_dict["Mean"]), label=str({k:v for k,v in row_dict.items() if k in ["Species", "Structure", "signal_resampled_type", "Condition"]}), **draw_params(row_dict))
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude (?)")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin*0.9, ymax*1.1)
            ax.set_title("Pwelch averages {}, healthy={}".format(grouped_df["signal_resampled_type"].iat[i], grouped_df["Healthy"].iat[i]))
            fig.legend(loc='lower center', ncols=3, fontsize=8,fancybox=True, shadow=True)
            fig.subplots_adjust(bottom=0.3)
            self.figs.update({"Pwelch averages {}, healthy={}".format(grouped_df["signal_resampled_type"].iat[i], grouped_df["Healthy"].iat[i]):fig})

    def export_figs(self, folder):
        for t, fig in self.figs.items():
            fig.savefig(str(pathlib.Path(folder) / (t +".png")))

        
        
        # ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")
        # for j in range(len(df.index)):
        #     row = df.iloc[j, :]
            
        
        # ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")
        

import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_params(row_dict):
    colors=plt.cm.get_cmap("turbo")

    if row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "gpe":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.1), 1), keep_alpha=True)}
    elif row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "stn":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.15), 0.7), keep_alpha=True)}
    elif row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "str":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.2), 1), keep_alpha=True)}
    elif row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "snr":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.25), 0.7), keep_alpha=True)}
    
    if row_dict["Species"].lower() == "monkey" and row_dict["Structure"].lower() == "gpe":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.4+0.1), 1), keep_alpha=True)}
    elif row_dict["Species"].lower() == "monkey" and row_dict["Structure"].lower() == "stn":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.4+0.15), 0.7), keep_alpha=True)}
    elif row_dict["Species"].lower() == "monkey" and row_dict["Structure"].lower() == "msn":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.4+0.2), 1), keep_alpha=True)}
    
    if row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "gpe":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.1), 1), keep_alpha=True)}
    elif row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "stn_dlor":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.15), 0.7), keep_alpha=True)}
    elif row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "stn_vmnr":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.2), 1), keep_alpha=True)}
    elif row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "msn":
        return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.25), 0.7), keep_alpha=True)}
    

    return {}


        


