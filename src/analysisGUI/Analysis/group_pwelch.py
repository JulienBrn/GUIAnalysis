from analysisGUI.gui import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging
import functools
from analysisGUI.gui import mk_result_tab, export_fig

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
    
    # def view(self, row, ax, fig):
    #     x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
    #     err=0
    #     handles=[]
    #     if len(row["pwelch"]) >3:
    #         for p in row["pwelch"]:
    #             try:
    #                 ax.plot(x, toolbox.get(p), color="blue")[0]
    #             except:
    #                 err+=1
    #         handles.append(ax.plot([], color="blue", label="plots")[0])
    #     else:
    #         for p in row["pwelch"]:
    #             try:
    #                 handles.append(ax.plot(x, toolbox.get(p), color="blue", label="plot"+str())[0])
    #             except:
    #                 err+=1
    #     handles.append(ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")[0])
    #     handles.append(ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")[0])
        

    #     ax.set_xlabel("Frequency (Hz)")
    #     ax.set_ylabel("Amplitude (?)")
    #     ax.set_xlim(3, 60)
    #     ax.set_title("PWELCH\n"+str(row[["Species", "Structure", "signal_resampled_type", "Condition", "nplots"]].to_dict()) + ", nb_error_plots: "+str(err))
    #     ax.legend(handles=handles, fancybox=True, shadow=True)

    def view_bis(self, row, rtab):
        x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])

        def small_key(k : str):
            if "Structure" in k:
                return k.replace("Structure", "Struct")
            if "signal_resampled_type" in k:
                return k.replace("signal_resampled_type", "type")
            return k
        
        for rtype in ["Amplitude", "Max"]:
            title = f"PWelch {rtype}\n"+str({small_key(k):toolbox.get(v) for k,v in row[["Species",  "Condition", "Structure", "signal_resampled_type", "nplots", "nb_non_err"]].to_dict().items()})
            result_tab,mpls = mk_result_tab(1,1)
            rtab.addTab(result_tab, title.replace("\n", ": "))
            fig = mpls[0,0].canvas.fig
            ax = mpls[0,0].canvas.ax
            handles=[]
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude (?)")
            xmin=3
            xmax=35
            ymin=np.inf
            ymax=-np.inf
            ax.set_xlim(xmin, xmax)
            ax.set_title(title)
            if rtype=="Amplitude":
                for p in [toolbox.get(r) for r in row["pwelch"] if not isinstance(toolbox.get(r), toolbox.Error)]:
                    ymax = max(ymax, np.amax(np.abs(p)[(x >= xmin) & (x <= xmax)]))
                    ymin = min(ymin, np.amin(np.abs(p)[(x >= xmin) & (x <= xmax)]))
                    ax.plot(x, p, color="blue")[0]
                handles.append(ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")[0])
                handles.append(ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")[0])
                ax.set_ylim(ymin, ymax)
                ax.set_ylim(0.7*np.min(toolbox.get(row["Mean"])[(x >= xmin) & (x <= xmax)]), 1.3*np.max(toolbox.get(row["Mean"])[(x >= xmin) & (x <= xmax)]))
            elif rtype=="Max":
                for f, m in [(float(toolbox.get(f)), float(toolbox.get(m))) for f, m in zip(row["best_f"], row["best_amp"]) if not isinstance(toolbox.get(f), toolbox.Error) and not isinstance(toolbox.get(m), toolbox.Error)]:
                    ax.scatter(f, m, color="blue")

            handles.append(ax.plot([], color="blue", label="items")[0])
            ax.legend(handles=handles, fancybox=True, shadow=True)
            result_tab.export = functools.partial(export_fig, fig=fig, title=title, canvas=mpls[0,0].canvas)

        # title = "PWELCH\n"+str(row[["Species", "Structure", "signal_resampled_type", "Condition", "nplots"]].to_dict()) + ", nb_non_error_plots: "+str(toolbox.get(row["nb_non_err"]))
        # result_tab,mpls = mk_result_tab(1,1)
        # rtab.addTab(result_tab, title.replace("\n", ": "))
        # fig = mpls[0,0].canvas.fig
        # ax = mpls[0,0].canvas.ax
        # x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        # err=0
        # handles=[]
        # if len(row["pwelch"]) >3:
        #     for p in row["pwelch"]:
        #         try:
        #             ax.plot(x, toolbox.get(p), color="blue")[0]
        #         except:
        #             err+=1
        #     handles.append(ax.plot([], color="blue", label="plots")[0])
        # else:
        #     for p in row["pwelch"]:
        #         try:
        #             handles.append(ax.plot(x, toolbox.get(p), color="blue", label="plot"+str())[0])
        #         except:
        #             err+=1
        # handles.append(ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")[0])
        # handles.append(ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")[0])
        

        # ax.set_xlabel("Frequency (Hz)")
        # ax.set_ylabel("Amplitude (?)")
        # ax.set_xlim(3, 60)
        # ax.set_title(title)
        # ax.legend(handles=handles, fancybox=True, shadow=True)


        # title = "PWELCH best_f\n"+str(row[["Species", "Structure", "signal_resampled_type", "Condition", "nplots"]].to_dict()) + ", nb_non_error_plots: "+str(toolbox.get(row["nb_non_err"]))
        # from analysisGUI.gui import mk_result_tab
        # result_tab,mpls = mk_result_tab(1,1)
        # rtab.addTab(result_tab, title.replace("\n", ": "))
        # fig = mpls[0,0].canvas.fig
        # ax = mpls[0,0].canvas.ax
        # x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        # err=0
        # handles=[]
        # if len(row["pwelch"]) >3:
        #     for f, m in zip(row["best_f"], row["best_amp"]):
        #         try:
        #             ax.scatter(toolbox.get(f), toolbox.get(m), color="blue")[0]
        #         except:
        #             err+=1
        #     handles.append(ax.plot([], color="blue", label="best")[0])
        # # else:
        # #     for p in row["pwelch"]:
        # #         try:
        # #             handles.append(ax.plot(x, toolbox.get(p), color="blue", label="plot"+str())[0])
        # #         except:
        # #             err+=1
        # # handles.append(ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")[0])
        # # handles.append(ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")[0])
        

        # ax.set_xlabel("Frequency (Hz)")
        # ax.set_ylabel("Amplitude (?)")
        # ax.set_xlim(5, 40)
        # ax.set_title(title)
        # ax.legend(handles=handles, fancybox=True, shadow=True)

    # def view_all(self, ax, fig):
    #     df = self.get_df()
    #     row=df.iloc[0, :]
    #     x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
    #     # ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")
    #     for j in range(len(df.index)):
    #         row = df.iloc[j, :]
    #         ax.plot(x, toolbox.get(row["Mean"]), label=str(row[["Species", "Structure", "signal_resampled_type", "Condition"]].to_dict()))
        
    #     # ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")
        

    #     ax.set_xlabel("Frequency (Hz)")
    #     ax.set_ylabel("Amplitude (?)")
    #     ax.set_xlim(3, 60)
    #     ax.set_title("Pwelch averages")
    #     fig.legend(loc='lower center', ncols=3, fontsize=8,fancybox=True, shadow=True)
    #     fig.subplots_adjust(bottom=0.5)

    def view_all_bis(self, rtab):
        df = self.get_df()
        df=df[df["nb_non_err"].apply(toolbox.get)>0].reset_index(drop=True)
        df=df[(~df["Structure"].str.lower().str.contains("ecog"))].reset_index(drop=True)
        df = add_colors(df)
        
        row=df.iloc[0, :]
        x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        groups = [g.reset_index(drop=True) for _, g in df[df["nb_non_err"].apply(toolbox.get)>0].groupby(by=["signal_resampled_type", "Healthy"])]
        
        for i, d in enumerate(groups):
            title = "PWelch avg {}, healthy={}".format(d["signal_resampled_type"].iat[0], d["Healthy"].iat[0])
            result_tab,mpls = mk_result_tab(1,1)
            rtab.addTab(result_tab, f"{i+1}/{len(groups)} {title}")
            fig = mpls[0,0].canvas.fig
            ax = mpls[0,0].canvas.ax
            xmin=3
            xmax = 35
            ymax= 0
            ymin = 1
            for j in range(len(d.index)):
                row_dict = d.iloc[j, :].to_dict()
                ymax = max(ymax, np.amax(np.abs(toolbox.get(row_dict["Mean"]))[(x >= xmin) & (x <= xmax)]))
                ymin = min(ymin, np.amin(np.abs(toolbox.get(row_dict["Mean"]))[(x >= xmin) & (x <= xmax)]))
                row_dict["nb_non_err"] = int(toolbox.get(row_dict["nb_non_err"]))
                ax.plot(x, toolbox.get(row_dict["Mean"]), label=str({k:toolbox.get(v) for k,v in row_dict.items() if k in ["Species", "Structure", "nplots", "nb_non_err"]}), color=row_dict["color"])

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude (?)")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin*0.9, ymax*1.1)
            ax.set_title(title)
            fig.legend(loc='lower center', ncols=3, fontsize=8,fancybox=True, shadow=True)
            fig.subplots_adjust(bottom=0.3)
            result_tab.export = functools.partial(export_fig, fig=fig, title=title, canvas=mpls[0,0].canvas)




        # self.figs = {}
        # from analysisGUI.gui import mk_result_tab
        # df = self.get_df()
        # row=df.iloc[0, :]
        # x = np.arange(0, row["pwelch_max_f"]+1/row["pwelch_fs"], 1/row["pwelch_fs"])
        # grouped_df = df.groupby(by=["signal_resampled_type", "Healthy"]).agg(lambda x:tuple(x)).reset_index()
        

        # for i in range(len(grouped_df.index)):
        #     result_tab,mpls = mk_result_tab(1,1)
        #     rtab.addTab(result_tab, "pwelch {}, healthy={}".format(grouped_df["signal_resampled_type"].iat[i], grouped_df["Healthy"].iat[i]))
        #     fig = mpls[0,0].canvas.fig
        #     ax = mpls[0,0].canvas.ax
        #     xmin=3
        #     xmax = 35
        #     ymax= 0
        #     ymin = 1
        #     for j in range(len(grouped_df["Mean"].iat[i])):
        #         row_dict = {}
        #         for col in grouped_df.columns:
        #             l = grouped_df[col].iat[i]
        #             row_dict.update({col: l if not isinstance(l, tuple) else l[j]})
        #         if "ecog" not in row_dict["Structure"].lower():
        #             ymax = max(ymax, np.amax(toolbox.get(row_dict["Mean"])[(x >= xmin) & (x <= xmax)]))
        #             ymin = min(ymin, np.amin(toolbox.get(row_dict["Mean"])[(x >= xmin) & (x <= xmax)]))
        #             ax.plot(x, toolbox.get(row_dict["Mean"]), label=str({k:v for k,v in row_dict.items() if k in ["Species", "Structure", "signal_resampled_type", "Condition"]}), **draw_params(row_dict))
        #     ax.set_xlabel("Frequency (Hz)")
        #     ax.set_ylabel("Amplitude (?)")
        #     ax.set_xlim(xmin, xmax)
        #     ax.set_ylim(ymin*0.9, ymax*1.1)
        #     ax.set_title("Pwelch averages {}, healthy={}".format(grouped_df["signal_resampled_type"].iat[i], grouped_df["Healthy"].iat[i]))
        #     fig.legend(loc='lower center', ncols=3, fontsize=8,fancybox=True, shadow=True)
        #     fig.subplots_adjust(bottom=0.3)
        #     result_tab.export = functools.partial(export_fig, fig=fig, title=title, canvas=mpls[0,0].canvas)

    # def export_figs(self, folder):
    #     for t, fig in self.figs.items():
    #         fig.savefig(str(pathlib.Path(folder) / (t +".png")))

        
        
        # ax.plot(x, toolbox.get(row["Mean"]), color="red", label="avg")
        # for j in range(len(df.index)):
        #     row = df.iloc[j, :]
            
        
        # ax.plot(x, toolbox.get(row["Median"]), color="yellow", label="median")
        

import matplotlib.pyplot as plt
import matplotlib as mpl

def add_colors(df: pd.DataFrame):
    group_df = df.drop_duplicates(subset=["Species", "Structure"], keep="first")[["Species", "Structure"]]
    group_df["colormap"] = group_df["Species"].apply(lambda s: "Blues" if s.lower()=="rat" else "Reds" if s.lower()=="monkey" else "Greens")
    group_df["num_colors_in_species"] = group_df.groupby("Species")["colormap"].transform("count")
    group_df["one"] = 1
    group_df["color_id_in_species"] = group_df.groupby("Species")["one"].cumsum() -1
    group_df.drop(columns="one", inplace=True)
    group_df["color"] = group_df.apply(lambda r: 
        mpl.colors.rgb2hex(mpl.colors.to_rgba(
            plt.cm.get_cmap(r["colormap"])((0.5 + r["color_id_in_species"])/r["num_colors_in_species"])
            , 1), keep_alpha=True), axis=1)
    
    res=df.merge(group_df, how="left", on=["Species", "Structure"])
    return res


# def draw_params(row_dict):
#     colors=plt.cm.get_cmap("turbo")

#     if row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "gpe":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.1), 1), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "stn":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.15), 0.7), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "str":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.2), 1), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "rat" and row_dict["Structure"].lower() == "snr":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.25), 0.7), keep_alpha=True)}
    
#     if row_dict["Species"].lower() == "monkey" and row_dict["Structure"].lower() == "gpe":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.4+0.1), 1), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "monkey" and row_dict["Structure"].lower() == "stn":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.4+0.15), 0.7), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "monkey" and row_dict["Structure"].lower() == "msn":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.4+0.2), 1), keep_alpha=True)}
    
#     if row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "gpe":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.1), 1), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "stn_dlor":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.15), 0.7), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "stn_vmnr":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.2), 1), keep_alpha=True)}
#     elif row_dict["Species"].lower() == "human" and row_dict["Structure"].lower() == "msn":
#         return {"color": mpl.colors.rgb2hex(mpl.colors.to_rgba(colors(0.7+0.25), 0.7), keep_alpha=True)}
    

#     return {}


        


