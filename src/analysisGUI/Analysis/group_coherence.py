from toolbox import GUIDataFrame
import pathlib 
import pandas as pd, numpy as np, scipy, math
import toolbox
import logging
import functools
from toolbox import mk_result_tab, export_fig

logger = logging.getLogger(__name__)

class CoherenceGroups(GUIDataFrame):
    def __init__(self, signals, computation_m: toolbox.Manager):
        super().__init__("analysis.coherence.groups", 
            {
                "coherence.groups.best_f.min":"8",
                "coherence.groups.best_f.max":"35",
            }
            , computation_m, {"db":signals})
        self.computation_m = computation_m
        self.key_columns = ["Species", "Healthy", "Structure_1", "signal_resampled_type_1", "Structure_2","signal_resampled_type_2", "Same_Electrode"]
    
    def compute_df(self, db: pd.DataFrame, **params):
        self.tqdm.pandas(desc="Computing group coherence_df") 
        df = db.copy()
        df = df.groupby(["Species",  "Condition", "Structure_1", "signal_resampled_type_1", "Structure_2","signal_resampled_type_2", "Same_Electrode"]).agg(lambda x:tuple(x)).reset_index()
        df["Healthy"] = df["Condition"].apply(lambda x: "healthy" in x or "CTL" in x)
        df["Structure_1_old"] = df["Structure_1"]
        df["Structure_2_old"] = df["Structure_2"]
        df["Structure_1"] = df["Structure_1_old"].apply(lambda x: "STR" if ("MSN" in x.upper() or "STR" in x.upper()) else "STN" if ("STN"== x.upper() or "STN_DLOR"== x.upper()) else "GPe" if "GPE" in x.upper() else x)
        df["Structure_2"] = df["Structure_2_old"].apply(lambda x: "STR" if ("MSN" in x.upper() or "STR" in x.upper()) else "STN" if ("STN"== x.upper() or "STN_DLOR"== x.upper()) else "GPe" if "GPE" in x.upper() else x)
        mfilter = (df["Same_Electrode"] == False) & (df["Structure_1"] ==  df["Structure_2"]) & (df["Structure_1"].isin(["GPe", "STN", "STR"])) & (df["signal_resampled_type_1"].isin(["bua", "spike_bins"])) & (df["signal_resampled_type_2"].isin(["bua", "spike_bins"])) &  (df["signal_resampled_type_1"] != df["signal_resampled_type_2"])
        df = df[mfilter].copy()
        def is_identical(x):
            try:
                return isinstance(x, tuple) and len(set(x)) == 1
            except: 
                False
        for key,val in params.items():
            if "coherence_" in key:
                df[str(key[len("coherence_"):])] = val

        for col in df.columns:
            df[col] = df[col].apply(lambda x: x if not is_identical(x) else x[0])
        
        df["coherence"]=df["coherence"].progress_apply(lambda x: x if  isinstance(x, tuple) else tuple([x]))
        df["best_f"]=df["best_f"].progress_apply(lambda x: x if  isinstance(x, tuple) else tuple([x]))
        df["best_val"]=df["best_val"].progress_apply(lambda x: x if  isinstance(x, tuple) else tuple([x]))
        df.insert(0, "nplots", df["coherence"].progress_apply(lambda x: len(x) if  isinstance(x, tuple) else 1))
        df.insert(0, "Mean", df["coherence"].progress_apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.mean(np.vstack(tuple(kwargs.values())),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_coherence_avg", True, error_method="filter")))
        df.insert(0, "Median_amp", df["coherence"].progress_apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: np.median(np.abs(np.vstack(tuple(kwargs.values()))),axis=0), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_coherence_median_amp", True, error_method="filter")))
        df.insert(0, "nb_non_err", df["coherence"].progress_apply(lambda t: self.computation_m.declare_computable_ressource(
            lambda **kwargs: len(kwargs), {"r"+str(i):a for i, a in enumerate(t)}, toolbox.np_loader, "groups_coherence_nb_non_err", True, error_method="filter")))
        df.insert(0, "best_group_f_ind", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
            lambda a, min, max, fs: np.argmax(np.abs(a[int(min*fs):int(max*fs)])) + int(min*fs), {"a": row["Mean"], "min": float(row["groups_best_f_min"]), "max": float(row["groups_best_f_max"]), "fs": float(row["coherence_fs"])}, toolbox.float_loader, "groups_coherence_best_f_ind", True), axis=1))
        df.insert(0, "best_group_f", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
            lambda a, min, max, fs: float(np.argmax(np.abs(a[int(min*fs):int(max*fs)])) + int(min*fs))/fs, {"a": row["Mean"], "min": float(row["groups_best_f_min"]), "max": float(row["groups_best_f_max"]), "fs": float(row["coherence_fs"])}, toolbox.float_loader, "groups_coherence_best_f", True), axis=1))
        
        def compute_band_dist(s, e, fs, nbins, **plots):
            l = [np.abs(np.sum(plot[int(s*fs): int(e*fs)])/(int(e*fs) - int(s*fs))) for plot in plots.values()]
            s = pd.DataFrame({"val": l})
            res = s.quantile([i/nbins for i in range(nbins)]).reset_index()
            return res
        def compute_nb_relevant(s, e, fs, min_val, min_consecutive, **plots):
            if min_consecutive != 2:
                raise NotImplementedError("only nconsecutive=2 implemented")
            nb = 0
            for plot in plots.values():
                b = np.abs(plot) > min_val
                f = b[1:]
                s = b[:-1]
                if 2 in s+f:
                    nb+=1
            return nb

        for sband, eband in [(8, 15), (14, 30)]:
           df.insert(0, f"dist_band_{sband}, {eband}", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
            compute_band_dist, {"s": sband, "e": eband, "fs": float(row["coherence_fs"]), "nbins": 100, **{"r"+str(i):a for i, a in enumerate(row["coherence"])}}, toolbox.df_loader, "coherence_band_dist", True,  error_method="filter"), axis=1))
           df.insert(0, f"nb_relevant_{sband}, {eband}", df.progress_apply(lambda row: self.computation_m.declare_computable_ressource(
            compute_band_dist, {"s": sband, "e": eband, "fs": float(row["coherence_fs"]), "min_val": 0.05, "min_consecutive":2, **{"r"+str(i):a for i, a in enumerate(row["coherence"])}}, toolbox.float_loader, "coherence_band_relevant", True,  error_method="filter"), axis=1))
        return df
    

    def view_bis(self, row, rtab):
        if toolbox.get(row["nb_non_err"])<=0:
            return
        x = np.arange(0, row["coherence_max_f"]+1/row["coherence_fs"], 1/row["coherence_fs"])

        def small_key(k : str):
            if "Structure" in k:
                return k.replace("Structure", "Struct")
            if "signal_resampled_type" in k:
                return k.replace("signal_resampled_type", "type")
            return k
        
        for rtype in ["Amplitude", "Max", "Phase"]:
            title = f"Coherence {rtype}\n"+str({small_key(k):toolbox.get(v) for k,v in row[["Species",  "Condition", "Structure_1", "signal_resampled_type_1", "Structure_2", "signal_resampled_type_2", "Same_Electrode", "nplots", "nb_non_err"]].to_dict().items()})
            if rtype!="Phase":
                result_tab,mpls = mk_result_tab(1,1)
            else:
                result_tab,mpls = mk_result_tab(1,1, projection="polar")
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
                for p in [toolbox.get(r) for r in row["coherence"] if not isinstance(toolbox.get(r), toolbox.Error)]:
                    ymax = max(ymax, np.amax(np.abs(p)[(x >= xmin) & (x <= xmax)]))
                    ymin = min(ymin, np.amin(np.abs(p)[(x >= xmin) & (x <= xmax)]))
                    ax.plot(x, np.abs(p), color="blue")
                try:
                    handles.append(ax.plot(x, np.abs(toolbox.get(row["Mean"])), color="red", label="avg")[0])
                    handles.append(ax.plot(x, np.abs(toolbox.get(row["Median_amp"])), color="yellow", label="median")[0])
                except BaseException as e: logger.error(e)
                ax.set_ylim(ymin, ymax)
                ax.set_ylim(0.7*np.min(np.abs(toolbox.get(row["Mean"])[(x >= xmin) & (x <= xmax)])), 1.3*np.max(np.abs((toolbox.get(row["Mean"])[(x >= xmin) & (x <= xmax)]))))
            elif rtype=="Max":
                for f, m in [(float(toolbox.get(f)), complex(toolbox.get(m))) for f, m in zip(tuple(row["best_f"]), tuple(row["best_val"])) if not isinstance(toolbox.get(f), toolbox.Error) and not isinstance(toolbox.get(m), toolbox.Error)]:
                    ax.scatter(f, np.abs(m), color="blue")
            elif rtype=="Phase":
                ax.set_xlim(0, 2*np.pi)
                f_ind = int(row["best_group_f_ind"].get_result())
                ax.set_title(ax.get_title()+f"\nF={float(row['best_group_f'].get_result())}")
                for p in [toolbox.get(r) for r in row["coherence"] if not isinstance(toolbox.get(r), toolbox.Error)]:
                    ax.vlines(np.angle(p[f_ind], deg=False), 0, np.abs(p[f_ind]), color="blue")
                handles.append(ax.vlines(np.angle(toolbox.get(row["Mean"])[f_ind], deg=False), 0, np.abs(p[f_ind]), color="red", label="mean"))

            handles.append(ax.plot([], color="blue", label="items")[0])
            ax.legend(handles=handles, fancybox=True, shadow=True)
            result_tab.export = functools.partial(export_fig, fig=fig, title=title, canvas=mpls[0,0].canvas)

    
    def view_all_bis(self, rtab):
        self.figs = {}
        df = self.get_df()
        df=df[df["nb_non_err"].apply(toolbox.get)>0].reset_index(drop=True)
        df=df[(~df["Structure_1"].str.lower().str.contains("ecog")) & (~df["Structure_2"].str.lower().str.contains("ecog"))].reset_index(drop=True)
        df = add_colors(df)
        
        row=df.iloc[0, :]
        x = np.arange(0, row["coherence_max_f"]+1/row["coherence_fs"], 1/row["coherence_fs"])
        groups = [g.reset_index(drop=True) for _, g in df[df["nb_non_err"].apply(toolbox.get)>0].groupby(by=["signal_resampled_type_1", "signal_resampled_type_2", "Healthy", "Same_Electrode"])]
        
        for i, d in enumerate(groups):
            title = "Coherence avg {}, {}, healthy={}, same_elec={}".format(d["signal_resampled_type_1"].iat[0], d["signal_resampled_type_2"].iat[0], d["Healthy"].iat[0],  d["Same_Electrode"].iat[0])
            result_tab,mpls = mk_result_tab(1,1)
            rtab.addTab(result_tab, f"{i+1}/{len(groups)} {title}")
            fig = mpls[0,0].canvas.fig
            ax = mpls[0,0].canvas.ax
            xmin=3
            xmax = 35
            ymax= -np.inf
            ymin = np.inf
            for j in range(len(d.index)):
                row_dict = d.iloc[j, :].to_dict()
                if "ecog" not in row_dict["Structure_1"].lower() and "ecog" not in row_dict["Structure_2"].lower():
                    try:
                        ymax = max(ymax, np.amax(np.abs(toolbox.get(row_dict["Mean"]))[(x >= xmin) & (x <= xmax)]))
                        ymin = min(ymin, np.amin(np.abs(toolbox.get(row_dict["Mean"]))[(x >= xmin) & (x <= xmax)]))
                        row_dict["nb_non_err"] = int(toolbox.get(row_dict["nb_non_err"]))
                        ax.plot(x, np.abs(toolbox.get(row_dict["Mean"])), label=str({k:toolbox.get(v) for k,v in row_dict.items() if k in ["Species", "Structure_1", "Structure_2", "nplots", "nb_non_err"]}), color=row_dict["color"])
                    except: pass
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude (?)")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin*0.9, ymax*1.1)
            ax.set_title(title)
            fig.legend(loc='lower center', ncols=3, fontsize=8,fancybox=True, shadow=True)
            fig.subplots_adjust(bottom=0.3)
            result_tab.export = functools.partial(export_fig, fig=fig, title=title, canvas=mpls[0,0].canvas)

    # def export_figs(self, folder):
    #     for t, fig in self.figs.items():
    #         fig.savefig(str(pathlib.Path(folder) / (t +".png")))







    # def view(self, row, ax, fig):
    #     x = np.arange(0, row["coherence_max_f"]+1/row["coherence_fs"], 1/row["coherence_fs"])
    #     err=0
    #     handles=[]
    #     if len(row["coherence"]) >5:
    #         for i, p in enumerate(row["coherence"]):
    #             try:
    #                 r = np.abs(toolbox.get(p))
    #                 if np.mean(r) > 0.7:
    #                     nrow = row.apply(lambda x: x if not isinstance(x, tuple) or i>= len(x) else x[i])
    #                     logger.warning("High coherence (avg = {}) for {}.".format(np.mean(r), nrow.to_dict()))
    #                 ax.plot(x, np.abs(toolbox.get(p)), color="blue")[0]
    #             except:
    #                 err+=1
    #         handles.append(ax.plot([], color="blue", label="plots")[0])
    #     else:
    #         for i, p in enumerate(row["coherence"]):
    #             try:
    #                 r = np.abs(toolbox.get(p))
    #                 if np.mean(r) > 0.7:
    #                     nrow = row.apply(lambda x: x if not isinstance(x, tuple) or i>= len(x) else x[i])
    #                     logger.warning("High coherence (avg = {}) for {}.".format(np.mean(r), nrow.to_dict()))
    #                 handles.append(ax.plot(x, np.abs(toolbox.get(p)), label="plot"+str(i))[0])
    #             except:
    #                 err+=1
    #     handles.append(ax.plot(x, np.abs(toolbox.get(row["Mean"])), color="red", label="avg")[0])
    #     handles.append(ax.plot(x, np.abs(toolbox.get(row["Median"])), color="yellow", label="median")[0])
        

    #     ax.set_xlabel("Frequency (Hz)")
    #     ax.set_ylabel("Amplitude (?)")
    #     ax.set_xlim(3, 60)
    #     ax.set_title("Coherence Amplitude\n"+str(row[["Species",  "Condition", "Structure_1", "signal_resampled_type_1", "Structure_2", "signal_resampled_type_2", "Same_Electrode", "nplots"]].to_dict()) + ", nb_error_plots: "+str(err))
    #     ax.legend(handles=handles, fancybox=True, shadow=True)


import matplotlib.pyplot as plt
import matplotlib as mpl

def add_colors(df: pd.DataFrame):
    group_df = df.drop_duplicates(subset=["Species", "Structure_1", "Structure_2"], keep="first")[["Species", "Structure_1", "Structure_2"]]
    group_df["colormap"] = group_df["Species"].apply(lambda s: "Blues" if s.lower()=="rat" else "Reds" if s.lower()=="monkey" else "Greens")
    group_df["num_colors_in_species"] = group_df.groupby("Species")["colormap"].transform("count")
    group_df["one"] = 1
    group_df["color_id_in_species"] = group_df.groupby("Species")["one"].cumsum() -1
    group_df.drop(columns="one", inplace=True)
    group_df["color"] = group_df.apply(lambda r: 
        mpl.colors.rgb2hex(mpl.colors.to_rgba(
            plt.cm.get_cmap(r["colormap"])((0.5 + r["color_id_in_species"])/r["num_colors_in_species"])
            , 1), keep_alpha=True), axis=1)
    
    res=df.merge(group_df, how="left", on=["Species", "Structure_1", "Structure_2"])
    return res

