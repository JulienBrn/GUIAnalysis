from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
from typing import List
import bisect

logger=logging.getLogger(__name__)

folder_manager = Manager("./cache/folder_contents")

def mk_monkey_input(base_folder , rescan) -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path(base_folder),
         "columns": ["Condition", "Subject", "Structure", "Date"],
         "pattern": "**/*.mat"}, df_loader, "monkey_input_file_df", save=True)
   if rescan:
      df_handle.invalidate_all()
   df: pd.DataFrame = df_handle.get_result()
   try:
     metadata_df: pd.DataFrame = pd.read_csv(base_folder + "/BothMonkData_withTime.csv", sep=",")
   except:
     logger.warning("BothMonkData_withTime.csv file necessary for monkey metadata not found at {}. Continuing without monkey data...".format(base_folder + "/BothMonkData_withTime.csv"))
     return pd.DataFrame([], columns=["path", "filename", "ext", "Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel", "signal_type", "signal_fs", "file_path", "file_keys"])
   metadata_df["Structure"] = metadata_df["Structure"].str.slice(0,3)
   metadata_df["filename"] = "unit"+ metadata_df["unit"].astype(str)
   df = df.merge(metadata_df, how="left", on=["Condition", "Subject", "Structure", "Date", "filename"])
   if len(df[df["Start"].isna()].index) > 0:
      logger.warning("Ignoring the following entries:\n{}".format(df[df["Start"].isna()]))
      df = df[~df["Start"].isna()].reset_index(drop=True)
   print(df)
   df["Species"] = "Monkey"
   df["Session"] = "MS_#"+ df.groupby(by=["Date", "Subject"]).ngroup().astype(str)
   df["SubSessionInfo"] = list(zip(df["Start"], df["End"]))
   df["SubSessionInfoType"] = "times"
   df["Channel"] = df["filename"]

   def mk_subsession_group(d):
      # times: List[float] = [t for tuple in d["SubSessionInfo"].to_list() for t in tuple]
      times: List[float] = d["Start"].to_list() + d["End"].to_list()
      times = sorted(set(times))
      # print(times)
      def transform_row(r):
         start = r["Start"]
         end = r["End"]
         start_index = bisect.bisect_left(times, start)
         end_index = bisect.bisect_left(times, end)
         start_times =times[start_index:end_index]
         end_times =times[start_index+1:end_index+1]
         # print(start, end)
         # print(pd.DataFrame(zip(start_times, end_times), columns=["Start", "End"]))
         # input()
         ret = pd.DataFrame(zip(start_times, end_times), columns=["Start", "End"])
         ret["path"] = r["path"]
         return ret
      return pd.concat(d.apply(transform_row, axis=1).values, ignore_index=True)
      # return None
   
   subsession_df = df.groupby(by=["Session"]).apply(mk_subsession_group).reset_index()
   print(subsession_df)
   df = subsession_df.merge(df, on=["Session", "path"], how="left", suffixes=("new", ""))
   df["SubSessionInfonew"] = list(zip(df["Startnew"], df["Endnew"]))
   df["SubSessionInfo"] = df["SubSessionInfonew"].astype(str) + "/" +  df["SubSessionInfo"].astype(str)
   # input()
         
   
   def get_monkey_signals(row: pd.Series) -> pd.DataFrame:
      row_raw = row.copy()
      row_raw["signal_type"] = "raw"
      row_raw["signal_fs"] = 25000
      row_raw["file_path"] = row["path"]
      start_index = int((row["Startnew"] - row["Start"])* row_raw["signal_fs"])
      end_index = int((row["Endnew"] - row["Start"])* row_raw["signal_fs"])
      row_raw["file_keys"] = ("RAW", 0, (start_index, end_index))
      row_raw["Duration"] = row["Endnew"] - row["Startnew"]

      row_spikes = row.copy()
      row_spikes["signal_type"] = "spike_times"
      row_spikes["signal_fs"] = 40000
      row_spikes["file_path"] = row["path"]
      start_index = int((row["Startnew"] - row["Start"])* row_spikes["signal_fs"])
      end_index = int((row["Endnew"] - row["Start"])* row_spikes["signal_fs"])
      row_spikes["file_keys"] = ("SUA", 0, (start_index, end_index))
      row_spikes["Duration"] = row["Endnew"] - row["Startnew"]

      res = pd.DataFrame([row_raw, row_spikes])
      return res

   tqdm.pandas(desc="Creating monkey metadata")
   return pd.concat(df.progress_apply(get_monkey_signals, axis=1).values, ignore_index=True)

def mk_human_input(base_folder , rescan) -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path(base_folder),
         "columns":["Structure", "Date_HT", "Electrode_Depth"],
         "pattern": "**/*.mat"}, df_loader, "human_input_file_df", save=True)
   if rescan:
      df_handle.invalidate_all()
   df: pd.DataFrame = df_handle.get_result()
   df["Species"] = "Human"
   df["Condition"] = "pd"
   df["Date"] = df["Date_HT"].str.slice(0, 10)
   df["Subject"] = "#"+ df.groupby("Date").ngroup().astype(str)
   df["Session"] = "HS_#"+ df.groupby(by=["Date_HT", "Electrode_Depth", "Subject"]).ngroup().astype(str)
   df["Channel"] = df["filename"] 
   df["SubSessionInfo"] = 0
   df["SubSessionInfoType"] = "order"

   def get_human_signals(row: pd.Series) -> pd.DataFrame:
      row_raw = row.copy()
      row_raw["signal_type"] = "mua"
      row_raw["signal_fs"] = 48000 if row["Date"] < "2015_01_01" else 44000
      row_raw["file_path"] = row["path"]
      row_raw["file_keys"] = ("MUA", (0,))

      row_spikes = row.copy()
      row_spikes["signal_type"] = "spike_times"
      row_spikes["signal_fs"] = 1
      row_spikes["file_path"] = row["path"]
      row_spikes["file_keys"] = ("SUA",)

      res = pd.DataFrame([row_raw, row_spikes])
      
      
      f = matlab_loader.load(row["path"])
      def filter_line(r):
         ktuple = ast.literal_eval(r["file_keys"]) if isinstance(r["file_keys"], str) else r["file_keys"]
         ret = f
         for key in ktuple:
            ret = ret[key]
         if ret.size > 10:
            # if ret.size<20:
            #    logger.info("Saving {}[{}]\nContents: {}".format(row["path"], r["file_keys"], ret))
            return True
         else:
            logger.warning("Removing {}[{}]\nContents: {}".format(row["path"], r["file_keys"], ret))
            return False
      filtered_res = res[res.apply(filter_line, axis=1)]
      return filtered_res
   tqdm.pandas(desc="Creating human metadata")
   res = pd.concat(df.progress_apply(get_human_signals, axis=1).values, ignore_index=True)
   # duplicated = res[res["signal_type"]=="mua"].duplicated(subset=["Session"])
   # duplicated_bis = res[res["signal_type"]=="mua"].duplicated(subset=["Date", "Electrode_Depth", "Subject"])
   # logger.info("#Human Session BUA duplicated = {}".format(duplicated.sum()))
   # logger.info("#Human Session BUA duplicatedbis = {}".format(duplicated_bis.sum()))
   # df_loader.save("add_human_duplication.tsv", res.loc[duplicated_bis & (~duplicated), ["file_path", "file_keys"]].copy())

   res_raw=res[res["signal_type"]=="mua"].copy()
   res_spikes=res[res["signal_type"]!="mua"].copy()
   res_raw.drop_duplicates(subset=["Session"], inplace=True)
   res=pd.concat([res_raw, res_spikes], ignore_index=True)
   return res
   
def mk_rat_input(base_folder , rescan) -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path(base_folder),
         "columns":["Condition", "Subject", "Date", "Session", "Structure"],
         "pattern": "**/*.mat"}, df_loader, "rat_input_file_df", save=True)
   if rescan:
      df_handle.invalidate_all()
   df: pd.DataFrame = df_handle.get_result()
   df["Species"] = "Rat"
   session_regex = re.compile("(?P<word>[a-z]+)(?P<num>[0-9]+)", re.IGNORECASE)
   df[["Session", "SubSessionInfo"]] = df.apply(
      lambda row: ["RS_"+ str(row["Date"]) + "_"+ str(session_regex.fullmatch(row["Session"]).group("word")), session_regex.fullmatch(row["Session"]).group("num")], 
      axis=1, result_type="expand"
   )
   df["SubSessionInfoType"] = "order"
   df["Structure"] = df.apply(lambda row: row["Structure"] if row["Structure"]!="Striatum" else "STR", axis=1)
   rat_raw_regexs = [
         re.compile("(?P<sln>.*)_Probe(?P<channel>.*)"),
         re.compile("(?P<sln>.*)_(?P<channel>EEG)ipsi", re.IGNORECASE),
         re.compile("(?P<sln>.*)_ipsi(?P<channel>EEG)", re.IGNORECASE),
         re.compile("(?P<sln>.*)_(?P<channel>EEG)", re.IGNORECASE)
      ]
   rat_spikes_regexs = [
         re.compile("(?P<sln>.*)_Pr(?P<channel>[0-9]+)_(?P<neuron>.*)"),
         re.compile("(?P<sln>.*)_Probe(?P<channel>[0-9]+)(?P<neuron>.*)"),
         re.compile("(?P<sln>.*)_mPr(?P<channel>[0-9]+)_(?P<neuron>.*)"), #make better
         re.compile("(?P<sln>.*)_P(?P<channel>[0-9]+)_(?P<neuron>.*)"),  #make better
         re.compile("(?P<sln>.*)_Pr_(?P<channel>[0-9]+)_(?P<neuron>.*)"),
         re.compile("(?P<sln>.*)_Pr_(?P<channel>[0-9]+)(?P<neuron>)"),
         re.compile("(?P<sln>.*)_Pr(?P<channel>[0-9]+)(?P<neuron>)"),
         re.compile("(?P<sln>.*)_(?P<neuron>(SS)|(mSS))_Pr_(?P<channel>[0-9]+)"),
         re.compile("(?P<sln>.*)_(?P<neuron>(SS)|(mSS))_(?P<channel>STN)"),#make better
         re.compile("(?P<sln>.*)_(?P<neuron>(SS)|(mSS))_(?P<channel>STN).*"),#make better
         re.compile("(?P<sln>.*)_(?P<channel>All)_(?P<neuron>STR)"),
         re.compile("(?P<channel>.*)(?P<neuron>)"),
      ]
   def raise_print(o):
      logger.error("Exception thrown with object:\n{}".format(o))
      raise BaseException("Error")
   def get_rat_signals(row: pd.Series) -> pd.DataFrame:
      with h5py.File(row["path"], 'r') as file:
         if row["filename"] != "Units":
            channel_dict = {key:{
               "Channel": key_match.group("channel") if key_match else raise_print(key),
               "signal_fs": int(1/file[key]["interval"][0,0])
               } for key in file.keys() for key_match in [next((v for v in [regex.fullmatch(key) for regex in rat_raw_regexs] if v), None)]}
            res = pd.DataFrame.from_dict(channel_dict, orient="index").reset_index(names="file_keys")
            res["file_keys"] = res.apply(lambda row: (row["file_keys"], "values"), axis=1)
            res["file_path"] = row["path"]
            res["signal_type"] = "raw"
         else:
            channel_dict = {key:{
               "Channel": "chan?:" + key_match.group("channel") + "neuron:"+ key_match.group("neuron")  if key_match else raise_print(key),
               "signal_fs": 1
               } for key in file.keys() for key_match in [next((v for v in [regex.fullmatch(key) for regex in rat_spikes_regexs] if v), None)]}
            res = pd.DataFrame.from_dict(channel_dict, orient="index").reset_index(names="file_keys")
            res["file_keys"] = res.apply(lambda row: (row["file_keys"], "times"), axis=1)
            res["file_path"] = row["path"]
            res["signal_type"] = "spike_times"
         for col in row.index:
            res[col] = row[col]
         return res
   tqdm.pandas(desc="Creating rat metadata")
   return pd.concat(df.progress_apply(get_rat_signals, axis=1).values, ignore_index=True)


class InputDataDF:
   
  def __init__(self, dataframe_manager, computation_m, step_signals):
     self.dataframe_manager = dataframe_manager
     self.computation_m = computation_m
     self.metadata = {
      "input.human.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review",
      "input.human.rescan": "False",
      "input.human.size": "-1",
      "input.monkey.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review",
      "input.monkey.rescan": "False",
      "input.monkey.size": "-1",
      "input.rat.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/NicoAnalysis/Rat_Data",
      "input.rat.rescan": "False",
      "input.rat.size": "-1",
     }
     self.invalidated = True
     self.step_signals = step_signals

  
  name = "inputs"
  result_columns = ["signal"]
  invalidated = True



  def get_df(self):
     signal_cols = ["Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel or Neuron", "signal_type", "signal_fs", "signal"]
     if self.invalidated:
        self._dataframe = _get_df(self.dataframe_manager, self.computation_m, self.metadata)
        self.step_signals["input"] = self._dataframe.copy()[signal_cols]
        self.invalidated = False
     return self._dataframe
  
  def view_item(self, canvas, row):
    
    if not "times"in row["signal_type"]:
      canvas.ax = canvas.fig.subplots(2, 1)
      y = row["signal"].get_result()
      x = np.arange(0, y.shape[0]/row["signal_fs"], 1.0/row["signal_fs"])
      canvas.ax[0].plot(x, y)
      canvas.ax[0].set_xlabel("Time (s)")
      canvas.ax[0].set_ylabel("Amplitude (?)")
      canvas.ax[1].psd(y, Fs=row["signal_fs"])

      
  
  # def view_items(self, canvas, row_indices):
  #    pass
  


  
def _get_df(dataframe_manager, computation_m, metadata):
 try:
  INPUT_Columns = ["Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel", "signal_type", "signal_fs", "file_path", "file_keys"]
  logger.info("Getting input_df")
  logger.info("Getting monkey_df")


  if not metadata["input.monkey.rescan"] in ["True", "False"]:
     raise BaseException("Unknown value for option input.monkey.rescan. Should be in {}".format([True, False]))
  
  monkey_input_handle = dataframe_manager.declare_computable_ressource(
          mk_monkey_input, {"rescan": metadata["input.monkey.rescan"]=="True", "base_folder": metadata["input.monkey.base_folder"]},
          df_loader, "monkey_input_df", True
      )
  if metadata["input.monkey.rescan"] == "True":
     monkey_input_handle.invalidate_all()

#   monkey_input_handle.invalidate_all()#to remove

  monkey_input = monkey_input_handle.get_result().drop(columns=["path", "filename", "ext"])
  if metadata["input.monkey.size"].isdigit():
     monkey_input=monkey_input.iloc[0:int(metadata["input.monkey.size"]), :]
  

  logger.info("Getting human_df")
  if not metadata["input.human.rescan"] in ["True", "False"]:
     raise BaseException("Unknown value for option input.human.rescan. Should be in {}".format([True, False]))
  human_input_handle = dataframe_manager.declare_computable_ressource(
        mk_human_input, {"rescan": metadata["input.human.rescan"]=="True", "base_folder": metadata["input.human.base_folder"]}, 
        df_loader, "human_input_df", True
    )
  if metadata["input.human.rescan"] == "True":
     human_input_handle.invalidate_all()
     human_input_handle2 = dataframe_manager.declare_computable_ressource(
        mk_human_input, {"rescan": False, "base_folder": metadata["input.human.base_folder"]}, 
        df_loader, "human_input_df", True
     )
     human_input_handle2.invalidate_all()

#   human_input_handle.invalidate_all()#to remove
# 
  human_input = human_input_handle.get_result().drop(columns=["path", "filename", "ext", "Date_HT", "Electrode_Depth"])
  if metadata["input.human.size"].isdigit():
     human_input=human_input.iloc[0:int(metadata["input.human.size"]), :]

  logger.info("Getting rat_df")
  if not metadata["input.rat.rescan"] in ["True", "False"]:
     raise BaseException("Unknown value for option input.rat.rescan. Should be in {}".format([True, False]))
  rat_input_handle = dataframe_manager.declare_computable_ressource(
        mk_rat_input, {"rescan": metadata["input.rat.rescan"]=="True", "base_folder": metadata["input.rat.base_folder"]}, 
        df_loader, "rat_input_df", True
    )
  
  if metadata["input.rat.rescan"] == "True":
     rat_input_handle.invalidate_all()
     rat_input_handle2 = dataframe_manager.declare_computable_ressource(
        mk_rat_input, {"rescan": False, "base_folder": metadata["input.rat.base_folder"]}, 
        df_loader, "rat_input_df", True
    )
     rat_input_handle2.invalidate_all()

#   rat_input_handle.invalidate_all()#to remove

  rat_input = rat_input_handle.get_result().drop(columns=["path", "filename", "ext"])
  if metadata["input.rat.size"].isdigit():
     rat_input=rat_input.iloc[0:int(metadata["input.rat.size"]), :]

  input_df = pd.concat([monkey_input, human_input, rat_input], ignore_index=True)[INPUT_Columns]

  subcols = [col for col in input_df.columns if col!="file_path"]
#   if input_df.duplicated(subset=subcols).any():
#       logger.error(
#         "Duplicates in input dataframe. Duplicates are:\n{}".format(
#             input_df.duplicated(subset=subcols, keep=False).sort_values()))
#   else:
#       if input_df.isnull().sum().sum() != 0:
#         logger.warning("Number of null values are\n{}".format(input_df.isnull().sum()))
#       else:
#         logger.info("Metadata seems ok")


  def get_file_ressource(d):
    if pathlib.Path(d["file_path"].iat[0]).stem != "Units":
        ret =  computation_m.declare_ressource(d["file_path"].iat[0], matlab_loader, check=False)
    else:
        ret =  computation_m.declare_ressource(d["file_path"].iat[0], matlab_loader, check=False)
    return d.apply(lambda row: ret, axis=1)

  tqdm.pandas(desc="Declaring file ressources")

  input_df["file_ressource"] = input_df.groupby("file_path", group_keys=False).progress_apply(get_file_ressource)

  tqdm.pandas(desc="Declaring array ressources")

  def get_array_ressource(file_ressource, file_keys):
    ktuple = ast.literal_eval(file_keys) if isinstance(file_keys, str) else file_keys
    res = file_ressource
    for key in ktuple:
        if hasattr(key, "__len__") and len(key) ==2:
           res = res[key[0]:key[1]]
        else:
           res = res[key]
    return res.reshape(-1) if hasattr(res, "reshape") else res

  input_df = mk_block(
    input_df, ["file_ressource", "file_keys"], get_array_ressource, 
    (np_loader, "signal", False), computation_m)

  input_df["Channel or Neuron"] = input_df["Channel"]
  final_cols = ["signal_type", "signal_fs", "signal", "Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel or Neuron", "file_path", "file_keys"]
  
  return input_df.copy()[final_cols]
 except:
  logger.error("Problem getting input_df. Probably the given folder paths are wrong")
  return pd.DataFrame([], columns=["signal_type", "signal_fs", "signal", "Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel or Neuron", "file_path", "file_keys"])
