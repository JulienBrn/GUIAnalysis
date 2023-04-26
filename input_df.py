from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm


logger=logging.getLogger(__name__)

folder_manager = Manager("./cache/folder_contents")

def mk_monkey_input(base_folder , rescan) -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path(base_folder),
         "columns": ["Condition", "Subject", "Structure", "Date"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)
   if rescan:
      df_handle.invalidate_all()
   df: pd.DataFrame = df_handle.get_result()
   df["Species"] = "Monkey"
   df["Session"] = "MS_#"+ df.groupby(by=["Date", "filename", "Subject"]).ngroup().astype(str)
   df["SubSessionInfo"] = 0
   df["SubSessionInfoType"] = "order"
   df["Channel"] = df["filename"]
   
   def get_monkey_signals(row: pd.Series) -> pd.DataFrame:
      row_raw = row.copy()
      row_raw["signal_type"] = "raw"
      row_raw["signal_fs"] = 25000
      row_raw["file_path"] = row["path"]
      row_raw["file_keys"] = ("RAW", (0,))

      row_spikes = row.copy()
      row_spikes["signal_type"] = "spike_times"
      row_spikes["signal_fs"] = 25000
      row_spikes["file_path"] = row["path"]
      row_spikes["file_keys"] = ("SUA", (0,))

      res = pd.DataFrame([row_raw, row_spikes])
      return res

   tqdm.pandas(desc="Creating monkey metadata")
   return pd.concat(df.progress_apply(get_monkey_signals, axis=1).values, ignore_index=True)

def mk_human_input(base_folder , rescan) -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path(base_folder),
         "columns":["Structure", "Date_HT", "Electrode_Depth"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)
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
      row_spikes["signal_fs"] = 48000 if row["Date"] < "2015_01_01" else 44000
      row_spikes["file_path"] = row["path"]
      row_spikes["file_keys"] = ("SUA", (0,))

      res = pd.DataFrame([row_raw, row_spikes])
      return res
   tqdm.pandas(desc="Creating human metadata")
   return pd.concat(df.progress_apply(get_human_signals, axis=1).values, ignore_index=True)
   
def mk_rat_input(base_folder , rescan) -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path(base_folder),
         "columns":["Condition", "Subject", "Date", "Session", "Structure"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)
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
   
  def __init__(self, dataframe_manager, computation_m):
     self.dataframe_manager = dataframe_manager
     self.computation_m = computation_m
     self.metadata = {
      "input.human.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review",
      "input.human.rescan": "False",
      "input.monkey.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review",
      "input.monkey.rescan": "False",
      "input.rat.base_folder": "/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/NicoAnalysis/Rat_Data",
      "input.rat.rescan": "False",
    }
    #  self._dataframe = self._get_df(dataframe_manager, computation_m)

  
  name = "inputs"
  result_columns = ["signal"]

  def get_df(self):
     return _get_df(self.dataframe_manager, self.computation_m, self.metadata)
  
  def view_item(self, canvas, row):
    
    if not "times"in row["signal_type"]:
      canvas.ax = canvas.fig.subplots(2, 1)
      y = row["signal"].get_result()
      x = np.arange(0, y.shape[0]/row["signal_fs"], 1.0/row["signal_fs"])
      canvas.ax[0].plot(x, y)
      canvas.ax[0].set_xlabel("Time (s)")
      canvas.ax[0].set_ylabel("Amplitude (?)")
      canvas.ax[1].psd(y, Fs=row["signal_fs"])

      
  
  def view_items(self, canvas, row_indices):
     pass
  


  
def _get_df(dataframe_manager, computation_m, metadata):
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

  monkey_input = monkey_input_handle.get_result().drop(columns=["path", "filename", "ext"])
  
  

  logger.info("Getting human_df")
  if not metadata["input.human.rescan"] in ["True", "False"]:
     raise BaseException("Unknown value for option input.human.rescan. Should be in {}".format([True, False]))
  human_input_handle = dataframe_manager.declare_computable_ressource(
        mk_human_input, {"rescan": metadata["input.human.rescan"]=="True", "base_folder": metadata["input.human.base_folder"]}, 
        df_loader, "human_input_df", True
    )
  if metadata["input.human.rescan"] == "True":
     human_input_handle.invalidate_all()
  human_input = human_input_handle.get_result().drop(columns=["path", "filename", "ext", "Date_HT", "Electrode_Depth"])


  logger.info("Getting rat_df")
  if not metadata["input.rat.rescan"] in ["True", "False"]:
     raise BaseException("Unknown value for option input.rat.rescan. Should be in {}".format([True, False]))
  rat_input_handle = dataframe_manager.declare_computable_ressource(
        mk_rat_input, {"rescan": metadata["input.rat.rescan"]=="True", "base_folder": metadata["input.rat.base_folder"]}, 
        df_loader, "rat_input_df", True
    )
  
  if metadata["input.rat.rescan"] == "True":
     rat_input_handle.invalidate_all()
  rat_input = rat_input_handle.get_result().drop(columns=["path", "filename", "ext"])

  input_df = pd.concat([monkey_input, human_input, rat_input], ignore_index=True)[INPUT_Columns]

  subcols = [col for col in input_df.columns if col!="file_path"]
  if input_df.duplicated(subset=subcols).any():
      logger.error(
        "Duplicates in input dataframe. Duplicates are:\n{}".format(
            input_df.duplicated(subset=subcols, keep=False).sort_values(by=subcols)))
  else:
      if input_df.isnull().sum().sum() != 0:
        logger.warning("Number of null values are\n{}".format(input_df.isnull().sum()))
      else:
        logger.info("Metadata seems ok")


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
        res = res[key]
    return res

  input_df = mk_block(
    input_df, ["file_ressource", "file_keys"], get_array_ressource, 
    (np_loader, "signal", False), computation_m)

  input_df["Channel or Neuron"] = input_df["Channel"]
  final_cols = ["signal_type", "signal_fs", "signal", "Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel or Neuron", "file_path", "file_keys"]
  return input_df.copy()[final_cols]