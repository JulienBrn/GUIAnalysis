from toolbox import json_loader, RessourceHandle, Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
from importlib.resources import files as package_data
import os

logger=logging.getLogger(__name__)

import sys
import matplotlib, importlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie
from PyQt5.QtCore import pyqtSlot
from  toolbox import DataFrameModel
from analysisGUI.main_window_ui import Ui_MainWindow
from PyQt5.uic import loadUi
import PyQt5.QtCore as QtCore
from multiprocessing import Process
from threading import Thread
import tkinter
from PyQt5.QtCore import QThread, pyqtSignal
from analysisGUI.mplwidget import MplCanvas, MplWidget


def mk_result_tab():
   result_tab = QtWidgets.QWidget()
   verticalLayout_4 = QtWidgets.QVBoxLayout(result_tab)
   mpl = MplWidget(result_tab)
   verticalLayout_4.addWidget(mpl)
   toolbar = NavigationToolbar(mpl.canvas, parent=result_tab)
   return result_tab, mpl

class GetResult(QThread):
   progress = pyqtSignal(int)
   def __init__(self, model, indices, cols):
      super().__init__()
      self.model = model
      self.indices = indices
      self.cols = cols
   def run(self):
      model = self.model
      indices = self.indices
      df = model._dataframe
      for index in tqdm(indices):
         for colind, col in enumerate(self.cols):
            if isinstance(df[col].iat[index], toolbox.RessourceHandle):
               df[col].iat[index].get_result()
               model.dataChanged.emit(
                  model.createIndex(index,colind),  model.createIndex(index+1,colind+1), (QtCore.Qt.EditRole,)
               ) 
         self.progress.emit(1)

class GetDataframe(QThread):
   dfcomputed = pyqtSignal(pd.DataFrame)
   def __init__(self, df):
      super().__init__()
      self.df = df
   def run(self):
    try:
      res = self.df.get_df().reset_index(drop=True)
      self.dfcomputed.emit(res)
    except:
      logger.error("Impossible to get dataframe")
      self.dfcomputed.emit(pd.DataFrame([], columns=["Error"]))

class ViewResult(QThread):
   ready = pyqtSignal(int)
   def __init__(self, df, result_tabs, rows):
      super().__init__()
      self.df = df
      if hasattr(df, "get_nb_figs"):
         nb_figs = df.get_nb_figs(rows)
      else:
         nb_figs = 1
      self.canvas=[]
      for i in range(nb_figs):
         result_tab, mpl = mk_result_tab()
         self.canvas.append(mpl.canvas)
         result_tabs.addTab(result_tab, "res"+str(i))
      self.rows = rows
   def run(self):
      if hasattr(self.df, "show_figs"):
         for l in self.df.show_figs(self.rows, self.canvas):
            logger.info("Emitting {}".format(l))
            for i in l:
               self.ready.emit(i)
      else:
         canvas = self.canvas[0]
         if len(self.rows.index) == 1 or not hasattr(self.df, "view_items"):
            self.df.view_item(canvas, self.rows.iloc[0, :])
         else:
            self.df.view_items(canvas, self.rows)
         self.ready.emit(0)

class Window(QMainWindow, Ui_MainWindow):
    setup_ready = pyqtSignal(dict)
    def __init__(self, parent=None):
      super().__init__(parent)
      self.setupUi(self)
      self.dfs = []
      self.df_listmodel = QStandardItemModel()
      self.listView.setModel(self.df_listmodel)
      self.df_models = []
      self.curr_df = None
      self.process= None
      self.tableView.setSortingEnabled(True)
      self.splitter.setStretchFactor(1,6)
      self.progressBar.setValue(0)
      #   self.toolbar = NavigationToolbar(self.mpl.canvas, parent=self.result_tab)
      #   self.result_tabs.setTabsClosable(True)

      self.setup_model = QStandardItemModel()
      self.setup_model.setHorizontalHeaderLabels(['Parameter Name', 'Parameter Value'])
      self.treeView.setModel(self.setup_model)
      self.treeView.header().setDefaultSectionSize(250)
      self.setup_params=[self.setup_model.invisibleRootItem(), None, {}]


      def compute(indices):
         if self.curr_df is None:
            logger.error("Nothing to compute")
         elif self.process and self.process.isRunning():
            logger.error("A computational process is already running, please wait")
         else:
            self.progressBar.setMaximum(len(indices))
            self.progressBar.setValue(0)
            def progress(amount):
               self.progressBar.setValue(self.progressBar.value()+amount)

            self.process = GetResult(self.tableView.model(), indices, self.dfs[self.curr_df].result_columns)
            self.process.progress.connect(progress)
            self.process.start()
          
      def view(indices):
         # if not self.mpl.canvas.ax is None:
         #   if hasattr(self.mpl.canvas.ax, "flat"):
         #     for ax in self.mpl.canvas.ax.flat:
         #       ax.remove()
         #   else: 
         #       self.mpl.canvas.ax.remove()
         #   self.mpl.canvas.draw()
         
         self.result_tabs.clear()
         #  result_tab, mpl = mk_result_tab()
         #  self.result_tabs.addTab(result_tab, "res")
         # #  self.toolbar.setParent(None)
         #  self.mpl.reset()
         #  self.toolbar = NavigationToolbar(self.mpl.canvas, parent=self.result_tab)
         if hasattr(self.dfs[self.curr_df], "view_params"):
            params = {}
            def rec_print(root, prefix):
               if root[2] == {}:
                  params[prefix+root[0].text()] = root[1].text()
               for child, val in root[2].items():
                  rec_print(val, prefix+root[0].text()+".")
            for child, val in self.view_params_dict[2].items():
               rec_print(val, "")
            self.dfs[self.curr_df].view_params = params
         
         self.process = ViewResult(self.dfs[self.curr_df], self.result_tabs, self.tableView.model()._dataframe.iloc[indices, :])
         self.tabWidget.setCurrentWidget(self.result_tab)
         self.loader_label.setVisible(True)
         def when_ready(i):
            self.process.canvas[i].draw()
            #   self.toolbar.update()
            if i == len(self.process.canvas)-1:
               self.loader_label.setVisible(False)
               self.figs =  [canvas.fig for canvas in self.process.canvas]
         self.process.ready.connect(when_ready)
         self.process.start()
            # self.process.run()
            # when_ready()

      def invalidate(indices):
         for i in tqdm(indices):
            for col in self.dfs[self.curr_df].result_columns:
               val = self.tableView.model()._dataframe[col].iat[i]
               if isinstance(val, RessourceHandle):
                  val.invalidate_all()
         self.tableView.model().dataChanged.emit(
            self.tableView.model().createIndex(0,0), self.tableView.model().createIndex(len(self.tableView.model()._dataframe.index), len(self.tableView.model()._dataframe.columns)), (QtCore.Qt.EditRole,)
         ) 
      def get_next():
         current_position = self.tableView.selectionModel().selectedRows()
         if current_position == []:
            current_position = -1
         else:
            current_position = current_position[0].row()
         query_str = str(self.query.text())
         if query_str == "":
            return
         self.tableView.clearSelection()
         #   logger.info("curr_pos = {}, query = {}".format(current_position, query_str))
         def verifies_query(row):
            items = query_str.split(",")
            row_str = " ".join([str(v) for v in row.values])
            for it in items:
               if not it in row_str:
                  return False
            return True
         positions = self.tableView.model()._dataframe.apply(verifies_query, axis=1)
         new_pos = self.tableView.model()._dataframe[(positions) & (self.tableView.model()._dataframe.index > current_position)]
         logger.info(new_pos)
         if len(new_pos.index) ==0:
            new_pos = self.tableView.model()._dataframe[(positions)]
         if len(new_pos.index) >0:
            logger.info("moving to {}".format(new_pos.index[0]))
            self.tableView.scrollTo(self.tableView.model().createIndex(new_pos.index[0],0))
            select = QtCore.QItemSelection()
            for i in new_pos.index:
               select.select(self.tableView.model().createIndex(i,0), self.tableView.model().createIndex(i,len(self.tableView.model()._dataframe.columns)))
            self.tableView.selectionModel().select(select , QtCore.QItemSelectionModel.Select)
            self.tableView.model().dataChanged.emit(
            self.tableView.model().createIndex(0,0), self.tableView.model().createIndex(len(self.tableView.model()._dataframe.index), len(self.tableView.model()._dataframe.columns)), (QtCore.Qt.EditRole,)
            ) 
      def get_prev():
         current_position = self.tableView.selectionModel().selectedRows()
         if current_position == []:
            current_position = len(self.tableView.model()._dataframe.index)
         else:
            current_position = current_position[len(current_position)-1].row()
         query_str = str(self.query.text())
         if query_str == "":
            return
         self.tableView.clearSelection()
      #   logger.info("curr_pos = {}, query = {}".format(current_position, query_str))
         def verifies_query(row):
            items = query_str.split(",")
            row_str = " ".join([str(v) for v in row.values])
            for it in items:
               if not it in row_str:
                  return False
            return True
         positions = self.tableView.model()._dataframe.apply(verifies_query, axis=1)
         new_pos = self.tableView.model()._dataframe[(positions) & (self.tableView.model()._dataframe.index < current_position)]
         logger.info(new_pos)
         if len(new_pos.index) ==0:
            new_pos = self.tableView.model()._dataframe[(positions)]
         if len(new_pos.index) >0:
            logger.info("moving to {}".format(new_pos.index[0]))
            self.tableView.scrollTo(self.tableView.model().createIndex(new_pos.index[0],0))
            select = QtCore.QItemSelection()
            for i in new_pos.index:
               select.select(self.tableView.model().createIndex(i,0), self.tableView.model().createIndex(i,len(self.tableView.model()._dataframe.columns)))
            self.tableView.selectionModel().select(select , QtCore.QItemSelectionModel.Select)
            self.tableView.model().dataChanged.emit(
               self.tableView.model().createIndex(0,0), self.tableView.model().createIndex(len(self.tableView.model()._dataframe.index), len(self.tableView.model()._dataframe.columns)), (QtCore.Qt.EditRole,)
            ) 

      self.invalidate.clicked.connect(lambda: invalidate([i.row() for i in self.tableView.selectionModel().selectedRows()]))
      self.compute.clicked.connect(lambda: compute([i.row() for i in self.tableView.selectionModel().selectedRows()]))
      self.view.clicked.connect(lambda: view([i.row() for i in self.tableView.selectionModel().selectedRows()]))
      self.exportall.clicked.connect(lambda: self.export_all_figures())
      self.export_btn.clicked.connect(self.save_df_file_dialog)
      self.next.clicked.connect(get_next)
      self.previous.clicked.connect(get_prev)
      self.tabWidget.currentChanged.connect(lambda index: self.on_computation_tab_clicked() if index==1 else None)

        

      def load_config():
         path, ok = QFileDialog.getOpenFileName(self, caption="Setup parameters to load from", filter="*.json")
         try:
            self.set_setup_params(json_loader.load(path))
         except:
            logger.error("Impossible to load configuration file")
      self.load_params.clicked.connect(load_config)

      def export_config():
         path, ok = QFileDialog.getSaveFileName(self, caption="Save setup parameters to", filter="*.json")
         try:
            json_loader.save(path, self.get_setup_params())
         except BaseException as e:
            logger.error("Impossible to save configuration :{}\n".format(e))
      self.export_params.clicked.connect(export_config)




      self.loader_label = QtWidgets.QLabel(self.centralwidget)
      # self.label.setGeometry(QtCore.QRect(0, 0, 0, 0))
      
      self.loader_label.setMinimumSize(QtCore.QSize(250, 250))
      self.loader_label.setMaximumSize(QtCore.QSize(250, 250))

      # Loading the GIF
      self.movie = QMovie(str(package_data("analysisGUI.ui").joinpath("loader.gif")))
      self.loader_label.setMovie(self.movie)

      self.movie.start()
      self.loader_label.setVisible(False)

    def resizeEvent(self, event):
          #  super(Ui_MainWindow, self).resizeEvent(event)
           super(QMainWindow, self).resizeEvent(event)
           self.move_loader()

    def move_loader(self):
      self.loader_label.move(int(self.size().width()/2), int(self.size().height()/2)-125)
      pass


    def add_df(self, df, switch = False):
        self.dfs.append(df)
        self.df_models.append(None)#DataFrameModel(df.get_df().reset_index(drop=True))
        self.df_listmodel.appendRow(QStandardItem(df.name))

        params = df.metadata
        for p, val in params.items():
           keys = p.split(".")
           curr = self.setup_params
           for k in keys:
              if not k in curr[2]:
                curr[0].appendRow([QStandardItem(k), QStandardItem("")])
                if not curr[1] is None:
                  curr[1].setEditable(False)
                curr[0].child(curr[0].rowCount() - 1, 1).setEditable(True)
                curr[0].child(curr[0].rowCount() - 1, 0).setEditable(False)
                curr[2][k]=[curr[0].child(curr[0].rowCount() - 1, 0), curr[0].child(curr[0].rowCount() - 1, 1), {}]
              curr = curr[2][k]
           curr[1].setText(str(val))
        self.treeView.expandAll()

        if switch:
          self.on_listView_clicked(self.listView.model().index(len(self.dfs) -1, 0))
          
          # self.curr_df = len(self.dfs) -1
          # self.tableView.setModel(self.df_models[self.curr_df])
       
    def on_computation_tab_clicked(self):
      setup_params = self.get_setup_params()
      self.setup_ready.emit(setup_params)
      for i in range(len(self.df_models)):
        if {k:v for k, v in setup_params.items() if k in self.dfs[i].metadata} != self.dfs[i].metadata:
          self.dfs[i].metadata = {k:v for k, v in self.get_setup_params().items() if k in self.dfs[i].metadata}
          self.dfs[i].invalidated=True


      if self.curr_df is None:
        self.curr_df = 0
      self.on_listView_clicked(self.listView.model().index(self.curr_df, 0))

    def set_setup_params(self, params):
      for p, val in params.items():
          keys = p.split(".")
          curr = self.setup_params
          for k in keys:
            if not k in curr[2]:
              curr[0].appendRow([QStandardItem(k), QStandardItem("")])
              if not curr[1] is None:
                curr[1].setEditable(False)
              curr[0].child(curr[0].rowCount() - 1, 1).setEditable(True)
              curr[0].child(curr[0].rowCount() - 1, 0).setEditable(False)
              curr[2][k]=[curr[0].child(curr[0].rowCount() - 1, 0), curr[0].child(curr[0].rowCount() - 1, 1), {}]
            curr = curr[2][k]
          curr[1].setText(str(val))
      self.treeView.expandAll()
       
    def get_setup_params(self):
      params = {}
      def rec_print(root, prefix):
        if root[2] == {}:
           params[prefix+root[0].text()] = root[1].text()
        for child, val in root[2].items():
            rec_print(val, prefix+root[0].text()+".")
      for child, val in self.setup_params[2].items():
        rec_print(val, "")
      return params
        
    def save_df_file_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(
            self,
            "Select file to export to", 
            filter = "(*.tsv)"
        )
        if filename:
            to_save_df = self.dfs[self.curr_df].get_df()
            to_save_df["coherence_pow_path"] = to_save_df.apply(lambda row: str(row["coherence_pow"].get_disk_path()), axis=1)
            to_save_df["coherence_pow_core_path"] = to_save_df.apply(lambda row: str(row["coherence_pow"].manager.d[row["coherence_pow"].id]._core_path), axis=1)
            df_loader.save(filename, to_save_df)

    def export_all_figures(self):
       dir = QFileDialog.getExistingDirectory(self, "Select folder to export to")
       if dir:
          for i, fig in enumerate(self.figs):
             fig.savefig(pathlib.Path(dir) / "figure_{}.png".format(i), dpi=200)
          logger.info("exported")


    @QtCore.pyqtSlot("QModelIndex")
    def on_listView_clicked(self, model_index):
       self.listView.setCurrentIndex(model_index)
       self.curr_df = model_index.row()
       if self.dfs[self.curr_df].invalidated:
          # self.dfs[self.curr_df].metadata = {k:v for k, v in self.get_setup_params().items() if k in self.dfs[self.curr_df].metadata}
          self.loader_label.setVisible(True)
          self.process = GetDataframe(self.dfs[self.curr_df])
          def dataframe_ready(df):
             self.df_models[self.curr_df] = DataFrameModel(df)
             self.tableView.setModel(self.df_models[self.curr_df])
             self.loader_label.setVisible(False)
          self.process.dfcomputed.connect(dataframe_ready)
          self.process.start()
       else:
          self.df_models[self.curr_df] = DataFrameModel(self.dfs[self.curr_df].get_df())
          self.tableView.setModel(self.df_models[self.curr_df])
          self.tableView.setModel(self.df_models[self.curr_df])

       self.view_params_model = QStandardItemModel()
       self.view_params_model.setHorizontalHeaderLabels(['Parameter Name', 'Parameter Value'])
       self.view_params.setModel(self.view_params_model)
       self.view_params.header().setDefaultSectionSize(120)
       self.view_params_dict=[self.view_params_model.invisibleRootItem(), None, {}]

       view_params = self.dfs[self.curr_df].view_params if hasattr(self.dfs[self.curr_df], "view_params") else {}
       for p, val in view_params.items():
          keys = p.split(".")
          curr = self.view_params_dict
          for k in keys:
            if not k in curr[2]:
              curr[0].appendRow([QStandardItem(k), QStandardItem("")])
              if not curr[1] is None:
                curr[1].setEditable(False)
              curr[0].child(curr[0].rowCount() - 1, 1).setEditable(True)
              curr[0].child(curr[0].rowCount() - 1, 0).setEditable(False)
              curr[2][k]=[curr[0].child(curr[0].rowCount() - 1, 0), curr[0].child(curr[0].rowCount() - 1, 1), {}]
            curr = curr[2][k]
          curr[1].setText(str(val))
       self.view_params.expandAll()


