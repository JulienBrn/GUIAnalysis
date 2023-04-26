from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm


import sys
import matplotlib, importlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel
from PyQt5.QtCore import pyqtSlot
from  toolbox import DataFrameModel
from main_window_ui import Ui_MainWindow
from PyQt5.uic import loadUi
import PyQt5.QtCore as QtCore
from multiprocessing import Process
from threading import Thread
import tkinter
from PyQt5.QtCore import QThread, pyqtSignal

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
      for index in indices:
         for colind, col in enumerate(self.cols):
            if isinstance(df[col].iat[index], toolbox.RessourceHandle):
               df[col].iat[index].get_result()
               model.dataChanged.emit(
                  model.createIndex(index,colind),  model.createIndex(index+1,colind+1), (QtCore.Qt.EditRole,)
               ) 
         self.progress.emit(1)

class Window(QMainWindow, Ui_MainWindow):
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
           if len(indices) == 1:
              self.dfs[self.curr_df].view_item(self.mpl.canvas, self.tableView.model()._dataframe.iloc[indices[0], :])
              self.tabWidget.setCurrentWidget(self.result_tab)

        self.compute.clicked.connect(lambda: compute([i.row() for i in self.tableView.selectionModel().selectedRows()]))
        self.view.clicked.connect(lambda: view([i.row() for i in self.tableView.selectionModel().selectedRows()]))
        self.export_btn.clicked.connect(self.save_df_file_dialog)

    def add_df(self, df, switch = False):
        self.dfs.append(df)
        self.df_models.append(DataFrameModel(df.get_df().reset_index(drop=True)))
        self.df_listmodel.appendRow(QStandardItem(df.name))

        if self.curr_df is None or switch:
          self.curr_df = len(self.dfs) -1
          self.tableView.setModel(self.df_models[self.curr_df])
          self.listView.setCurrentIndex(self.listView.model().index(self.curr_df, 0))

    def save_df_file_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(
            self,
            "Select file to export to", 
            filter = "(*.tsv)"
        )
        if filename:
            df_loader.save(filename, self.dfs[self.curr_df].get_df())




    @QtCore.pyqtSlot("QModelIndex")
    def on_listView_clicked(self, model_index):
       self.curr_df = model_index.row()
       self.tableView.setModel(self.df_models[self.curr_df])

