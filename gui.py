from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm


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

class GetDataframe(QThread):
   dfcomputed = pyqtSignal(pd.DataFrame)
   def __init__(self, df):
      super().__init__()
      self.df = df
   def run(self):
      res = self.df.get_df().reset_index(drop=True)
      self.dfcomputed.emit(res)

class ViewResult(QThread):
   ready = pyqtSignal()
   def __init__(self, df, canvas, rows):
      super().__init__()
      self.df = df
      self.canvas = canvas
      self.rows = rows
   def run(self):
      self.df.view_item(self.canvas, self.rows)
      self.ready.emit()

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
        self.toolbar = NavigationToolbar(self.mpl.canvas, self.mpl)


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
           if len(indices) == 1:
              if not self.mpl.canvas.ax is None:
                if hasattr(self.mpl.canvas.ax, "flat"):
                  for ax in self.mpl.canvas.ax.flat:
                    ax.remove()
                else: 
                   self.mpl.canvas.ax.remove()
                self.mpl.canvas.draw()
              self.process = ViewResult(self.dfs[self.curr_df], self.mpl.canvas, self.tableView.model()._dataframe.iloc[indices[0], :])
              self.tabWidget.setCurrentWidget(self.result_tab)
              self.loader_label.setVisible(True)
              def when_ready():
                 self.mpl.canvas.draw()
                 self.loader_label.setVisible(False)
              self.process.ready.connect(when_ready)
              self.process.start()
              # self.process.run()
              # when_ready()

        self.compute.clicked.connect(lambda: compute([i.row() for i in self.tableView.selectionModel().selectedRows()]))
        self.view.clicked.connect(lambda: view([i.row() for i in self.tableView.selectionModel().selectedRows()]))
        self.export_btn.clicked.connect(self.save_df_file_dialog)
        self.tabWidget.currentChanged.connect(lambda index: self.on_computation_tab_clicked() if index==1 else None)





        self.loader_label = QtWidgets.QLabel(self.centralwidget)
        # self.label.setGeometry(QtCore.QRect(0, 0, 0, 0))
        
        self.loader_label.setMinimumSize(QtCore.QSize(250, 250))
        self.loader_label.setMaximumSize(QtCore.QSize(250, 250))
  
        # Loading the GIF
        self.movie = QMovie("ui/loader.gif")
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
      for i in range(len(self.df_models)):
        if self.df_models[i] is None or {k:v for k, v in self.get_setup_params().items() if k in self.dfs[i].metadata} != self.dfs[i].metadata:
          self.dfs[i].metadata = {k:v for k, v in self.get_setup_params().items() if k in self.dfs[i].metadata}
          self.dfs[i].invalidated=True


      if self.curr_df is None:
        self.curr_df = 0
      self.on_listView_clicked(self.listView.model().index(self.curr_df, 0))

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
            df_loader.save(filename, self.dfs[self.curr_df].get_df())


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
        self.tableView.setModel(self.df_models[self.curr_df])


