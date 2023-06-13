from __future__ import annotations
from typing import List, Tuple, Dict, Any
from toolbox import json_loader, RessourceHandle, Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
from importlib.resources import files as package_data
import os

logger=logging.getLogger(__name__)

import sys, time
import matplotlib, importlib
import matplotlib.pyplot as plt


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie, QCursor
from PyQt5.QtCore import pyqtSlot, QItemSelectionModel, QModelIndex
import toolbox



class MTableView(QTableView):
    def __init__(self, parent):
        super().__init__(parent)

    def contextMenuEvent(self, event):
        selection = [(i.row(), i.column()) for i in self.selectionModel().selection().indexes()]
        self.menu = QMenu(self)
        computeAction = QAction('Compute', self)
        computeAction.triggered.connect(lambda: self.computeSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        self.menu.addAction(computeAction)
        # add other required actions
        self.menu.popup(QCursor.pos())
      

    def computeSlot(self, selec, df):
      from analysisGUI.gui import Task
      win = self.window()
      items = [df.iloc[i.row(), i.column()] for i in selec if isinstance(df.iloc[i.row(), i.column()], RessourceHandle)]
      def run(task_info):
          for item in task_info["progress"](items):
              item.get_result()
      task = Task(win, "compute", lambda task_info: True, lambda task_info: self.model().dataChanged.emit(selec[0], selec[-1]), run, {})
      win.add_task(task)
    #   self.parent.add_task()
    #   for ind in selec:
    #       item = df.iloc[ind.row(), ind.column()]
    #       if is
    #   item.get_result()
    #   self.model().dataChanged.emit(updatel[0])

