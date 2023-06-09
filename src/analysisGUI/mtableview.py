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




class MTableView(QTableView):
    def __init__(self, parent):
        super().__init__(parent)

    def contextMenuEvent(self, event):
      self.menu = QMenu(self)
      renameAction = QAction('Rename', self)
      renameAction.triggered.connect(lambda: self.renameSlot(event))
      self.menu.addAction(renameAction)
      # add other required actions
      self.menu.popup(QCursor.pos())
      

    def renameSlot(self, event):
      print(event)
      # get the selected row and column
      row = self.rowAt(event.pos().y())
      col = self.columnAt(event.pos().x())
      # get the selected cell
      cell = self.model()._dataframe.iloc[row, col]
      # get the text inside selected cell (if any)
      # get the widget inside selected cell (if any)
      # widget = self.tableWidget.cellWidget(row, col)
      logger.info("Rename action on cell {} called".format(cell))

