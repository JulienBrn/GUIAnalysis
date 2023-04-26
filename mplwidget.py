from PyQt5 import QtGui
import  PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
  def __init__(self):
    self.fig = Figure()
    self.ax = self.fig.add_subplot(111)
    FigureCanvas.__init__(self, self.fig)
    # FigureCanvas.setSizePolicy(self,
    # QtGui.QSizePolicy.Expanding,
    # QtGui.QSizePolicy.Expanding)
    FigureCanvas.updateGeometry(self)

class MplWidget(PyQt5.QtWidgets.QWidget):
  def __init__(self, parent = None):
    PyQt5.QtWidgets.QWidget.__init__(self, parent)
    self.canvas = MplCanvas()
    self.vbl = PyQt5.QtWidgets.QVBoxLayout()
    self.vbl.addWidget(self.canvas)
    self.setLayout(self.vbl)