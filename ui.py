from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)

        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 800, 700))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())

        self.tabWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")

        self.tabTrain = QtWidgets.QWidget()
        self.tabTrain.setObjectName("tabTrain")

        self.datasetTable = QtWidgets.QTableView(self.tabTrain)
        self.datasetTable.setGeometry(QtCore.QRect(3, 50, 790, 390))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.datasetTable.setFont(font)
        self.datasetTable.setObjectName("datasetTable")

        self.label = QtWidgets.QLabel(self.tabTrain)
        self.label.setGeometry(QtCore.QRect(10, 15, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(8)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.loadButton = QtWidgets.QPushButton(self.tabTrain)
        self.loadButton.setGeometry(QtCore.QRect(650, 10, 131, 28))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.loadButton.setFont(font)
        self.loadButton.setObjectName("loadButton")

        self.trainButton = QtWidgets.QPushButton(self.tabTrain)
        self.trainButton.setGeometry(QtCore.QRect(335, 460, 130, 35))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.trainButton.setFont(font)
        self.trainButton.setObjectName("trainButton")

        self.progressText = QtWidgets.QTextBrowser(self.tabTrain)
        self.progressText.setGeometry(QtCore.QRect(140, 540, 520, 101))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.progressText.setFont(font)
        self.progressText.setObjectName("progressText")

        self.label_2 = QtWidgets.QLabel(self.tabTrain)
        self.label_2.setGeometry(QtCore.QRect(140, 510, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.tabWidget.addTab(self.tabTrain, "")
        self.testTab = QtWidgets.QWidget()
        self.testTab.setObjectName("testTab")

        self.handwritingImage = QtWidgets.QLabel(self.testTab)
        self.handwritingImage.setGeometry(QtCore.QRect(40, 30, 711, 251))
        self.handwritingImage.setObjectName("handwritingImage")
        self.handwritingImage.setStyleSheet("background-color: #DDDDDD;") 

        self.browseImageButton = QtWidgets.QPushButton(self.testTab)
        self.browseImageButton.setGeometry(QtCore.QRect(210, 300, 170, 41))
        self.browseImageButton.setObjectName("browseImageButton")

        self.analysisButton = QtWidgets.QPushButton(self.testTab)
        self.analysisButton.setGeometry(QtCore.QRect(420, 300, 170, 41))
        self.analysisButton.setObjectName("analysisButton")

        self.personalityText = QtWidgets.QTextBrowser(self.testTab)
        self.personalityText.setGeometry(QtCore.QRect(40, 450, 711, 141))
        self.personalityText.setObjectName("personalityText")

        self.label_3 = QtWidgets.QLabel(self.testTab)
        self.label_3.setGeometry(QtCore.QRect(42, 420, 91, 16))
        self.label_3.setObjectName("label_3")

        self.tabWidget.addTab(self.testTab, "")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.loadButton.clicked.connect(self.onLoadButtonClick)
        self.browseImageButton.clicked.connect(self.onBrowseImageClick)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("Pengenalan Kepribadian Berdasarkan Tulisan Tangan")
        self.label.setText("Dataset")
        self.loadButton.setText("Muat Dataset")
        self.trainButton.setText("Latih Dataset")
        self.label_2.setText("Progress")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTrain), "Pelatihan")
        self.browseImageButton.setText("Pilih Citra")
        self.analysisButton.setText("Analisis Kepribadian")
        self.label_3.setText("Kepribadian")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.testTab), "Pengujian")

    def onLoadButtonClick(self):
        dataset = pd.read_csv('sample_data.csv')
        model = DatasetModel(dataset)

        self.datasetTable.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.datasetTable.setModel(model)

    def onBrowseImageClick(self):
        dialog = QtWidgets.QFileDialog()
        filename = dialog.getOpenFileName(None, "Cari Gambar Tulisan Tangan", "", "Image File (*.png *.jpg *.jpeg)")
        if filename is not None:
            pixmap = QtGui.QPixmap(filename[0]).scaled(700, 240, QtCore.Qt.KeepAspectRatio)
            self.handwritingImage.setPixmap(pixmap)
            self.handwritingImage.setScaledContents(False)
            self.handwritingImage.setAlignment(QtCore.Qt.AlignCenter)

class DatasetModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
