from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import zones, pressure
import feature_extract
import rsvm
import collections
import numpy as np

class Ui_MainWindow(object):
    datasetHeader = pd.DataFrame({'Nama File':[],'Rerata':[],'Persentase':[],'Zona Atas':[],'Zona Tengah':[],'Zona Bawah':[],'Tekanan Tulisan':[],'Dominasi Zona':[]})
    dataset = {}
    rsvmModel = []
    imageToPredic = ""

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
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")

        self.tabTrain = QtWidgets.QWidget()
        self.tabTrain.setObjectName("tabTrain")

        self.datasetTable = QtWidgets.QTableView(self.tabTrain)
        self.datasetTable.setGeometry(QtCore.QRect(3, 50, 790, 390))
        self.datasetTable.setFont(font)
        self.datasetTable.setObjectName("datasetTable")

        model = DatasetModel(self.datasetHeader)
        self.datasetTable.setModel(model)

        self.label = QtWidgets.QLabel(self.tabTrain)
        self.label.setGeometry(QtCore.QRect(10, 15, 100, 16))
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.loadByCSVButton = QtWidgets.QPushButton(self.tabTrain)
        self.loadByCSVButton.setGeometry(QtCore.QRect(615, 10, 170, 30))
        self.loadByCSVButton.setFont(font)
        self.loadByCSVButton.setObjectName("loadButton")

        self.loadByFolderButton = QtWidgets.QPushButton(self.tabTrain)
        self.loadByFolderButton.setGeometry(QtCore.QRect(435, 10, 170, 30))
        self.loadByFolderButton.setFont(font)
        self.loadByFolderButton.setObjectName("loadByFolderButton")

        self.trainButton = QtWidgets.QPushButton(self.tabTrain)
        self.trainButton.setGeometry(QtCore.QRect(250, 460, 130, 35))
        self.trainButton.setFont(font)
        self.trainButton.setObjectName("trainButton")
        self.trainButton.setEnabled(False)

        self.saveDatasetButton = QtWidgets.QPushButton(self.tabTrain)
        self.saveDatasetButton.setGeometry(QtCore.QRect(400, 460, 130, 35))
        self.saveDatasetButton.setFont(font)
        self.saveDatasetButton.setObjectName("trainButton")
        self.saveDatasetButton.setEnabled(False)

        self.progressText = QtWidgets.QTextBrowser(self.tabTrain)
        self.progressText.setGeometry(QtCore.QRect(10, 530, 780, 110))
        self.progressText.setFont(font)
        self.progressText.setObjectName("progressText")

        self.label_2 = QtWidgets.QLabel(self.tabTrain)
        self.label_2.setGeometry(QtCore.QRect(10, 510, 100, 16))
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
        self.analysisButton.setEnabled(False)

        self.featureImageTable = QtWidgets.QTableView(self.testTab)
        self.featureImageTable.setGeometry(QtCore.QRect(145, 370, 510, 65))
        self.featureImageTable.setFont(font)
        self.featureImageTable.setObjectName("featureImageTable")

        model = QtGui.QStandardItemModel()
        model.setHorizontalHeaderLabels(self.datasetHeader.columns.values[1:-2])
        self.featureImageTable.setModel(model)

        self.personalityText = QtWidgets.QTextBrowser(self.testTab)
        self.personalityText.setGeometry(QtCore.QRect(40, 480, 711, 141))
        self.personalityText.setObjectName("personalityText")

        self.label_3 = QtWidgets.QLabel(self.testTab)
        self.label_3.setGeometry(QtCore.QRect(42, 450, 91, 16))
        self.label_3.setObjectName("label_3")

        self.tabWidget.addTab(self.testTab, "")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.loadByFolderButton.clicked.connect(self.onLoadFolderClick)
        self.loadByCSVButton.clicked.connect(self.onLoadCSVClick)
        self.trainButton.clicked.connect(self.onTrainButtonClick)
        self.saveDatasetButton.clicked.connect(self.onSaveDatasetButtonClick)
        self.browseImageButton.clicked.connect(self.onBrowseImageClick)
        self.analysisButton.clicked.connect(self.onAnalysisButtonClick)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("Pengenalan Kepribadian Berdasarkan Tulisan Tangan")
        self.label.setText("Dataset")
        self.loadByCSVButton.setText("Muat Dataset Dari CSV")
        self.loadByFolderButton.setText("Muat Dataset Dari Folder")
        self.trainButton.setText("Latih Dataset")
        self.saveDatasetButton.setText("Simpan Dataset")
        self.label_2.setText("Hasil Pelatihan")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTrain), "Pelatihan")
        self.browseImageButton.setText("Pilih Citra")
        self.analysisButton.setText("Analisis Kepribadian")
        self.label_3.setText("Kepribadian")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.testTab), "Pengujian")

    def onLoadFolderClick(self):
        dialog = QtWidgets.QFileDialog()
        folderName = dialog.getExistingDirectory(None, "Pilih Folder Dataset")

        if folderName != '':
            dataset = feature_extract.extract(folderName)
            if dataset.shape[0] > 0:
                self.showDataset(dataset=dataset)
            else:
                QtWidgets.QMessageBox.information(None, "Informasi", "Dataset Kosong!")

    def onLoadCSVClick(self):
        dialog = QtWidgets.QFileDialog()
        filename = dialog.getOpenFileName(None, "Pilih Dataset CSV", "", "CSV File (*.csv)")
        if filename[0] != '':
            dataset = pd.read_csv(filename[0])
            try:
                if (self.datasetHeader.columns.values == dataset.columns.values).all():
                    if dataset.shape[0] > 0:
                        self.showDataset(dataset=dataset)
                    else:
                        QtWidgets.QMessageBox.information(None, "Informasi", "Dataset Kosong!")
                    self.showDataset(dataset)
                else:
                    QtWidgets.QMessageBox.warning(None, "Terjadi Kesalahan", "Harap masukan dataset yang valid untuk menjalankan proses pelatihan data!")
            except:
                QtWidgets.QMessageBox.warning(None, "Terjadi Kesalahan", "Harap masukan dataset yang valid untuk menjalankan proses pelatihan data!")

    def showDataset(self, dataset):
        self.dataset = dataset
        model = DatasetModel(dataset)
        self.datasetTable.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.datasetTable.setModel(model)

        self.label.setText("Dataset ("+str(dataset.shape[0])+")")

        if dataset.shape[0] > 0:
            self.trainButton.setEnabled(True)
            self.saveDatasetButton.setEnabled(True)
        else:
            self.trainButton.setEnabled(False)
            self.saveDatasetButton.setEnabled(False)

    def onTrainButtonClick(self):
        model_zone = rsvm.train_zone(self.dataset)
        model_pressure = rsvm.train_pressure(self.dataset)
        self.rsvmModel = model_zone + model_pressure

        trainAccuracy = round(((model_zone[3][0] + model_pressure[3][0]) / 2) * 100,2)
        testAccuracy = round(((model_zone[4][0] + model_pressure[4][0]) / 2) * 100,2)

        modelText = "Akurasi Model : <b>"+str(trainAccuracy)+"%</b><br>Akurasi Pengujian : <b>"+str(testAccuracy)+"%</b>"
        modelText += "<br><br><b>Model Zona Atas-Tengah:</b><br><b style='text-decoration: overline;'>u</b> : "+str(model_zone[0].get('w'))+"<br><b>&gamma; :</b> "+str(model_zone[0].get('b'))
        modelText += "<br><br><b>Model Zona Atas-Bawah:</b><br><b style='text-decoration: overline;'>u</b> : "+str(model_zone[1].get('w'))+"<br><b>&gamma; :</b> "+str(model_zone[1].get('b'))
        modelText += "<br><br><b>Model Zona Tengah-Bawah:</b><br><b style='text-decoration: overline;'>u</b> : "+str(model_zone[2].get('w'))+"<br><b>&gamma; :</b> "+str(model_zone[2].get('b'))
        
        modelText += "<br><hr><br><b>Model Tekanan Kuat-Sedang:</b><br><b style='text-decoration: overline;'>u</b> : "+str(model_pressure[1].get('w'))+"<br><b>&gamma; :</b> "+str(model_pressure[1].get('b'))
        modelText += "<br><br><b>Model Tekanan Kuat-Ringan:</b><br><b style='text-decoration: overline;'>u</b> : "+str(model_pressure[2].get('w'))+"<br><b>&gamma; :</b> "+str(model_pressure[2].get('b'))
        modelText += "<br><br><b>Model Tekanan Sedang-Ringan:</b><br><b style='text-decoration: overline;'>u</b> : "+str(model_pressure[1].get('w'))+"<br><b>&gamma; :</b> "+str(model_pressure[1].get('b'))
       
        self.progressText.setHtml(modelText)

    def onSaveDatasetButtonClick(self):
        fileName = QtWidgets.QFileDialog.getSaveFileName(None, "Simpan Dataset", "", "CSV File (*.csv")
        if fileName != '':
            self.dataset.to_csv(fileName[0], index=False)
            QtWidgets.QMessageBox.information(None, "Informasi", "Dataset berhasil disimpan")

    def onBrowseImageClick(self):
        dialog = QtWidgets.QFileDialog()
        filename = dialog.getOpenFileName(None, "Pilih Gambar Tulisan Tangan", "", "Image File (*.png *.jpg *.jpeg)")
        if filename[0] != '':
            self.imageToPredic = filename[0]
            pixmap = QtGui.QPixmap(filename[0]).scaled(700, 240, QtCore.Qt.KeepAspectRatio)
            self.handwritingImage.setPixmap(pixmap)
            self.handwritingImage.setScaledContents(False)
            self.handwritingImage.setAlignment(QtCore.Qt.AlignCenter)

            self.analysisButton.setEnabled(True)

    def onAnalysisButtonClick(self):
        if not self.rsvmModel:
            QtWidgets.QMessageBox.warning(None, "Terjadi Kesalahan", "Silahkan lakukan proses pelatihan terlebih dahulu untuk melakukan pengenalan kepribadian!")
        else:
            x = np.array([zones.extract(self.imageToPredic)])
            y = np.array([pressure.extract(self.imageToPredic)])
            
            predict_zone = rsvm.predict_zone(x)
            result_zone = rsvm.result_zone(predict_zone[0])

            predict_pressure = rsvm.predict_pressure(y)
            result_pressure = rsvm.result_pressure(predict_pressure[0])

            result = "<big>"+result_zone+"<br><br>"+result_pressure+"</big>"
            self.personalityText.setHtml(result)

            extractData = pd.DataFrame({'Rerata':[y[0][0]],'Persentase':[y[0][1]],'Zona Atas':[x[0][0]],'Zona Tengah':[x[0][1]],'Zona Bawah':[x[0][2]]})
            model = DatasetModel(extractData)
            self.featureImageTable.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
            self.featureImageTable.setModel(model)
        

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
