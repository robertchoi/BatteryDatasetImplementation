import sys
import csv
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model


form_class = uic.loadUiType("wt0.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.btn_clicked)
        self.spinBox.valueChanged.connect(self.spinBoxChanged)

        self.tableWidget.setRowCount(100)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        #self.plot([1,2,3],[30,40,50])

    def getYvalue(self, minValue, maxValue):
        if minValue*2 > maxValue*1.2:
            return (minValue*2)
        else:
            return (maxValue*1.2)    

    def plot(self, timeVal, regVal):
        self.graphWidget.clear()
        self.graphWidget.setXRange(min(timeVal), max(timeVal), padding=0)
        self.graphWidget.setYRange(min(regVal), self.getYvalue(min(regVal), max(regVal)), padding=0)
        self.graphWidget.plot(timeVal, regVal)

    def btn_clicked(self):
        fname = QFileDialog.getOpenFileName(self)
        self.bank_label.setText(fname[0])   

        global df
        df = pd.read_csv(fname[0], encoding = 'utf8')
        maxcell = max(df['CellNo'])
        print("max cell = %d"  % maxcell)
        self.spinBox.setMaximum(maxcell)

        global cell_index
        cell_index = 1
        cell_index = df.index[df['CellNo']==cell_index].tolist()

        global df_cell
        df_cell = df.iloc[cell_index, :]
        print(df_cell)

        x_val = np.arange(len(df_cell))
        y_val = df_cell.ResistValue.to_numpy()
        t_val = df_cell.KeyTime.to_numpy()
        
        self.tableWidget.setRowCount(len(df_cell))
        for i in range(0, len(y_val)):
            s = '{0:0.3f}'.format(y_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,1,itemY)
            s = '{}'.format(t_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,0,itemY)      
        
        self.setTableWidgetData()
        #print(x_val)
        #print(y_val)
        self.plot(x_val,y_val)

        ptest_val = y_val[0:20]
        ptest_label = self.make_dataset(ptest_val, 20)

        print(ptest_label)
        model = keras.models.load_model("finalTest.h5")
        #prediction = model.predict(ptest_label, batch_size=16)
        #pred = scaler.inverse_transform(prediction[-1])
        #print(pred)
                

    def make_dataset(self, data, window_size=20):
        label_list = []
        for i in range(len(data) - window_size-19):
            label_list.append(np.array(label.iloc[i+window_size+20]))
        return np.array(label_list)


    def setTableWidgetData(self):
        column_headers = ['타임 스템프', '내부저항', '예측치']
        self.tableWidget.setHorizontalHeaderLabels(column_headers)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

    def spinBoxChanged(self):
        val = self.spinBox.value()
        print("cell number = %d" % val)
        cell_index = val
        cell_index = df.index[df['CellNo']==cell_index].tolist()
        df_cell = df.iloc[cell_index, :]
        print(df_cell)

        x_val = np.arange(len(df_cell))
        y_val = df_cell.ResistValue.to_numpy()
        t_val = df_cell.KeyTime.to_numpy()

        self.tableWidget.setRowCount(len(df_cell))
        for i in range(0, len(y_val)):
            s = '{0:0.3f}'.format(y_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,1,itemY)         
            s = '{}'.format(t_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,0,itemY)             
        #print(x_val)
        #print(y_val)
        self.plot(x_val,y_val)





if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()