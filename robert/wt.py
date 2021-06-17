import sys
import csv
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


form_class = uic.loadUiType("wt0.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.btn_clicked)
        self.spinBox.valueChanged.connect(self.spinBoxChanged)

        self.tableWidget.setRowCount(100)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.tableWidget.cellDoubleClicked.connect(self.table_DoubleClicked)


        self.tableWidget_2.setRowCount(1)
        self.tableWidget_2.setColumnCount(20)
        self.tableWidget_2.setEditTriggers(QAbstractItemView.NoEditTriggers)

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

    
        sc1 = MinMaxScaler()
        scale_cols = ['ResistValue']
        df_scaled = sc1.fit_transform(df_cell[scale_cols])
        df_scaled = pd.DataFrame(df_scaled)
        df_scaled.columns = scale_cols
        print("df_scaled")
        print(df_scaled)
        scaled_val = df_scaled.ResistValue.to_numpy()


        global s_val
        s_val = df_scaled.ResistValue
        
        self.tableWidget.setRowCount(len(df_cell))
        for i in range(0, len(y_val)):
            s = '{0:0.3f}'.format(scaled_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,2,itemY)     # scaled_val

            s = '{0:0.3f}'.format(y_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,1,itemY)     # ResistValue
            s = '{}'.format(t_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,0,itemY)    # time stamp


        for i in range(0, 10):
            s = "0"
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget_2.setItem(0,i,itemY)      
        
        self.setTableWidgetData()
        #print(x_val)
        #print(y_val)
        self.plot(x_val,y_val)



    def predict_proc(self, st=19):
        ptest_label = self.make_dataset(s_val, 20, st)
        ptest_label = np.array(ptest_label).reshape(20,20,1)
        print(ptest_label)
        #MODEL_PATH = "F:\\github\\BatteryDatasetImplementation\\robert\\model_d10.h5"
        global model
        MODEL_PATH = "model_d10.h5"
        model = load_model(MODEL_PATH, compile = False)
        prediction = model.predict(ptest_label)
        global scaler
        scaler = MinMaxScaler()
        scaler.fit(np.array(ptest_label[0]))
        pred = scaler.inverse_transform(prediction[-1])
        print(pred)
        print(type(pred))

        for i in range(0, len(pred)):
            s = '{0:0.3f}'.format(pred[i,0])
            #print(s)
            itemY = QTableWidgetItem(s)
            if i+st+1 < len(df_cell):
                self.tableWidget.setItem(i+st+1,3,itemY)

        for i in range(0, 20):
            s = '{0:0.3f}'.format(pred[i,0])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget_2.setItem(0,i,itemY)  




                
    def table_DoubleClicked(self):
        select_val = self.tableWidget.currentRow();
        print(select_val)
        self.tableWidget.selectRow(select_val)
        self.predict_proc(select_val)   


    def make_dataset(self, data, window_size=20, selectTime=19):
        label_list = []
        for i in range(20):
            label_list.append(np.array(data.iloc[selectTime-19+i:selectTime-19+i+window_size]))
        return np.array(label_list)


    def setTableWidgetData(self):
        column_headers = ['타임 스템프', '내부저항', 'scaled', '예측치']
        self.tableWidget.setHorizontalHeaderLabels(column_headers)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        self.tableWidget_2.resizeRowsToContents()

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

        sc1 = MinMaxScaler()
        scale_cols = ['ResistValue']
        df_scaled = sc1.fit_transform(df_cell[scale_cols])
        df_scaled = pd.DataFrame(df_scaled)
        df_scaled.columns = scale_cols
        print("df_scaled")
        print(df_scaled)
        scaled_val = df_scaled.ResistValue.to_numpy()

        s_val = df_scaled.ResistValue        

        self.tableWidget.setRowCount(len(df_cell))
        for i in range(0, len(y_val)):
            s = '{0:0.3f}'.format(scaled_val[i])
            #print(s)
            itemY = QTableWidgetItem(s)
            self.tableWidget.setItem(i,2,itemY)     # scaled_val

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