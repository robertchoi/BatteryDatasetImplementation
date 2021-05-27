#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def createFolder(directory):
    #저장 폴더 생성
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory' + directory)


# In[3]:


def changePath(directory):
    #데이터 폴더 선택
    currentPath = os.getcwd()
    change_path = currentPath+directory
    try:
        if os.path.exists(directory[1:]):
            os.chdir(change_path)
    except OSError:
        print('Error: Changing Path ' + directory)


# In[4]:


def SavePlotPNG():
    file_list = os.listdir(os.getcwd())
    #file_list.remove('cell')
    file_count = len(file_list)
    createFolder('plotpng')
    current_path = os.getcwd()
    
    for i in range(file_count):
        
        file_path = file_list[i]
        delete_csv_path = file_path[:-4]
        df = pd.read_csv(file_path, encoding = 'utf8')
        
        os.chdir(current_path + '\plotpng')
        plt.figure(figsize=(12,6))
        plt.plot(df.loc[:,'Resistance'], label="Resistance")
        plt.legend()
        plt.savefig(delete_csv_path + '.png')
        plt.show()
        os.chdir(current_path)


# In[5]:


if __name__ =='__main__':
    print('데이터 폴더 입력(\ 포함)')
    data_folder_path = input()


# In[6]:


#데이터 폴더 입력
changePath(data_folder_path)


# In[7]:


os.getcwd()


# In[8]:


column_list = ['MeasuredDate', 'Resistance', 'Volt', 'Temp']


# In[9]:


SavePlotPNG()


# In[ ]:




