#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:39:06 2017

@author: zaheerbabar
"""
import pandas as pd

import numpy as np


words=['molecular']

Data=pd.read_csv("DataFile.csv")

DataClusterNr=Data[Data['UNK']=='molecular']['clusterNr']

DataClusterNr=np.array(DataClusterNr)
DataCluster=Data[Data['clusterNr']==DataClusterNr[0]]['UNK']

Data1=pd.read_csv("DataFile1.csv")

DataClusterNr=Data1[Data1['UNK']=='molecular']['clusterNr']

DataClusterNr=np.array(DataClusterNr)
DataCluster1=Data1[Data1['clusterNr']==DataClusterNr[0]]['UNK']

Data2=pd.read_csv("DataFile2.csv")

DataClusterNr=Data2[Data2['UNK']=='molecular']['clusterNr']

DataClusterNr=np.array(DataClusterNr)
DataCluster2=Data2[Data2['clusterNr']==DataClusterNr[0]]['UNK']

Data3=pd.read_csv("DataFile3.csv")

DataClusterNr=Data3[Data3['UNK']=='molecular']['clusterNr']

DataClusterNr=np.array(DataClusterNr)
DataCluster3=Data3[Data3['clusterNr']==DataClusterNr[0]]['UNK']

Data4=pd.read_csv("DataFile4.csv")

DataClusterNr=Data4[Data4['UNK']=='molecular']['clusterNr']

DataClusterNr=np.array(DataClusterNr)
DataCluster4=Data4[Data4['clusterNr']==DataClusterNr[0]]['UNK']

Data5=pd.read_csv("DataFile5.csv")

DataClusterNr=Data5[Data5['UNK']=='molecular']['clusterNr']

DataClusterNr=np.array(DataClusterNr)
DataCluster5=Data5[Data5['clusterNr']==DataClusterNr[0]]['UNK']

orignal_cluster=DataCluster

compare_Cluster=DataCluster5


print("From Original Embeddings to First Index missing- intersection: ", set(orignal_cluster).intersection(set(compare_Cluster)))

print()

print("Words those had company with",words, "in original embeddings cluster but not in one index missing cluster:", set(orignal_cluster).difference(set(compare_Cluster)))

print()
print("Words that got company with",words, "after one index missing:", set(compare_Cluster).difference(set(orignal_cluster)))


