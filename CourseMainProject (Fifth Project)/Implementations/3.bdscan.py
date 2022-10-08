import pandas
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import csv
from random import randint

loc = 'C:/Users/sherw/OneDrive/Desktop/ML_Project/Dataset/Clustering/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls'
data = pandas.read_excel(loc,'Training_Data')
data = np.array([ data[column].values for column in data.columns]).T
data = np.array(data[:,0:5],dtype = float)
print(data.shape)

c = []
ep = 0.2
ms = 5
db = DBSCAN(eps = ep, min_samples = ms)
db = db.fit(data)
p = db.labels_

with open( 'C:/Users/sherw/OneDrive/Desktop/ML_Project/3.dbscan.csv', 'w',newline = '') as h:
    writer = csv.writer(h)
    for i in p:
        writer.writerow([int(i)])



        
    
