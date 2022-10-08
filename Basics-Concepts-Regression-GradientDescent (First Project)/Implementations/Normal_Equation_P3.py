import csv
import matplotlib.pyplot as plt
import numpy as np

#defining address of csv data file:
file_address = 'G:/AAA_Uni_Courses/Machine_Learning/HomeWorks/HW1/Dataset.csv'

#defining used list variables:
x=[]
y=[]
x_main=[]
List1 =[]
List2 =[]


#requesting an input as number of features:
m_features = int(input())


#reading csv file and importing data into x, y, x_main and List1 llists:
with open(file_address) as f:
    reader = csv.reader(f)
    for i in reader:
        if i[0] == 'x':
            continue
        for j in range(0,  m_features +1 ):
            List1.append((float(i[0]))** j )
        x.append(List1)
        y.append([float(i[1])])
        x_main.append([float(i[0])])
        List1=[]
#converting x and y list into numpy array for using matrix functions:        
xvec = np.asarray(x)
yvec = np.asarray(y)

#defining main equation of normal equation:
thetas = np.linalg.inv(np.dot(xvec.transpose() , xvec)).dot(xvec.transpose()).dot(yvec)
thetas = thetas[: : -1]
thetas = thetas.transpose()[0]

#plotting data alongwith normal equation result:
poly = np.poly1d(thetas)
y_predict = poly(x_main)
plt.plot(x_main , y_predict , 'r')
plt.scatter(x_main , y , 1)
plt.show()
