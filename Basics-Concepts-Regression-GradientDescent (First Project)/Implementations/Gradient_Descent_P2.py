import csv
import matplotlib.pyplot as plt
import numpy as np

#File address:
file_address = 'G:/AAA_Uni_Courses/Machine_Learning/HomeWorks/HW1/Dataset.csv'
x=[]
y=[]

#Reading data and trasfer to lists x, y using CSV:
with open(file_address) as f:
    reader = csv.reader(f)
    for i in reader:
        if i[0] == 'x':
            continue
        x.append([float(i[0])])
        y.append([float(i[1])])

#Defining Gradient_Descent algorithm as a function with requested inputs\
#        which returns theta vector:
def Gradient_Descent( alpha , iteration , theta , d , reg_par ):
    thetas = [theta] * (d+1)
    thetrev = thetas [: : -1]
    poly = np.poly1d(thetrev)
    h_th = poly(xvec)
    for i in range(0, iteration):
        h_th = poly(xvec)
        for i in range(0 , d+1):
            thetas[i] -= (alpha / n) * (float(sum((h_th - yvec) * xvec ** i)) + reg_par * thetas[i])
        thetrev = thetas [: : -1]
        poly = np.poly1d(thetrev)
    return(thetas)
    
#Requesting inputs:
print('Just give me requested variables and I will give you best possible fit!')
Learning_Rate, Iteration, Theta_Initial, Dimension, Regularization_Parameter =\
                                        float(input('give me Learning_Rate:'))\
                                        ,int(input('give me Iteration:'))\
                                        ,int(input('give me Theta_Initial:'))\
                                        ,int(input('give me Dimension:'))\
                                        ,float(input('give me Regularization_Parameter:'))

#Main part of code:
#using inputs with defined function:
thetas = Gradient_Descent(Learning_Rate , Iteration , Theta_Initial ,Dimension , Regularization_Parameter)

#some essential defenitions:
plt.scatter(x,y , 1)
print('thetas =' , thetas)
thetas = thetas[: : -1]                                          
poly = np.poly1d(thetas)
y_predict = poly(xvec)
y = np.array(y)

#defining MSE and printing it:
MSE = sum(((y - y_predict)**2))/n
print('MSE:', MSE[0])

#ploting linear regression alongwith data with a title of MSE:
plt.plot(xvec , y_predict , 'r')
plt.title("MSE:%f" %MSE)
plt.show()

