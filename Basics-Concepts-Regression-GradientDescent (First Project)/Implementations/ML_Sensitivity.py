import csv
import matplotlib.pyplot as plt
import numpy as np

file_address = 'G:/AAA_Uni_Courses/Machine_Learning/HomeWorks/HW1/Dataset.csv'
x=[]
y=[]
matrix = []
with open(file_address) as f:
    reader = csv.reader(f)
    for i in reader:
        if i[0] == 'x':
            continue
        x.append([float(i[0])])
        y.append([float(i[1])])
        matrix.append(i)
        
n = len(x)
xvec = np.asarray(x)
yvec = np.asarray(y)
def Gradient_Descent( alpha , iteration , theta , d , reg_par ):
    thetas = [theta] * (d+1)
    thetrev = thetas [: : -1]
    poly = np.poly1d(thetrev)
    

    h_th = poly(xvec)
    for i in range(0, iteration):
        h_th = poly(xvec)
        for i in range(0 , d+1):
            if i==0 :
                thetas[i] -= (alpha / n) * (float(sum((h_th - yvec) * xvec ** i)))
            elif i != 0 :
                thetas[i] -= (alpha / n) * (float(sum((h_th - yvec) * xvec ** i)) + reg_par * thetas[i])
        thetrev = thetas [: : -1]
        poly = np.poly1d(thetrev)
    return(thetas)
    
'''
print('Just give me requested variables and will give you best possible fit!')
Learning_Rate, Iteration, Theta_Initial, Dimension, Regularization_Parameter =\
                                        float(input('give me Learning_Rate:'))\
                                        ,int(input('give me Iteration:'))\
                                        ,int(input('give me Theta_Initial:'))\
                                        ,int(input('give me Dimension:'))\
                                        ,float(input('give me Regularization_Parameter:'))
                                                                                      
thetas = Gradient_Descent(Learning_Rate , Iteration , Theta_Initial ,Dimension , Regularization_Parameter)
'''
N = int(input('give me number of inputs:'))
Input=[]
for i in range (0,N):
    Input.append(int(input()))
    
for i in range(0,N):
    thetas = Gradient_Descent(0.8, 6000, 0, 8, Input[i])
    plt.subplot(1 , N , i+1)
    plt.scatter(x , y , 1)
    print('thetas =' , thetas)
    thetas = thetas[: : -1]                                          
    poly = np.poly1d(thetas)
    y_predict = poly(xvec)
    y = np.array(y)
    MSE = sum(((y - y_predict)**2))/n
    print('MSE:', MSE[0])
    plt.plot(xvec , y_predict , 'r'  )
    plt.title("MSE:%.2f \n Dim:%i" %(MSE,Input[i]))
plt.show()

#learning rate = 0.8 iteration= 3000 theta=0  d=(8 khoobe//18 aalie!) reg_par= 0.05

