import csv
import matplotlib.pyplot as plt
import numpy as np

file_address = 'G:/AAA_Uni_Courses/Machine_Learning/HomeWorks/HW1/Dataset.csv'
x=[]
y=[]
matrix = []
theta2a=[] #####
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
    theta2a=[]
    thetas = [theta] * (d+1)
    thetrev = thetas [: : -1]
    poly = np.poly1d(thetrev)
    h_th = poly(xvec)
    for i in range(0, iteration):
        h_th = poly(xvec)
        for j in range(0 , d+1):
            if j == 0:
                thetas[j] -= (alpha / n) * (float(sum((h_th - yvec) * xvec ** j)))
            elif j != 0:
                thetas[j] -= (alpha / n) * (float(sum((h_th - yvec) * xvec ** j)) + reg_par * thetas[j])
        theta2a.append(thetas[:]) #####
        thetrev = thetas [: : -1]
        poly = np.poly1d(thetrev)
    return(theta2a , thetas)  #####
    
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
N = 2 #int(input('give me number of inputs:'))
Input=[0,0.05]
#for i in range (0,N):
#    Input.append(float(input()))

thetaiteration = [] #####
for i in range(0,N):
    thetaiteration , thetas = Gradient_Descent(0.8, 4000, 0, 8, Input[i])
    thetaiteration = np.array(thetaiteration)
    for j in range (0, 9):
        plt.subplot(1,9, j+1)
        plt.title("Θ%i" %j)
        plt.plot(list(range(1,4001)), thetaiteration[: , j] )

    thetas = thetas[: : -1]                                          
    poly = np.poly1d(thetas)
    y_predict = poly(xvec)
    y = np.array(y)
    MSE_list=[]
    MSE = sum(((y - y_predict)**2))/n
    MSE_list.append(MSE)
    print("MSE %s :" %Input[i] ,MSE)
    '''plt.subplot(2 , N/2 , i+1)
    plt.scatter(x , y , 1)
    print('thetas =' , thetas)
    thetas = thetas[: : -1]                                          
    poly = np.poly1d(thetas)
    y_predict = poly(xvec)
    y = np.array(y)
    MSE = sum(((y - y_predict)**2))/n
    print('MSE:', MSE[0])
    plt.plot(xvec , y_predict , 'r'  )
    plt.title("MSE:%f" %MSE)'''

plt.show()

#learning rate = 0.8 iteration= 4000 theta=0  d=(8 khoobe//18 aalie!) reg_par= 0.05
