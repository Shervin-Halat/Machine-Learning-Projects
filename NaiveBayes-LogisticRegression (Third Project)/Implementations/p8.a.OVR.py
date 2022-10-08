from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from mlxtend.data import loadlocal_mnist


#warning handler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
#


#load train and test set
x , y = loadlocal_mnist(\
    images_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/train-images.idx3-ubyte', \
    labels_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/train-labels.idx1-ubyte')

x_test , y_test = loadlocal_mnist(\
    images_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/t10k-images.idx3-ubyte', \
    labels_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/t10k-labels.idx1-ubyte')

print(type(y))

logisticRegr = LogisticRegression\
               (  solver = 'lbfgs' , max_iter=50)

num_logreg = []
for i in range(10):
    num_logreg.append([])
   
for class_number in range(10):
    temp = np.array([] ,dtype = int)
    for i in y:
        if i == class_number:
            temp[class_number] = np.append(temp[class_number] , 1)
        else:
            temp[class_number] = np.append(temp[class_number] , 0)
    num_logreg[class_number] =  logisticRegr.fit(x , temp)


for i in range(5):
    print('real class is {}'.format(y_test[i]))
    for j in range(5):
        print(num_logreg[j].predict(x_test[i].reshape(1,-1)))
        print(num_logreg[j].predict_proba(x_test[i].reshape(1,-1)))


'''
def prediction(testset):
    for i in testset:
        



    
#logisticRegr = LogisticRegression\
#               (  solver = 'lbfgs' , multi_class = 'ovr' , max_iter=100)
logisticRegr.fit(x, y )
print('here')

plt.imshow(x_test[0].reshape(28,28) , cmap = 'gray')

predictions1 = logisticRegr.predict(x_test)
score1 = logisticRegr.score(x_test, y_test)
print(score1)

#predictions2 = logisticRegr.predict(x)
score2 = logisticRegr.score(x, y)
print(score2)

cm = metrics.confusion_matrix(y_test, predictions1)
print(cm)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", linewidths=0.5, \
            square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0:.2}'.format(score1)
plt.title(all_sample_title, size = 15);
plt.show()
'''
