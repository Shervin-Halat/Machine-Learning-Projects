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
    images_path='D:/AAA_Uni_Courses/1_TERM/Machine_Learning/HomeWorks/HW3/98131018/Dataset/train-images.idx3-ubyte', \
    labels_path='D:/AAA_Uni_Courses/1_TERM/Machine_Learning/HomeWorks/HW3/98131018/Dataset/train-labels.idx1-ubyte')

x_test , y_test = loadlocal_mnist(\
    images_path='D:/AAA_Uni_Courses/1_TERM/Machine_Learning/HomeWorks/HW3/98131018/Dataset/t10k-images.idx3-ubyte', \
    labels_path='D:/AAA_Uni_Courses/1_TERM/Machine_Learning/HomeWorks/HW3/98131018/Dataset/t10k-labels.idx1-ubyte')
print(type(y))

logisticRegr = LogisticRegression\
               (  solver = 'lbfgs' , multi_class = 'ovr' , max_iter=100)
logisticRegr.fit(x, y )
print('here')

plt.imshow(x_test[0].reshape(28,28) , cmap = 'gray')

predictions1 = logisticRegr.predict(x_test)
print(predictions1)
print(type(predictions1))
print(predictions1.shape)

score1 = logisticRegr.score(x_test, y_test)
print(score1)
print(type(score1))

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
