from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from mlxtend.data import loadlocal_mnist

'''
#warning handler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
#
'''

#load train and test set
x , y = loadlocal_mnist(\
    images_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/train-images.idx3-ubyte', \
    labels_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/train-labels.idx1-ubyte')

x_test , y_test = loadlocal_mnist(\
    images_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/t10k-images.idx3-ubyte', \
    labels_path='C:/Users/sherw/OneDrive/Desktop/ML3/Dataset/t10k-labels.idx1-ubyte')

print(type(y))

logisticRegr = LogisticRegression\
               (  solver = 'lbfgs')
logisticRegr.fit(x, y )
print('here')

plt.imshow(x_test[0].reshape(28,28) , cmap = 'gray')

predictions1 = logisticRegr.predict(x_test)
score1 = logisticRegr.score(x_test, y_test)
print(score1)

fig = plt.figure(1)
for i in range(25):
    p = plt.subplot2grid((5,5),(i//5,i%5))
    plt.imshow(x_test[i].reshape(28,28),cmap='gray')
    plt.title('y_pre {}/y_org {}'.format(logisticRegr.predict\
                                         (x_test[i].reshape(1,-1)) , y_test[i]))
    p.axes.get_xaxis().set_visible(False)
    p.axes.get_yaxis().set_visible(False)
    
fig.suptitle("25 test samples")
plt.show()

#predictions2 = logisticRegr.predict(x)
#score2 = logisticRegr.score(x, y)
#print(score2)

'''
cm = metrics.confusion_matrix(y_test, predictions1)
print(cm)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", linewidths=0.5, \
            square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0:.2}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()
'''
