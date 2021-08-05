#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier

#Importing data by read only
'''
#Importing the data
for dirname, _, filenames in os.walk('The data directory'):
    for filename in filenames:
        os.path.join(dirname, filename)
'''

#The directory of the data
Dir = 'The data directory'
Classes = ['cats', 'dogs']

#The image size
img_size = 100

#If you want to check up you are doing good
'''
#Example to make sure we are doing good
for category in Classes:
    path = os.path.join(Dir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break
'''

#Training data
training_data = []
def creat_traingin_data():
    for category in Classes:
        path = os.path.join(Dir, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                new_arr = cv2.resize(img_arr, (img_size, img_size))
                training_data.append([new_arr, class_num])
            except Exception as e:
                pass
creat_traingin_data()

#Convirting the image to array, because we want the model to train it
the_len = len(training_data)
X = []
y = []
for categories, label in training_data:
    X.append(categories)
    y.append(label)
X = np.array(X).reshape(the_len, -1)
y = np.array(y)

#Flatting the array
out = np.concatenate(X).ravel()

#The train test split
X_train, X_test, y_train, y_test = train_test_split(X,y)

#The SVM
svm = SVC(kernel='poly', C=0.1, gamma=1)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

#This is the grid search for SVM (it's taking so long time to run)
'''
#The grid search
parameters = {'C':[0.1,10], 'gamma':[1, 0.1, 0.01]}
grid = GridSearchCV(svm, parameters)
grid.fit(X_train, y_train)
grid_pred = grid.predict(X_test)
'''

#The quality of the SVM model
cm = confusion_matrix(y_test, svm_pred)
rep = classification_report(y_test, svm_pred)

'''
#The KNN
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

#This is the grid search for KNN (it's taking so long time to run)

#The grid search
parameters = {'n_neighbors':[10, 100, 1000]}
grid = GridSearchCV(knn, parameters)
grid.fit(X_train, y_train)
grid_pred = grid.predict(X_test)


#The quality of the KNN model
cm = confusion_matrix(y_test, knn_pred)
rep = classification_report(y_test, knn_pred)
'''
