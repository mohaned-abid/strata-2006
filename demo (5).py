import numpy as np
import pandas as pd

#importing data
train=pd.read_csv('D:/Mohanned 10/Downloads/work/train.csv')
test=pd.read_csv('D:/Mohanned 10/Downloads/work/test.csv')
realpred=pd.read_csv('D:/Mohanned 10/Downloads/work/realpred.csv')
test=test.iloc[:99,:]

#some  cleaning
for i in range(100):
    train.iloc[:,i].fillna(0,inplace=True)
    test.iloc[:,i].fillna(0,inplace=True)

#splitting 
trainy=train.iloc[:,-1] 
trainx=train.drop(train.columns[100],axis=1)
testx=test

#importing  your models
#randomforest classifier
from sklearn.ensemble import RandomForestClassifier
#svm classifier
from sklearn.svm import SVC
#k-nearest
from sklearn.neighbors import KNeighborsClassifier
#logistic regression
from sklearn.linear_model import LogisticRegression

#instancing our models
model1=RandomForestClassifier(n_estimators=200, bootstrap = True,max_features = 'sqrt')
model2=SVC()
model3=KNeighborsClassifier(n_neighbors=3)
model4= LogisticRegression(random_state=0,solver='liblinear' ,multi_class='ovr')

#fitting this shit
model1.fit(trainx,trainy)
model2.fit(trainx,trainy)
model3.fit(trainx,trainy)
model4.fit(trainx,trainy)

#predicting
y1=model1.predict(testx)
y2=model2.predict(testx)
y3=model3.predict(testx)
y4=model4.predict(testx)


#getting the accuracy
from sklearn.metrics import accuracy_score
k1=accuracy_score(realpred,y1)
k2=accuracy_score(realpred,y2)
k3=accuracy_score(realpred,y3)
k4=accuracy_score(realpred,y4)

print(k1,k2,k3,k4)


#0.8686868686868687 0.7777777777777778 0.6767676767676768 0.7777777777777778
#0.8888888888888888 0.8181818181818182 0.6767676767676768 0.8181818181818182
#0.898989898989899 0.8181818181818182 0.6060606060606061 0.8181818181818182
