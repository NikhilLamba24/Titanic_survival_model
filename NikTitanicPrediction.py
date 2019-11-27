import pandas as pd
import sklearn
import numpy as np
test = pd.read_csv("./test.csv")
train = pd.read_csv("./train.csv")
test1 = pd.read_csv("./gender_submission.csv")
#print(X)
#from sklearn.cross_validation import train_test_split
#X_train
  #      print(X_train[:,4])
   #     X_imp = X_train[:,4].reshape(1,891)
    #    print(X_imp)
    
#to remove nan values
train.Age.fillna(train.Age.mean(),inplace=True)  
print(train.Age)    
train.isnull().sum()   
 
test.Age.fillna(test.Age.mean(),inplace=True)  
print(test.Age)    
test.isnull().sum()

#for Nan values Error       
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'Nan', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X_train)
#X_train = imputer.transform(X_train)


#onehotencoder = OneHotEncoder(categorical_features =train['Sex'])
#train['Sex']= onehotencoder.fit_transform(train['Sex']).toarray().reshape(-1,1)
#train['Sex']=train['Sex'].reshape(-1,1)
#labelencoder_Y=LabelEncoder()
#Y = labelencoder_Y.fit_transform(Y)

X_train = train.iloc[:,[0,2,4,5,6,7]].values
Y_train = train.iloc[:,1].values

X_test = test.iloc[:,[0,1,3,4,5,6]].values
Y_test = test1.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
test['Sex'] = labelencoder_Y.fit_transform(test['Sex'])

from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
train['Sex'] = labelencoder_Y.fit_transform(train['Sex'])

#x=X_train.reshape(418,5)
#By Knn 
#from sklearn.neighbors import KNeighborsClassifier
#knn= KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
ans = knn.predict(X_test)
ans
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')
#x = X_train.reshape(-1,1)
logreg.fit(X_train, Y_train)
ans = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
a = accuracy_score(Y_test,ans)
a



submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": ans
        })
submission.to_csv("./MNIST-1/submission.csv", index=False)