import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()


#### load data
df_cancer=pd.DataFramedf_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

"""### to find relation between
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])


### to see the target graphe
sns.countplot(df_cancer["target"])

## to see correlation between our features
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot=True)"""


#### importing input and output variables
X=df_cancer.drop(["target"],axis=1)
y=df_cancer["target"]


### Splitting data in training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

### implementing machine on dataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
svc_model=SVC()
svc_model.fit(X_train,y_train)

### evaluate the model
"""y_pred=svc_model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)"""

####m improvig model by normalisation
min_train=X_train.min()
range_train=(X_train-min_train).max()
X_train_scaled=(X_train-min_train)/range_train

min_test=X_test.min()
range_test=(X_test-min_test).max()
X_test_scaled=(X_test-min_test)/range_test

svc_model.fit(X_train_scaled,y_train)
y_predict=svc_model.predict(X_test_scaled)
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)


print(classification_report(y_test,y_predict))


###we will do grid search to improve lot
param_grid={'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(),param_grid, refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

grid_prediction=grid.predict(X_test_scaled)
cm=confusion_matrix(y_test,grid_prediction)
   
