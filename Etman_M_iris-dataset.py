############# MY FIRST ML MODEL ON IRIS DATASET ##############
#Loading the needed libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# Scatter Plot Matrix
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pickle

#dataset url 
names = ['sepal_length','sepal_width','petal_length','petal_width',"species"]
path=r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
data = pd.read_csv(path,names=names) 
array = data.values

#data=data.drop(0)

#The code below Review the dimensions of iris dataset
print(data.shape)
print(data)
# data types for each attribute
print(data.dtypes)

# The code below summarizes the distribution of each attribut
description = data.describe()
print(description)
scatter_matrix(data)
plt.show()

# separate array into input and output components
X = array[:,0:3]
Y = array[:,3]
Y=Y.astype('int32')
#visualize the data
scatter_matrix(data)
plt.show()

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])

#prepare the models
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('Support Vector Machine', SVC()))
models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=3)))
models.append(("DT", DecisionTreeClassifier()))
models.append(("RF", RandomForestClassifier()))

# evaluate each models using Kfold
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits=10, random_state=5,shuffle=True)
    cv_results = cross_val_score( model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Accuraccy Equals %f  with standard deviation (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#Improve Accuracy with Ensemble Predictions
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("The Mean of the Results while using Random Forest Classification  ",results.mean())

#Improve Accuracy with Algorithm Tuning
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, Y)
print("The Grid Best Score ",grid.best_score_)
print(grid.best_estimator_.alpha)

#Finalize And Saving my Model (KNeighborsClassifier) which gives accuracy nearly 90%
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model =  KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print("KNeighborsClassifier Model Score",result)