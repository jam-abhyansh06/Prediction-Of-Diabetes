# Author : jam-abhyansh06

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('./dataset/diabetes.csv')
data.isnull().values.any() # to check if any value is null


correlation = data.corr()  # correlation matrix
correlation


# plot heatmap of correlation matrix
plt.figure(figsize=(20,20))
hmap = sns.heatmap(correlation, annot=True, cmap="cubehelix")

outcome_true = len(data.loc[data['Outcome'] == 1])
outcome_false = len(data.loc[data['Outcome'] == 0])


from sklearn.model_selection import train_test_split

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
predict_class = ['Outcome']

X = data[features].values
y = data[predict_class].values

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state = 12)


print('Total rows : {0}'.format(len(data)))
print("Rows missing in Glucose: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("Rows missing in BloodPressure: {0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("Rows missing in SkinThickness: {0}".format(len(data.loc[data['SkinThickness'] == 0])))
print("Rows missing in Insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))
print("Rows missing in BMI: {0}".format(len(data.loc[data['BMI'] == 0])))
print("Rows missing in DiabetesPedigreeFunction: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("Rows missing in Age: {0}".format(len(data.loc[data['Age'] == 0])))

from sklearn.preprocessing import Imputer
fill_values = Imputer(missing_values = 0, strategy='mean', axis = 0)
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


from sklearn.ensemble import RandomForestClassifier as RFC
random_forest_model = RFC(random_state = 12)
random_forest_model.fit(X_train, y_train.ravel())


from sklearn import tree
estimators = random_forest_model.estimators_

# rough visualize decision tree in random forest

plt.figure(figsize=(15,10))
for i in range(len(estimators)):
    tree.plot_tree(estimators[i], filled=True)
# tree.plot_tree(random_forest_model.estimators_[0], filled=True)



predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# Hyperparameter optimization

from sklearn.model_selection import RandomizedSearchCV
import xgboost

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,
                                     scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


from datetime import datetime

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y.ravel())
timer(start_time) # timing ends here for "start_time" variable


random_search.best_estimator_

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.3, gamma=0.0, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y.ravel(),cv=10)

score.mean()

