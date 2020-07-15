import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO


cd_data = 'data/'
data = 'RRDinerCoffeeData.csv'
coffeeData = pd.read_csv(cd_data + data)

coffeeData.rename(columns = {"spent_month":"spent_last_month", "spent_week":"spent_last_week", "SlrAY":"Salary"},
            inplace = True)

coffeeData['Decision'].replace(1.0, value='YES', inplace=True)
coffeeData['Decision'].replace(0.0, 'NO', inplace=True)

NOPrediction = coffeeData.dropna()

Prediction = coffeeData[pd.isnull(coffeeData["Decision"])]

features = ["Age", "Gender", "num_coffeeBags_per_year", "spent_last_week", "spent_last_month",
       "Salary", "Distance", "Online"]

X = NOPrediction[features]

y = NOPrediction['Decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=246)

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)

entr_model = tree.DecisionTreeClassifier(criterion="entropy", random_state = 1234)


entr_model.fit(X_train, y_train)


y_pred = entr_model.predict(X_test)

y_pred = pd.Series(y_pred)

entr_model
#____________________________________________________________________________________________________________________

# Declare a variable called entr_model, and assign it: tree.DecisionTreeClassifier(criterion="entropy", random_state = 1234)
entr_model = tree.DecisionTreeClassifier(criterion="entropy", random_state = 1234)

# Call fit() on entr_model, and pass in X_train and y_train, in that order
entr_model.fit(X_train, y_train)

# Call predict() on entr_model with X_test passed to it, and assign the result to a variable y_pred
y_pred = entr_model.predict(X_test)

# Assign y_pred the following: pd.Series(y_pred)
y_pred = pd.Series(y_pred)

# Check out entr_model
entr_model
