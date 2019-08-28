"""Importing machine learning models"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

"""
train_test_split” for train/test split and “cross_val_score” for k-fold cross validation. 
accuracy_score” is to evaluate the accuracy of the model in the train/test split method.
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from DataProcessing import Preprocessing,read_data

"""We will initialize the classifier models with their default parameters and add them to a model list."""

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))



class MachineLearning(Preprocessing):

    # def __init__(self, path):
    #     self.data = read_data(path)
    def ml_models(self):

        X = self.data.drop('Outcome', axis=1)
        y = self.pima['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        """
        Fitting each model in a loop and calculating the scores for each model
        """
        names = []
        scores = []

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
            names.append(name)

        display_score = pd.Dataframe({'Name': names, 'Score': scores})
        print(display_score)


ML = MachineLearning('diabetes.csv')
