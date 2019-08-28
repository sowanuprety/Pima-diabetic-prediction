from DataProcessing import read_data

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

import warnings

warnings.filterwarnings('ignore')


class MachineLearning:
    filenames = []

    def __init__(self, path):
        self.pima_df = read_data(path)
        outcome = self.pima_df['Outcome']
        data = self.pima_df[self.pima_df.columns[:8]]
        train, test = train_test_split(self.pima_df, test_size=0.25, random_state=0,
                                       stratify=self.pima_df['Outcome'])  # stratify the outcome
        self.train_X = train[train.columns[:8]]
        self.test_X = test[test.columns[:8]]
        # print(type(self.test_X))
        # print(self.test_X)
        self.train_Y = train['Outcome']
        self.test_Y = test['Outcome']

        self.trained_models = {}


    def svm(self):
        types = ['rbf', 'linear']
        for i in types:
            model = svm.SVC(kernel=i)
            model.fit(self.train_X, self.train_Y)
            prediction = model.predict(self.test_X)
            self.trained_models["svm_" + i] = model
            print('Accuracy for SVM kernel=', i, 'is', metrics.accuracy_score(prediction, self.test_Y))

    def logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.train_X, self.train_Y)
        prediction = model.predict(self.test_X)
        self.trained_models['logisticregression'] = model
        print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction, self.test_Y))

    def decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.train_X, self.train_Y)
        prediction = model.predict(self.test_X)
        self.trained_models['decisiontree'] = model
        print('The accuracy of the Decision Tree is', metrics.accuracy_score(prediction, self.test_Y))

    def persist_model(self):
        for each_model in self.trained_models.keys():
            filename = each_model + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(self.trained_models[each_model], f)
                MachineLearning.filenames.append(filename)


if __name__ == '__main__':
    mlModel = MachineLearning('diabetes.csv')
    mlModel.svm()
    mlModel.logistic_regression()
    mlModel.decision_tree()
    mlModel.persist_model()
    print(mlModel.filenames)
