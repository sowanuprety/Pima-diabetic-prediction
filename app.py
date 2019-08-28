# Define a python framework for a REST API

from MachineLearning import MachineLearning

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

trained_models = None
loaded_models = None  # Model Loaded from the pickle files
app = Flask(__name__)


@app.route('/trainmodel', methods=['POST'])
def train_model():
    success_status = False
    if request.method == 'POST':
        success_status = True
        global trained_models
        ml_model = MachineLearning('diabetes.csv')
        ml_model.svm()
        ml_model.logistic_regression()
        ml_model.decision_tree()
        ml_model.persist_model()
        trained_models = ml_model.filenames

        return render_template("trainmodels.html", success_status=success_status)

def get_model():
    global loaded_models
    with open('svm_linear.pkl', 'rb') as f:
        loaded_models = pickle.load(f)


@app.route('/')
def home():
    # get_model()
    # temp = ""
    # for i in trained_models:
    #     temp += str(i) + "\n"
    #
    # return_string = "The Trained Models Are: " + temp

    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        get_model()
        output_values = ["Non-Diabetic", "Diabetic"]
        data = request.form
        # data = np.array(data)[np.newaxis, :]
        # prediction = loaded_models.predict(data)
        # data = pd.DataFrame(data)
        # print(type(dict(data)))
        data = dict(data)
        data = pd.DataFrame(data, index=[0])
        # print(data)
        prediction = loaded_models.predict(data)
        # print(prediction)
        return_str = "The Prediction is: "+str(output_values[prediction[0]])
        # print(return_str)
    return return_str


if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0")
