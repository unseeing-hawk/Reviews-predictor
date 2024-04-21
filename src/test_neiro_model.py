from utils.utils import read_config

import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


config = read_config("utils/")

test_data = pd.read_csv(config["neiroTestDatasPath"])

X_test = test_data.iloc[:, :-3] # Выбираем все столбцы, кроме последних трех (предполагается, что они являются метками классов)
y_test = test_data.iloc[:, -3:] # Выбираем только последние три столбца (предполагается, что они являются метками классов)

def TestModel(model, test_data):
    predictions = model.predict(test_data)
    return predictions

trained_model = load_model(config["neuralModelPath"])
test_predictions = TestModel(trained_model, X_test)

print("Accuracy:", accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_predictions, axis=1)))
print("Classification Report:\n", classification_report(np.argmax(y_test, axis=1), np.argmax(test_predictions, axis=1)))
