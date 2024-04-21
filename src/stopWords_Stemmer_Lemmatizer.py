from utils.utils import read_config

import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


config = read_config("utils/")

train_data = pd.read_csv(config["machineTrainDatasPath"])

X_train = train_data['review']
y_train = train_data['sentiment']

def TrainModel():
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

trained_model = TrainModel()

joblib.dump(trained_model, config["stopWStemLemModelPath"])
