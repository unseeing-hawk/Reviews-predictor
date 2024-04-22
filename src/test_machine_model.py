from utils.utils import read_config

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


config = read_config("utils/")

test_data = pd.read_csv(config["machineTestDatasPath"])

X_test = test_data['review']
y_test = test_data['sentiment']


def printAccuracy(model):
    print("\n--------", model, " accuracy--------\n")
    trained_model = joblib.load(config[model])
    test_predictions = trained_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, test_predictions))
    print("Classification Report:\n", classification_report(y_test, test_predictions))

printAccuracy("stopWStemModelPath")
printAccuracy("stopWStemLemModelPath")

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_stemmer(text):
    return[porter.stem(word) for word in text.split()]

tfidf = joblib.load(config["tfidf"])
X_test = tfidf.transform(X_test)

printAccuracy("tfigfModelPath")
printAccuracy("tfigfLogisticModelPath")
printAccuracy("randomForestModelPath")
