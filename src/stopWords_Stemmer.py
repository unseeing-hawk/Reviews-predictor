from utils.utils import read_config

import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


config = read_config("utils/")

df = pd.read_csv(config["preprocessStopWStem"])

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Обучение модели
def TrainModel(train_data):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

trained_model = TrainModel(X_train)

# Сохранение модели в файл
joblib.dump(trained_model, config["stopWStemModelPath"])
