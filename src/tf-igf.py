from utils.utils import read_config

import joblib
import pandas as pd

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer


porter = PorterStemmer()

config = read_config("utils/")

train_data = pd.read_csv(config["machineTrainDatasPath"])

X_train = train_data['review']
y_train = train_data['sentiment']

def tokenizer_stemmer(text):
    return[porter.stem(word) for word in text.split()]

# Извлечение признаков с использованием TF-IDF
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=True,
                        preprocessor=None,  # Определен предобработчик в Data Cleaning
                        tokenizer=tokenizer_stemmer,
                        use_idf=True,
                        norm='l2',
                        smooth_idf=True)

X = tfidf.fit_transform(X_train)

trained_model = SVC(kernel='linear')  # линейное ядро для SVM
trained_model.fit(X, y_train)

# Сохранение модели в файл
joblib.dump(trained_model, config["tfigfModelPath"])
