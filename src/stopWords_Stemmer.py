import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pymystem3 import Mystem
import joblib
      

warnings.filterwarnings("ignore")

path = r'resourses/Data/'

df = pd.DataFrame(columns=['review', 'sentiment'])

for directory in os.listdir(path):
    if os.path.isdir(path + directory):
        files = np.array(os.listdir(path + directory))
        for file in files:
            with open(os.path.join(path + directory + '/', file), encoding='utf-8') as f:
                review = f.read()
                current_df = pd.DataFrame({'review': [review], 'sentiment': directory})
                df = pd.concat([df, current_df], ignore_index=True)

# Предобработка текста
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer('russian')
lemmatizer = Mystem()

def preprocess_text(text):
    words = text.split() 
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Убираем стоп-слова и нелитеральные токены
    return ' '.join(words)

df['review'] = df['review'].apply(preprocess_text)

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Обучение модели
def TrainModel(train_data):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

# Тестирование модели
def TestModel(model, test_data):
    predictions = model.predict(test_data)
    return predictions

trained_model = TrainModel(X_train)

# Сохранение модели в файл
joblib.dump(trained_model, 'resourses/stopWords_Stemmer_model.joblib')

test_predictions = TestModel(trained_model, X_test)

print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Classification Report:\n", classification_report(y_test, test_predictions))
