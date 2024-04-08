import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer

from pymystem3 import Mystem


porter = PorterStemmer()
warnings.filterwarnings("ignore")
path = r'resourses/Data/'

df = pd.DataFrame(columns=['reviews', 'sentiment'])

for directory in os.listdir(path):
    if os.path.isdir(path + directory):
        files = np.array(os.listdir(path + directory))
        for file in files:
            with open(os.path.join(path + directory + '/', file), encoding='utf-8') as f:
                review = f.read()
                current_df = pd.DataFrame({'reviews': [review], 'sentiment': directory})
                df = pd.concat([df, current_df], ignore_index=True)


nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer('russian')
lemmatizer = Mystem()

def preprocess_text(text):
    words = lemmatizer.lemmatize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


df['reviews'] = df['reviews'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['reviews'], df['sentiment'], test_size=0.2, random_state=42)

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

# Классификация с использованием логистической регрессии
# Модель логистической регрессии с перекрестной проверкой
trained_model = LogisticRegressionCV(cv=5,
                           scoring='accuracy',
                           random_state=1,
                           n_jobs=-1,
                           verbose=3,
                           max_iter=1000).fit(X, y_train)

# Сохранение модели в файл
joblib.dump(trained_model, 'resourses/tf-igf_logisticReg_model.joblib')

X_test_transformed = tfidf.transform(X_test)
test_predictions = trained_model.predict(X_test_transformed)

print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Classification Report:\n", classification_report(y_test, test_predictions))
