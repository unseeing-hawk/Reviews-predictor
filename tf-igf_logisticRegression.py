import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pymystem3 import Mystem
import joblib

from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()

# Ignore warnings
warnings.filterwarnings("ignore")

# Directory with review files
path = r'Data/'

# Create DataFrame
df = pd.DataFrame(columns=['reviews', 'sentiment'])

for directory in os.listdir(path):
    if os.path.isdir(path + directory):
        files = np.array(os.listdir(path + directory))
        for file in files:
            with open(os.path.join(path + directory + '/', file), encoding='utf-8') as f:
                review = f.read()
                current_df = pd.DataFrame({'reviews': [review], 'sentiment': directory})
                df = pd.concat([df, current_df], ignore_index=True)

# Text preprocessing
nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer('russian')
lemmatizer = Mystem()


def preprocess_text(text):
    words = lemmatizer.lemmatize(text.lower())  # Lemmatization
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Remove stop words and non-alphabetic tokens
    return ' '.join(words)


df['reviews'] = df['reviews'].apply(preprocess_text)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['reviews'], df['sentiment'], test_size=0.2, random_state=42)

def tokenizer_stemmer(text):
    return[porter.stem(word) for word in text.split()]

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=True,
                        preprocessor=None,  # Defined preprocessor in Data Cleaning
                        tokenizer=tokenizer_stemmer,
                        use_idf=True,
                        norm='l2',
                        smooth_idf=True)

X = tfidf.fit_transform(X_train)  # Use X_train here for feature extraction

# Document Classification using Logistic Regression
# Logistic Regression model with cross-validation
clf = LogisticRegressionCV(cv=5,
                           scoring='accuracy',
                           random_state=1,
                           n_jobs=-1,
                           verbose=3,
                           max_iter=1000).fit(X, y_train)  # Use X instead of X_train here

# Test the model
X_test_transformed = tfidf.transform(X_test)
test_predictions = clf.predict(X_test_transformed)

# Print the results
print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Classification Report:\n", classification_report(y_test, test_predictions))
