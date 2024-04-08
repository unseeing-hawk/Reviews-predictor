# from utils import *
from utils import read_data
from stopWords_Stemmer_Lemmatizer import preprocess_text

import joblib
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report

from nltk.stem.porter import PorterStemmer


porter = PorterStemmer()
warnings.filterwarnings("ignore")

df = read_data()
df['review'] = df['review'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

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
joblib.dump(trained_model, 'resourses/tf-igf.joblib')

X_test_transformed = tfidf.transform(X_test)
test_predictions = trained_model.predict(X_test_transformed)

print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Classification Report:\n", classification_report(y_test, test_predictions))
