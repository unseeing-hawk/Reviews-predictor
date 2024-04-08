from utils import *
from utils import read_data

import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline


warnings.filterwarnings("ignore")

# Предобработка текста
def preprocess_text(text):
    words = text.split() 
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]  # Убираем стоп-слова и нелитеральные токены
    return ' '.join(words)


df = read_data()
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
