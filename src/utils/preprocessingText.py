from utils import *

import warnings

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pymystem3 import Mystem


warnings.filterwarnings("ignore")

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer('russian')
lemmatizer = Mystem()


# Предобработка текста
def preprocess_text_Lemm(text):
    words = lemmatizer.lemmatize(text.lower())  # Лемматизация
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]  # Убираем стоп-слова и нелитеральные токены
    return ' '.join(words)

def preprocess_text(text):
    words = text.split() 
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]  # Убираем стоп-слова и нелитеральные токены
    return ' '.join(words)


df = read_data()
df_Lemm = df.copy()

config = read_config()

df['review'] = df['review'].apply(preprocess_text)
df.to_csv('../' + config['preprocessStopWStem'], index=False)

df_Lemm['review'] = df_Lemm['review'].apply(preprocess_text_Lemm)
df_Lemm.to_csv('../' + config['preprocessStopWStemLem'], index=False)
