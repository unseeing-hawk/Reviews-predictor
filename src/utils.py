import os
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pymystem3 import Mystem

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer('russian')
lemmatizer = Mystem()

__all__ = ['stop_words', 'stemmer', 'lemmatizer']
    
path = r'resourses/Data/'

def read_data():
    df = pd.DataFrame(columns=['review', 'sentiment'])

    for directory in os.listdir(path):
        if os.path.isdir(path + directory):
            files = np.array(os.listdir(path + directory))
            for file in files:
                with open(os.path.join(path + directory + '/', file), encoding='utf-8') as f:
                    review = f.read()
                    current_df = pd.DataFrame({'review': [review], 'sentiment': directory})
                    df = pd.concat([df, current_df], ignore_index=True)

    return df
