from utils.utils import read_config

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


config = read_config("utils/")
df = pd.read_csv(config['preprocessStopWStemLem'])

# Mapping 
sentiment_mapping = {'pos': 0, 'neu': 1, 'neg': 2}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max([len(x) for x in X_train_seq])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

num_classes = 3
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

train_data = np.column_stack((X_train_pad, y_train_one_hot))
np.savetxt(config['neiroTrainDatasPath'], train_data, delimiter=',')
test_data = np.column_stack((X_test_pad, y_test_one_hot))
np.savetxt(config['neiroTestDatasPath'], test_data, delimiter=',')

# Build neural model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_pad, y_train_one_hot, epochs=10, batch_size=16, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_one_hot, verbose=0)
print("Test Accuracy:", test_accuracy)

model.save(config['neuralModelPath'])
