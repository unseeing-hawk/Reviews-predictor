from utils import read_config

import pandas as pd

from sklearn.model_selection import train_test_split
      

config = read_config()
df = pd.read_csv('../' + config['preprocessStopWStemLem'])

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Запись данных обучающего набора в файл
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('../' + config['machineTrainDatasPath'], index=False)

# Запись данных тестового набора в файл
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('../' + config['machineTestDatasPath'], index=False)

train_sentiment_counts = train_data['sentiment'].value_counts()

# Подсчет количества уникальных значений в столбце 'sentiment' для тестового набора
test_sentiment_counts = test_data['sentiment'].value_counts()

# Вывод результатов
print("Обучающий набор:")
print(train_sentiment_counts)

print("\nТестовый набор:")
print(test_sentiment_counts)
