import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

from gensim.models import Word2Vec

from word2vec_vectorizer import MeanEmbeddingVectorizer

from utils import read_data
from utils import lemmatizer
from utils import stemmer
from utils import stop_words


# Ignore warnings
warnings.filterwarnings("ignore")

# Text preprocessing
def preprocess_text(text):
    words = lemmatizer.lemmatize(text.lower())
    return  [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]

df = read_data()
df['review'] = df['review'].apply(preprocess_text)

# # Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df['review'],
                                                    df['sentiment'],
                                                    test_size=0.2,
                                                    random_state=42)

def train_model(x_train, y_train):
    w2v = Word2Vec(x_train,
                   vector_size=50)

    # Document Classification using Logistic Regression
    # Logistic Regression model with cross-validation
    clf = LogisticRegressionCV(cv=5,
                               scoring='accuracy',
                               random_state=1,
                               n_jobs=-1,
                               max_iter=1000)
    pipeline = make_pipeline(MeanEmbeddingVectorizer(w2v), clf)
    pipeline.fit(x_train, y_train)
    return pipeline

# Train model
model = train_model(x_train, y_train)

# Test the model
y_predicted = model.predict(x_test)

# # Print the results
print("Accuracy:", accuracy_score(y_test, y_predicted))
print("Classification Report:\n", classification_report(y_test, y_predicted))
