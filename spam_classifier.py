# importing the Dataset

import pandas as pd

# read data
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])

# Data cleaning and preprocessing
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# initialize  object of stemmer
ps = PorterStemmer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])  # remove spaces and symbols and accept alphabet
    review = review.lower()  # convert all to lower
    review = review.split()  # split all in word
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # stem all words
    review = ' '.join(review)
    corpus.append(review)

print('corpus--', corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)


# evaluating
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix--', cm)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy score--', accuracy)

