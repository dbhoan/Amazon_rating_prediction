# -*- coding: utf-8 -*-
# Author: Hoan Bui Dang
# Python: 3.6

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords # Import the stop word list

from evaluation import confusion

def clean_review(s):
    return ' '.join(re.sub('[^a-zA-Z]',' ', s).lower().split())

print('Loading train data... ', end ='')
df = pd.read_csv('data_train.csv')
print('Done.')

# plot histogram of ratings
plt.hist(df.overall, bins = range(0,3))
plt.show()


train = pd.DataFrame()
train['rating'] = df['overall']
train['review'] = df['reviewText']

print('Cleaning train data... ', end = '')
# remove NaN from reviewText
train = train[~train['review'].isnull()]
train = train[~train['rating'].isnull()]
train['review'] = train['review'].apply(clean_review)
print('Done.')

#stopwords.words('English')
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000, \
                             ngram_range = (1,2) )
print('Vectorizing text... ', end = '')
features = vectorizer.fit_transform(train['review'].tolist())
vocab = vectorizer.get_feature_names()
dist = np.sum(features, axis=0)
print('Done.')


print('Training the model... ', end = '')
#classifier = MultinomialNB()
classifier = LogisticRegression()

model = classifier.fit(features,train['rating'])
print('Done.')

print('Loading test data... ', end = '')    
df2 = pd.read_csv('data_test.csv')
print('Done.')

print('Processing test data... ', end = '')    
test = pd.DataFrame()
test['rating'] = df2['overall']
test['review'] = df2['reviewText']
test = test[~test['review'].isnull()]
test = test[~test['rating'].isnull()]

test['review'] = test['review'].apply(clean_review)
test_features = vectorizer.transform(test['review'].tolist())
print('Done.')

print('Testing model... ', end = '')    
predict = model.predict(test_features)
test['predict'] = predict
#test.to_csv('reviews_home_binary_test_predict_LogisticRegression.csv', index=False)
print('Done.')

confusion(predict, test['rating'])
plt.hist(predict, bins = range(0,3))
plt.show()