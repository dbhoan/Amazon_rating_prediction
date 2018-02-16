# Sentiment analysis on Amazon product review

Data used in this project is from: http://jmcauley.ucsd.edu/data/amazon/.

The original data was in JSON format. I have grouped rating 4 and 5 together as "positive", 1 and 2 as "negative", and ignored 3. 
I then splitted the data into training and testing sets, and saved them in csv format.

logistic_regression.py:
  A logistic regression model, using bag-of-words, 2-grams and word vectorization method from scikit-learn.
