#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import Libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

if __name__ == '__main__':
    dataset = pd.read_csv('kickstarter_data_full.csv')
    data = dataset[['blurb', 'SuccessfulBool']]
    after_data = data.dropna(axis=0, how='any')
    reset_index = after_data.reset_index(drop=True)

    #    n = len(reset_index)
    #    clean_blurbs = []
    #    counter = 0
    #    for i in range(0,n):
    #        blurb = re.sub('[^a-zA-Z]', ' ', reset_index['blurb'][i])
    #        blurb = blurb.lower()
    #        blurb = blurb.split()
    #        ps = PorterStemmer()
    #        blurb = [ps.stem(word) for word in blurb if not word in set(stopwords.words('english'))]
    #        blurb = ' '.join(blurb)
    #        clean_blurbs.append(blurb)
    #        counter += 1

    # save clean_blurbs to file
    #    with open('results/clean_blurbs', 'wb') as f:
    #        pickle.dump(clean_blurbs, f)

    # load saved clean_blurbs
    with open('clean_blurbs', 'rb') as f:
        clean_blurbs = pickle.load(f)

    n = len(clean_blurbs)

    # CREATE BAG OF WORDS MODEL
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer()
    X = cv.fit_transform(clean_blurbs).toarray()
    y = reset_index.iloc[0:n, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Run through each classifier, train on X_train and y_train and the test them using the score function
    algs = [
        GaussianNB(),
        MultinomialNB(),
        BernoulliNB(),
        LogisticRegression(),
        SGDClassifier()
    ]

    with open('results/blurb_accuracy.txt', 'w') as f:
        for alg in algs:
            print(f"training {type(alg).__name__}")
            alg = alg.fit(X_train, y_train)
            print(f"{type(alg).__name__}: {alg.score(X_test, y_test)}", file=f)
            print(f"Finished training {type(alg).__name__}")

        print("", file=f)
        print("", file=f)
