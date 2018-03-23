#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
nltk.download('stopwords')

if __name__ == '__main__':
    data_set = pd.read_csv('../data/train_data.csv')
    data = data_set[['blurb', 'SuccessfulBool']]
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    n = len(data)
    clean_blurbs = []
    counter = 0
    for i in range(n):
        blurb = re.sub('[^a-zA-Z]', ' ', data['blurb'][i])
        blurb = blurb.lower()
        blurb = blurb.split()
        ps = PorterStemmer()
        blurb = [ps.stem(word) for word in blurb if not word in set(stopwords.words('english'))]
        blurb = ' '.join(blurb)
        clean_blurbs.append(blurb)
        counter += 1

    data['blurb'] = clean_blurbs
    fill = 2
    # save clean_blurbs to file
    with open('test_clean_blurbs_bag_df.pickle', 'wb') as f:
        pickle.dump(data, f)

    # load saved clean_blurbs
    # with open('clean_blurbs_bag_df.pickle', 'rb') as f:
    #     data = pickle.load(f)

    n = len(data)

    # CREATE BAG OF WORDS MODEL
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer()
    X = cv.fit_transform(data['blurb']).toarray()
    y = data.iloc[0:n, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    fill = 12
    # Run through each classifier, train on X_train and y_train and the test them using the score function
    algs = [
        GaussianNB(),
        MultinomialNB(),
        BernoulliNB(),
        LogisticRegression(),
        SGDClassifier()
    ]

    with open('results/blurb_accuracy_bag.txt', 'w') as f:
        for alg in algs:
            print(f"training {type(alg).__name__}")
            alg = alg.fit(X_train, y_train)
            print(f"{type(alg).__name__}: {alg.score(X_test, y_test)}", file=f)
            print(f"Finished training {type(alg).__name__}")

        print("", file=f)
        print("", file=f)
