#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pickle

if __name__ == '__main__':
    dataset = pd.read_csv('../data/train_data.csv')
    data = dataset[['blurb', 'SuccessfulBool']]
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # load saved df
    with open('clean_blurbs_bag_df.pickle', 'rb') as f:
        train_df = pickle.load(f)

    with open('test_clean_blurbs_bag_df.pickle', 'rb') as f:
        test_df = pickle.load(f)

    n = len(train_df)

    from sklearn.feature_extraction.text import CountVectorizer
    clean_blurbs = train_df['blurb'].values
    cv = CountVectorizer()
    train_X = cv.fit_transform(clean_blurbs)
    train_y = train_df.iloc[0:n, 1].values

    alg = BernoulliNB()
    alg.fit(train_X, train_y)
    # new = alg.predict_proba(train_X)

    test_clean_blurbs = test_df['blurb'].values
    cv2 = CountVectorizer()
    test_X = cv2.fit_transform(test_clean_blurbs)

    predictions = alg.predict_proba(test_X)
    pred_success = predictions[:, 1]
    updated_df = dataset
    updated_df['Probability_of_Success_from_Blurb'] = pred_success
    updated_df.to_csv('../data/test_data_updated.csv')

    fill = 2
