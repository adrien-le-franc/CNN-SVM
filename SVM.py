'''
script designed to train a SVM predictor on a set of labeled features previously
extracted and perform evaluation

the predictor is saved for further applications
'''

import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import argparse

# paths to be adapted to your own files
features_path = "../features/features3.txt"
labels_path = "../features/labels3.txt"

# to save predictors in a file
predictor_path = "predictor3.txt"


def  read_file(file_path):
    '''
    loads data into a np.array

    '''

    with  open(file_path) as f:
        X=np.array ([[ float(x) for x in l.strip (). split(" ")] for l in f.readlines ()])
    return X


def train_and_eval(features_path, labels_path, predictor_path):
    '''
    trains SVM and performs evaluation. A predictor is saved for further
    applications

    '''

    ### load data
    features = read_file(features_path)
    labels = read_file(labels_path)

    ### split test and training sets (20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
    test_size=0.2, random_state=42)

    ### train
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    ### save model in file for further using
    save = joblib.dump(clf,predictor_path,compress=9)

    ## unpack
    pretrained_clf = joblib.load(predictor_path)

    ### predict and evaluate
    yhat = pretrained_clf.predict(X_test).reshape(y_test.shape)
    Y = abs(yhat-y_test)
    error = 0
    for i in Y:
        if i > 0:
            error += 1
    print("error rate: {}%".format(float(error/len(yhat)*100)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trains and eval SVM")
    parser.add_argument('-f', '--features', type=str, required=True)
    parser.add_argument('-l', '--labels', type=str, required=True)
    parser.add_argument('-p', '--predictor', type=int, required=True)
    args = parser.parse_args()

    train_and_eval(args.features, args.labels, args.predictor)
