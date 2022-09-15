#!/usr/bin/env python


import argparse

import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm
import pandas as pd

from utils import read_corpus, cv_kfold, train_validate_split


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='datasets/train.txt', type=str,
                        help="Train file to learn from (default datasets/train.txt)")
    parser.add_argument("-ttf", "--test_file", default='datasets/test.txt', type=str,
                        help="Dev file to evaluate on (default datasets/test.txt)")
    parser.add_argument("-k", "--k_fold_cv", default=1, type=int,
                        help="K in K-fold cross validation")
    parser.add_argument("-r", "--k_fold_cv_repetitions", default=1, type=int,
                        help="Repetitions in K-fold cross validation")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X, y = read_corpus(args.train_file)

    vectorizers = {
        'CountVectorizer': CountVectorizer(),
        'TfidfVectorizer': TfidfVectorizer(),
        'HashingVectorizer': HashingVectorizer(),
    }
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVM': SVC(),
        'LinearSVC': LinearSVC(),
        'RandomForestClassifier': RandomForestClassifier(),
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'MultinomialNB': MultinomialNB(),
        'KNeighborsClassifier': KNeighborsClassifier(),
    }

    scorer = lambda *x: metrics.f1_score(*x, average='weighted')

    results_f1_micro = pd.DataFrame(index=models, columns=vectorizers)
    results_f1_macro = pd.DataFrame(index=models, columns=vectorizers)
    results_f1_weighted = pd.DataFrame(index=models, columns=vectorizers)
    results_time = pd.DataFrame(index=models, columns=vectorizers)

    joblib_memory = joblib.Memory()
    for model_name, model in tqdm(models.items(), total=len(models), desc='models'):
        for vec_name, vec in tqdm(vectorizers.items(), total=len(vectorizers), desc='vectorizer'):
            pipeline = Pipeline(
                steps=[
                    ('vec', vec),
                    ('cls', model)
                ],
                memory=joblib_memory,
            )

            X_train, X_val, y_train, y_val = train_test_split(X, y)
            try:
                validate_results = train_validate_split(pipeline, X_train, y_train, X_val, y_val, scorer, verbose=0)

                results_f1_micro.loc[model_name, vec_name] = metrics.f1_score(y_val, validate_results['pred_val'], average='micro')
                results_f1_macro.loc[model_name, vec_name] = metrics.f1_score(y_val, validate_results['pred_val'], average='macro')
                results_f1_weighted.loc[model_name, vec_name] = metrics.f1_score(y_val, validate_results['pred_val'], average='weighted')
                results_time.loc[model_name, vec_name] = validate_results['time']
            except:
                results_f1_micro.loc[model_name, vec_name] = None
                results_f1_macro.loc[model_name, vec_name] = None
                results_f1_weighted.loc[model_name, vec_name] = None
                results_time.loc[model_name, vec_name] = 0

            # kfold_result = cv_kfold(pipeline, X, y, scorer=scorer, k=3)
            #
            # print('mean_score', kfold_result['mean_score'])
            # print('mean_oof_score', kfold_result['mean_oof_score'])
            # print('full_oof_scores', kfold_result['full_oof_scores'])
            # print('oof_scores', kfold_result['oof_scores'])
            # print('oof_score', kfold_result['oof_score'])
            #
            # results.loc[model_name, vec_name] = kfold_result['oof_score']

    print('\nresults_f1_micro\n', results_f1_micro)
    print('\nresults_f1_macro\n', results_f1_macro)
    print('\nresults_f1_weighted\n', results_f1_weighted)
    print('\nresults_time\n', results_time)
