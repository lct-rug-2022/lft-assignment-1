#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""

import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from utils import read_corpus, cv_kfold


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='datasets/train.txt', type=str,
                        help="Train file to learn from (default datasets/train.txt)")
    parser.add_argument("-ttf", "--test_file", default='datasets/test.txt', type=str,
                        help="Dev file to evaluate on (default datasets/test.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-k", "--k_fold_cv", default=1, type=int,
                        help="K in K-fold cross validation")
    parser.add_argument("-r", "--k_fold_cv_repetitions", default=1, type=int,
                        help="Repetitions in K-fold cross validation")
    args = parser.parse_args()
    return args


def identity(inp):
    """Dummy function that just returns the input"""
    return inp


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.test_file, args.sentiment)  # use it carefully, really train set

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity)

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    classifier = Pipeline([
        ('vec', vec),
        ('cls', MultinomialNB())
    ])

    kfold_result = cv_kfold(
        classifier,
        X_train,
        Y_train,
        accuracy_score,
        k=5,
        verbose=1,
    )

    print('mean_score', kfold_result['mean_score'])
    print('mean_oof_score', kfold_result['mean_oof_score'])
    print('full_oof_scores', kfold_result['full_oof_scores'])
    print('oof_scores', kfold_result['oof_scores'])
    print('oof_score', kfold_result['oof_score'])
