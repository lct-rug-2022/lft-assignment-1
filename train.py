#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""

import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from utils import read_corpus


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='datasets/train.txt', type=str,
                        help="Train file to learn from (default datasets/train.txt)")
    parser.add_argument("-df", "--dev_file", default='datasets/dev.txt', type=str,
                        help="Dev file to evaluate on (default datasets/dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_false",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    args = parser.parse_args()
    return args


def identity(inp):
    """Dummy function that just returns the input"""
    return inp


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    # TODO: comment this
    classifier.fit(X_train, Y_train)

    # TODO: comment this
    Y_pred = classifier.predict(X_test)

    # TODO: comment this
    acc = accuracy_score(Y_test, Y_pred)
    print(f"Final accuracy: {acc}")
