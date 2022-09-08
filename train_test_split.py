#!/usr/bin/env python


import argparse

from sklearn.model_selection import train_test_split

from utils import read_corpus, write_corpus


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--dataset_file", default='datasets/reviews.txt', type=str,
                        help="Full dataset file to split (default datasets/reviews.txt)")
    parser.add_argument("-s", "--sentiment", action="store_false",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--train_file", default='datasets/train.txt', type=str,
                        help="Train file to learn from (default datasets/train.txt)")
    parser.add_argument("-ttf", "--test_file", default='datasets/test.txt', type=str,
                        help="Test file to evaluate on (default datasets/test.txt)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X, y = read_corpus(args.dataset_file, args.sentiment)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.9,
        random_state=42,
        shuffle=True
    )

    write_corpus(X_train, y_train, args.train_file)
    write_corpus(X_test, y_test, args.test_file)
