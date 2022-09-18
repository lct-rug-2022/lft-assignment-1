#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""

import argparse
import time
from pathlib import Path

import click
import joblib
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from utils import read_corpus, cv_kfold


@click.command()
@click.option('-tf', '--train_file', default=Path('datasets/train.txt'), type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), help='Train file to learn from')
@click.option('-ttf', '--test_file', default=Path('datasets/test.txt'), type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), help='Test data file')
@click.option('-m', '--model_file', default=None, type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=True, path_type=Path), help='Save model to file')
def main(
    train_file: Path,
    test_file: Path,
    model_file: Path,
) -> None:
    X_train, y_train = read_corpus(train_file)
    X_test, y_test = read_corpus(test_file)  # use it carefully, really train set

    pipeline = Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
        ('cls', LinearSVC(C=0.5885398838335058, intercept_scaling=0.6329639882756152,
                          max_iter=2000, multi_class='crammer_singer', random_state=42,
                          tol=0.009741897651227838))
    ])

    print('Training...')
    _training_start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - _training_start_time
    print(f'  time spent: {training_time:.2f}s')
    train_f1 = metrics.f1_score(y_train, pipeline.predict(X_train), average='micro')
    print(f'  f1-micro train score: {train_f1:.4f}')

    print('Validate model...')
    test_f1 = metrics.f1_score(y_test, pipeline.predict(X_test), average='micro')
    print(f'  f1-micro test score: {test_f1:.4f}')

    if model_file:
        print(f'Saving model to {model_file}...')
        joblib.dump(pipeline, model_file)
        print('  done')

if __name__ == '__main__':
    main()
