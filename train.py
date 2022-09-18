#!/usr/bin/env python
"""Training script of the best model """

import time
from pathlib import Path

import click
import joblib
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import read_corpus


@click.command()
@click.option('-tf', '--train_file', default=Path('datasets/train.txt'), type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), help='Train file to learn from')
@click.option('-ttf', '--test_file', default=Path('datasets/test.txt'), type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), help='Test data file')
@click.option('-m', '--model_file', default=Path('pipeline.pkl'), type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=True, path_type=Path), help='Save model to file')
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
    test_predict = pipeline.predict(X_test)
    test_f1 = metrics.f1_score(y_test, test_predict, average='micro')
    test_recall = metrics.recall_score(y_test, test_predict, average='micro')
    test_precision = metrics.precision_score(y_test, test_predict, average='micro')
    test_accuracy = metrics.accuracy_score(y_test, test_predict)
    print(f'  f1-micro test score: {test_f1:.4f}')
    print(f'  recall-micro test score: {test_recall:.4f}')
    print(f'  precision-micro test score: {test_precision:.4f}')
    print(f'  accuracy test score: {test_accuracy:.4f}')

    if model_file:
        print(f'Saving model to {model_file}...')
        joblib.dump(pipeline, model_file)
        print('  done')


if __name__ == '__main__':
    main()
