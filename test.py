#!/usr/bin/env python
"""Test script to run saved model on any data"""

from pathlib import Path

import click
import joblib
from sklearn import metrics

from utils import read_corpus


@click.command()
@click.option('-ttf', '--test_file', default=Path('datasets/test.txt'), type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), help='Test data file')
@click.option('-m', '--model_file', default=Path('pipeline.pkl'), type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), help='Save model to file')
def main(
    test_file: Path,
    model_file: Path,
) -> None:
    X_test, y_test = read_corpus(test_file)

    print(f'Loading model from {model_file}...')
    pipeline = joblib.load(model_file)
    print('  done')

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


if __name__ == '__main__':
    main()
