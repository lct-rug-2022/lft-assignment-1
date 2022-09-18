#!/usr/bin/env python
"""Dataset train/test split script"""

from pathlib import Path

import click
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('dataset_file', default=Path('datasets/reviews.txt'), type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), help='Train file to learn from')
@click.option('-tf', '--train_file', default=Path('datasets/train.txt'), type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path), help='Train data file')
@click.option('-ttf', '--test_file', default=Path('datasets/test.txt'), type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path), help='Test data file')
def main(
    dataset_file: Path,
    train_file: Path,
    test_file: Path,
) -> None:
    with open(dataset_file, encoding='utf-8') as f:
        lines = [l for l in f]

    X_train, X_test = train_test_split(
        lines,
        train_size=0.9,
        random_state=42,
        shuffle=True,
    )

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(''.join(X_train))

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(''.join(X_test))
