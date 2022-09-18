import typing as tp
from collections.abc import Callable
from copy import deepcopy
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from tqdm.autonotebook import tqdm


def read_corpus(
        corpus_file: str | Path,
        use_sentiment: bool = False,
        split: bool = False,
) -> tuple[list[list[str] | str], list[str]]:
    """
    Read text corpus as list
    :param corpus_file: filename to read dataset from
    :param use_sentiment: select between sentiment and classification labels to save
    :param split: return each text or as a str
    :return: list of tuples with text and corresponding label
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:] if split else ' '.join(tokens[3:]))
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def train_validate_split(
        model: BaseEstimator,
        data_train: list,
        target_train: list,
        data_val: list,
        target_val: list,
        scorer: Callable[[tp.Any, tp.Any], float],
        *,
        verbose: int = 1,
) -> dict[str, tp.Any]:
    """Fit predict model on current split
    :param model: Model to be trained
    :param data_train: train data to perform k-fold cv
    :param target_train: target train data
    :param data_val: validate data to scoring
    :param target_val: validate target to scoring
    :param scorer: function to score prediction. args: target, prediction
    :param verbose: lager - verbose
    :return: dict with results of cv
    """
    data_train, data_val = np.array(data_train), np.array(data_val)
    target_train, target_val = np.array(target_train), np.array(target_val)

    # Fit model in current fold
    if verbose > 1:
        print('  Fitting model...')
    start_time = time.time()
    model.fit(data_train, target_train)
    end_time = time.time()

    # predict for out-fold and save it for validation
    if verbose > 1:
        print('  Predicting oof...')
    pred_val = model.predict(data_val)

    # Score for out-fold
    if verbose > 1:
        print('  Scoring oof...')
    score_fold = scorer(target_val, pred_val)
    if verbose > 1:
        print(f'  Fold score: {score_fold}')

    return {
        'pred_val': pred_val,
        'score': score_fold,
        'time': end_time - start_time,
    }


def cv_kfold(
        model: BaseEstimator,
        data: list,
        target: list,
        scorer: Callable[[tp.Any, tp.Any], float],
        k: int = 5,
        *,
        verbose: int = 1,
        random_state: int = 42,
) -> dict[str, tp.Any]:
    """Fit predict model multiple times with k-fold cross validation
    :param model: Model to be trained
    :param data: train data to perform k-fold cv
    :param target: target train data
    :param scorer: function to score prediction. args: target, prediction
    :param k: number of folds in cross validation
    :param verbose: lager - verbose
    :param random_state: fixed random state
    :return: dict with results of cv
    """
    random_instance = np.random.RandomState(random_state)

    data = np.array(data)
    target = np.array(target)

    pred_train = np.empty(data.shape[0], dtype=data.dtype)

    mean_score = 0
    full_oof_score, split_oof_score = [], []
    times = []

    pred_split_train = np.empty(data.shape[0], dtype=data.dtype)
    full_oof_score.append([])

    kf = KFold(n_splits=k, shuffle=True, random_state=random_instance)
    if verbose:
        kf_gen = tqdm(enumerate(kf.split(data)), desc='Folds', total=k)
    else:
        kf_gen = enumerate(kf.split(data))
    for i, (train_index, val_index) in kf_gen:
        # select current train/val split
        data_train, data_val = data[train_index], data[val_index]
        target_train, target_val = target[train_index], target[val_index]

        # Fit model in current fold
        model_fold = deepcopy(model)
        fold_result = train_validate_split(
            model_fold,
            data_train, target_train,
            data_val, target_val,
            scorer,
            verbose=verbose,
        )

        times.append(fold_result['time'])
        pred_val = fold_result['pred_val']
        score_fold = fold_result['score']

        # save for out-fold validation
        pred_train[val_index] = pred_val
        pred_split_train[val_index] = pred_val

        # Score for out-fold
        mean_score += score_fold / float(k)
        full_oof_score[-1].append(score_fold)
        if verbose > 1:
            print(f'  Fold {i} score: {score_fold}')

        split_oof_score.append(scorer(target, pred_split_train))

    return {
        'train_pred': pred_train,
        'mean_score': mean_score,
        'mean_oof_score': np.mean(split_oof_score),
        'oof_scores': split_oof_score,
        'full_oof_scores': full_oof_score,
        'oof_score': scorer(target, pred_train),
        'times': times,
        'mean_time': np.mean(times),
    }
