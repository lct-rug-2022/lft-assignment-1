
def read_corpus(corpus_file: str, use_sentiment: bool = False) -> tuple[list[list[str]], list[str]]:
    """TODO: add function description"""
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def write_corpus(X: list[str], y: list[str], corpus_file: str, use_sentiment: bool = False) -> None:
    """TODO: add function description"""

    with open(corpus_file, 'w', encoding='utf-8') as f:
        if use_sentiment:
            f.writelines(
                ' '.join(['-', y_sample, *X_sample])
                for X_sample, y_sample in zip(X, y)
            )
        else:
            f.writelines(
                ' '.join([y_sample, '-', *X_sample])
                for X_sample, y_sample in zip(X, y)
            )
