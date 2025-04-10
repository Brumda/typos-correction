import pandas as pd

DATA_PATH = "./data/"


def get_data_from_file(data_type: str = 'train') -> tuple[list[str], list[str]]:
    """
    Options: ['train', 'test', 'small']
    Returns data from files in lists.
    Return order: corrupt, clean
    """
    with open(DATA_PATH + data_type + "_corrupt.txt", 'r', encoding='utf-8', newline='\n') as f:
        corrupt = [line.strip() for line in f if line != ""]

    with open(DATA_PATH + data_type + "_clean.txt", 'r', encoding='utf-8', newline='\n') as f:
        clean = [line.strip() for line in f if line != ""]

    return corrupt, clean


def tilde_format(num):
    return f"{num:,.2f}".replace(',', '~')


def get_df(_data, probs=False):
    rows = []
    for commit in _data.edits:
        for edit in commit:
            if edit["src"]["lang"] == "eng":
                text = edit["src"]["text"]
                target = edit["tgt"]["text"]
                if probs:
                    rows.append({"text": text, "target": target, "prob": edit["prob_typo"]})
                else:
                    rows.append({"text": text, "target": target})

    return pd.DataFrame(rows)


def write_column_to_file(series, filename, separator='\n'):
    with open(filename, 'w', encoding='utf-8', buffering=8192) as f:
        for chunk in series.astype(str):
            f.write(chunk.strip() + separator)


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', newline='\n') as f:
        return sum(1 for _ in f)
