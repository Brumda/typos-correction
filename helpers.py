import pandas as pd
from neuspell.seq_modeling.helpers import merge_subtokens
from transformers import BertTokenizerFast

try:
    from neuspell.commons import spacy_tokenizer, DEFAULT_DATA_PATH
    from neuspell.seq_modeling.helpers import load_vocab_dict, untokenize_without_unks

except ImportError:
    pass

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


def process_and_merge_elmo(corrupt: str, clean: str, predict: str):
    corrupt_tokens = [spacy_tokenizer(my_str) for my_str in corrupt.split()]
    clean_tokens = [spacy_tokenizer(my_str) for my_str in clean.split()]
    predict_tokens = [spacy_tokenizer(my_str) for my_str in predict.split()]

    return corrupt_tokens, clean_tokens, predict_tokens


def process_and_merge_bert(corrupt: str, clean: str, predict: str):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    tokenizer.do_basic_tokenize = True
    tokenizer.tokenize_chinese_chars = False

    corrupt = merge_subtokens(tokenizer.tokenize(corrupt))
    clean = merge_subtokens(tokenizer.tokenize(clean))
    predict = merge_subtokens(tokenizer.tokenize(predict))

    return corrupt, clean, predict
