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
