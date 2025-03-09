DATA_PATH = "./data/"


def get_data_from_file(data_type: str = 'train') -> tuple[list[str], list[str]]:
    """
    Options: ['train', 'test', 'small']
    Returns data from files in lists.
    Order: clean, corrupt
    """
    with open(DATA_PATH + data_type + "_clean.txt", 'r', encoding='utf-8', newline='\n') as f:
        input_clean = [line.strip() for line in f if line != ""]

    with open(DATA_PATH + data_type + "_corrupt.txt", 'r', encoding='utf-8', newline='\n') as f:
        input_corrupt = [line.strip() for line in f if line != ""]

    return input_clean, input_corrupt

def tilde_format(num):
    return f"{num:,.2f}".replace(',', '~')