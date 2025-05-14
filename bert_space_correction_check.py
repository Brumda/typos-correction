from neuspell import BertChecker
from helpers import get_data_from_file

corrupt, clean = get_data_from_file('test')
path = "checkpoints/subwordbert-probwordnoise/finetuned_model"
checker = BertChecker(device="cuda")
checker.from_pretrained(path)

wrong_sentences = []
for cor, cl in zip(corrupt, clean):
    if cl != checker.correct_string(cor, correct_spaces=True):
        wrong_sentences.append(cor)

print(*wrong_sentences, sep='\n')
with open("results.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(wrong_sentences))