import os

from neuspell import BertChecker

DATA_PATH = "../data/"
checker = BertChecker(device="cuda")
checker.from_pretrained()
checker.model.to("cuda")

print(next(checker.model.parameters()).device)
print(checker.device)

res = checker.evaluate(clean_file=os.path.join(DATA_PATH, "test_clean.txt"),
                       corrupt_file=os.path.join(DATA_PATH, "test_corrupt.txt"))
print(res)
