import os
import time

from neuspell import BertChecker

DATA_PATH = "./data/"
checker = BertChecker(device="cuda")
checker.from_pretrained()
checker.model.to("cuda")

print(next(checker.model.parameters()).device)
print(checker.device)

start_time = time.time()
res = checker.evaluate(clean_file=os.path.join(DATA_PATH, "test_clean.txt"),
                       corrupt_file=os.path.join(DATA_PATH, "test_corrupt.txt"))

end_time = time.time()
elapsed_time = end_time - start_time
with open("result.txt", "a") as f:
    f.write(f"Execution time: {elapsed_time:.4f} seconds\n")
    f.write(f"Result: {res}\n")
