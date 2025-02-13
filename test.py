import os
import time
import torch
import wandb

from neuspell import BertChecker

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
wandb.init(project="neuspell", name="bert-checker", resume="allow",
           config={
               'GPU': gpu_name,
           })
print(f"Running on: {gpu_name}")

DATA_PATH = "./data/"
CHECKPOINT = "checkpoints/subwordbert-probwordnoise/finetuned_model"

current_epoch = 1  # epoch to be trained next
train_epochs = 5  # how many epoch s to train
# TODO make this into arguments
train_clean_data = "train_clean.txt"
train_corrupt_data = "train_corrupt.txt"
test_clean_data = "test_clean.txt"
test_corrupt_data = "test_corrupt.txt"
# clean_data = "actual_testing_small.txt"
# corrupt_data = "actual_testing_small_second.txt"

checker = BertChecker(device="cuda")

if current_epoch > 1:
    checker.from_pretrained(os.path.join(CHECKPOINT, f'epoch_{current_epoch - 1:02d}'))  # load previous epoch
else:
    checker.from_pretrained()

checker.model.to("cuda")

if current_epoch <= 1:
    res, acc = checker.evaluate(clean_file=os.path.join(DATA_PATH, test_clean_data),
                                corrupt_file=os.path.join(DATA_PATH, test_corrupt_data))

    wandb.log({"test_accuracy": acc})
    with open("result.txt", "a") as f:
        f.write(f"Evaluation of pretrained BERT model:\n")
        f.write(f"Result:\n{res}\n")
        f.write(20 * "#" + "\n")

for epoch in range(current_epoch, train_epochs + current_epoch):
    start_time = time.time()
    checker.finetune(clean_file=os.path.join(DATA_PATH, train_clean_data),
                     corrupt_file=os.path.join(DATA_PATH, train_corrupt_data),
                     n_epochs=1)
    elapsed_time = time.time() - start_time

    checker.from_pretrained(os.path.join(CHECKPOINT, f'epoch_{epoch:02d}'))

    res, acc = checker.evaluate(clean_file=os.path.join(DATA_PATH, test_clean_data),
                                corrupt_file=os.path.join(DATA_PATH, test_corrupt_data))

    wandb.log({"test_accuracy": acc})
    with open("result.txt", "a") as f:
        f.write(f"Evaluation of BERT model after {epoch} epochs:\n")
        f.write(f"Result:\n{res}\n")
        f.write(20 * "#" + "\n")

wandb.finish()
