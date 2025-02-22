import os
import time
import torch
import wandb

from neuspell import ElmosclstmChecker, BertChecker

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Running on: {gpu_name}")

##############################################################################
pick_model = 1 # TODO make into argument
MODEL = {0: "subwordbert-probwordnoise",
         1: "elmoscrnn-probwordnoise", }

NAME = {0: "bert-checker",
        1: "elmo-checker", }

DATA_PATH = "./data/"
CHECKPOINT = f"checkpoints/{MODEL[pick_model]}/finetuned_model"
wandb.init(project="neuspell", name=NAME[pick_model], resume="allow",
           config={
               'GPU': gpu_name, })
##############################################################################
current_epoch = 1  # epoch to be trained next
train_epochs = 8  # how many epochs to train
_DATA = {"run":
            {"train": ["train_clean.txt", "train_corrupt.txt"],
             "test": ["test_clean.txt", "test_corrupt.txt"]},
        "test":
            {"train": ["actual_testing_small.txt", "actual_testing_small_second.txt"],
             "test": ["actual_testing_small.txt", "actual_testing_small_second.txt"]}}
# TODO make into argument
DATA = _DATA["run"] # pick between training and program testing
##############################################################################
CHECKERS = [BertChecker(device="cuda"), ElmosclstmChecker(device="cuda")]
checker = CHECKERS[pick_model]

if current_epoch > 1:
    checker.from_pretrained(os.path.join(CHECKPOINT, f'epoch_{current_epoch - 1:02d}'))  # load previous epoch
else:
    checker.from_pretrained()

checker.model.to("cuda")

if current_epoch <= 1:
    _, prints, acc = checker.evaluate(clean_file=os.path.join(DATA_PATH, DATA["test"][0]),
                                      corrupt_file=os.path.join(DATA_PATH, DATA["test"][1]))

    wandb.log({"test_accuracy": acc})
    with open("results.txt", "a") as f:
        f.write(f"Evaluation of pretrained ELMO model:\n")
        f.write(f"Result:\n{prints}\n")
        f.write(20 * "#" + "\n")

for epoch in range(current_epoch, train_epochs + current_epoch):
    start_time = time.time()
    checker.finetune(clean_file=os.path.join(DATA_PATH, DATA["train"][0]),
                     corrupt_file=os.path.join(DATA_PATH, DATA["train"][1]),
                     n_epochs=1)
    elapsed_time = time.time() - start_time

    checker.from_pretrained(os.path.join(CHECKPOINT, f'epoch_{epoch:02d}'))

    _, prints, acc = checker.evaluate(clean_file=os.path.join(DATA_PATH, DATA["test"][0]),
                                      corrupt_file=os.path.join(DATA_PATH, DATA["test"][1]))

    wandb.log({"test_accuracy": acc})
    with open("results.txt", "a") as f:
        f.write(f"Evaluation of ELMO model after {epoch} epochs:\n")
        f.write(f"Result:\n{prints}\n")
        f.write(20 * "#" + "\n")

wandb.finish()
