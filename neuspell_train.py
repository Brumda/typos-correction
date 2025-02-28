import argparse
import os
import time

import torch
from neuspell import ElmosclstmChecker, BertChecker

import wandb
from helpers import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert", help="Which model to use")
parser.add_argument("--program_test", default=False, type=bool, help="Test the program on small data")
parser.add_argument("--current_epoch", default=1, type=int, help="Epoch to be trained next")
parser.add_argument("--train_epochs", default=5, type=int, help="How many epochs to train")

args = parser.parse_args()

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Running on: {gpu_name}")

##############################################################################
MODEL = {"bert": {"model_name": "subwordbert-probwordnoise",
                  "wandb_run_name": "bert-checker",
                  "model": BertChecker(device="cuda"), },
         "elmo": {"model_name": "elmoscrnn-probwordnoise",
                  "wandb_run_name": "elmo-checker",
                  "model": ElmosclstmChecker(device="cuda")}, }

_DATA = {"run":
             {"train": ["train_clean.txt", "train_corrupt.txt"],
              "test": ["test_clean.txt", "test_corrupt.txt"]},
         "test":
             {"train": ["small_clean.txt", "small_corrupt.txt"],
              "test": ["small_clean.txt", "small_corrupt.txt"]}}

DATA = _DATA["test" if args.program_test else "run"]  # pick between training and program testing
##############################################################################
CHECKPOINT = f"checkpoints/{MODEL[args.model]['model_name']}/finetuned_model"
wandb.init(project="neuspell", name=MODEL[args.model]["wandb_run_name"], resume="allow",
           config={'GPU': gpu_name, })
##############################################################################
checker = MODEL[args.model]["model"]

if args.current_epoch > 1:
    checker.from_pretrained(os.path.join(CHECKPOINT, f'epoch_{args.current_epoch - 1:02d}'))  # load previous epoch
else:
    checker.from_pretrained()
    _, prints, acc = checker.evaluate(clean_file=os.path.join(DATA_PATH, DATA["test"][0]),
                                      corrupt_file=os.path.join(DATA_PATH, DATA["test"][1]))

    wandb.log({"test_accuracy": acc})
    with open("results.txt", "a") as f:
        f.write(f"Evaluation of pretrained ELMO model:\n")
        f.write(f"Result:\n{prints}\n")
        f.write(20 * "#" + "\n")

for epoch in range(args.current_epoch, args.train_epochs + args.current_epoch):
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
