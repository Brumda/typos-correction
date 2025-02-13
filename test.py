import os
import time
import torch
import wandb

from neuspell import BertChecker

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
wandb.init(project="test", name="bert-checker",
           config={
               'GPU': gpu_name,
           })
print(f"Running on: {gpu_name}")

DATA_PATH = "./data/"
CHECKPOINT = "checkpoints/subwordbert-probwordnoise/finetuned_model"
checker = BertChecker(device="cuda")
checker.from_pretrained()
checker.model.to("cuda")

current_epoch = 1
train_epochs = 2
# clean_data = "test_clean.txt"
# corrupt_data = "test_corrupt.txt"
clean_data = "actual_testing_small.txt"
corrupt_data = "actual_testing_small_second.txt"

res, acc = checker.evaluate(clean_file=os.path.join(DATA_PATH, clean_data),
                            corrupt_file=os.path.join(DATA_PATH, corrupt_data))

wandb.log({"test_accuracy": acc})
with open("result.txt", "a") as f:
    f.write(f"Evaluation of pretrained BERT model:")
    f.write(f"Result: {res}\n")
    f.write(20 * "#")



for epoch in range(current_epoch, train_epochs + 1):
    start_time = time.time()
    checker.finetune(clean_file=os.path.join(DATA_PATH, clean_data),
                     corrupt_file=os.path.join(DATA_PATH, corrupt_data),
                     n_epochs=1)
    elapsed_time = time.time() - start_time

    checker.from_pretrained(os.path.join(CHECKPOINT, f'epoch{epoch:02d}'))

    res, acc = checker.evaluate(clean_file=os.path.join(DATA_PATH, clean_data),
                                corrupt_file=os.path.join(DATA_PATH, corrupt_data))

    wandb.log({"test_accuracy": acc})
    with open("result.txt", "a") as f:
        f.write(f"Evaluation of BERT model after {epoch} epochs:")
        f.write(f"Result: {res}\n")
        f.write(20 * "#")

wandb.finish()
