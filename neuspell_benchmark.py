import argparse

import torch
from neuspell import BertChecker

# cant import unless specifically set up
try:
    from neuspell import ElmosclstmChecker
except ImportError:
    ElmosclstmChecker = None

import wandb
from benchmark import ModelBenchmark
from helpers import get_data_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert", help="Which model to use")
parser.add_argument("--finetuned", type=bool, default=False, help="Use finetuned or pretrained model")

args = parser.parse_args()

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

MODEL = {"bert": {"model_name":     "subwordbert-probwordnoise",
                  "wandb_run_name": "bert-checker",
                  "model":          BertChecker(device="cuda"), },
         "elmo": {"model_name":     "elmoscrnn-probwordnoise",
                  "wandb_run_name": "elmo-checker",
                  "model":          ElmosclstmChecker(device="cuda") if ElmosclstmChecker else None}, }

wandb.init(project="benchmark_" + MODEL[args.model]["wandb_run_name"], name=MODEL[args.model]["wandb_run_name"],
           config={'GPU': gpu_name, })

CHECKPOINT = f"checkpoints/{MODEL[args.model]['model_name']}/finetuned_model"
checker = MODEL[args.model]["model"]

if args.finetuned:
    checker.from_pretrained(CHECKPOINT)
else:
    checker.from_pretrained()

benchmark = ModelBenchmark()

corrupt, clean = get_data_from_file('test')
warm_up_runs = 2
num_runs = 5
name = MODEL[args.model]["wandb_run_name"] + "-finetuned" if args.finetuned else "-pretrained"
res = benchmark.benchmark_model(checker,
                                corrupt,
                                clean,
                                name,
                                lambda model, data: model.correct_string(data),
                                warm_up_runs=warm_up_runs,
                                num_runs=num_runs)
wandb.log(res.__dict__)

with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(name + " benchmark results:\n")
    f.write(f"{res}\n")
    f.write(f"{res.create_tex_table_perf_metrics()}\n")
    f.write(f"{res.create_tex_table_corr_metrics()}\n")

wandb.finish()
