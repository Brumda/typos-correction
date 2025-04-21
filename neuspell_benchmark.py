import argparse
import os

from neuspell import BertChecker

from helpers import process_and_merge_bert

# cant import unless specifically set up
try:
    from neuspell import ElmosclstmChecker
    from helpers import process_and_merge_elmo

    print("Elmo imported")
except ImportError:
    ElmosclstmChecker = None
    process_and_merge_elmo = None

import wandb
from benchmark import ModelBenchmark
from helpers import get_data_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert", help="Which model to use")
parser.add_argument("--finetuned", action="store_true", help="Use finetuned model")
parser.add_argument("--no_fix_spaces", action="store_true", help="Don't use fix spaces workaround for BERT")
parser.add_argument("--tokenize", action="store_true", help="Use the tokenization from NueSpell to evaluate the model.")

args = parser.parse_args()

MODEL = {"bert": {"model_name":     "subwordbert-probwordnoise",
                  "wandb_run_name": "bert-checker",
                  "model":          BertChecker(device="cuda"), },
         "elmo": {"model_name":     "elmoscrnn-probwordnoise",
                  "wandb_run_name": "elmo-checker",
                  "model":          ElmosclstmChecker(device="cuda") if ElmosclstmChecker else None}, }

name = MODEL[args.model]["wandb_run_name"] + ("-finetuned" if args.finetuned else "-pretrained") + (
        "-wo space correction" if args.no_fix_spaces else "") + ("-tokenized" if args.tokenize else "")

run = wandb.init(project="Benchmarks-" + name, name=name)

MODEL_FILES = f"checkpoints/{MODEL[args.model]['model_name']}/" + ("finetuned_model" if args.finetuned else "")
checker = MODEL[args.model]["model"]

if args.finetuned:
    checker.from_pretrained(MODEL_FILES)
else:
    checker.from_pretrained()

corrupt, clean = get_data_from_file('test')
warm_up_runs = 2
num_runs = 5

benchmark = ModelBenchmark(verbose=True)
if args.model == "bert" and args.no_fix_spaces:
    pred_func = lambda model, data: model.correct_string(data, correct_spaces=False)
else:
    pred_func = lambda model, data: model.correct_string(data)

tokenizer = None
vocab_path = None
if args.tokenize:
    if args.model == "elmo":
        tokenizer = process_and_merge_elmo
        vocab_path = os.path.join(MODEL_FILES, "vocab.pkl")
    elif args.model == "bert":
        tokenizer = process_and_merge_bert

res = benchmark.benchmark_model(checker,
                                corrupt,
                                clean,
                                name,
                                pred_func,
                                tokenizer=tokenizer,
                                vocab_path=vocab_path,
                                warm_up_runs=warm_up_runs,
                                num_runs=num_runs)
run.log(res.__dict__)

with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(name + " benchmark results:\n")
    f.write(f"{res}\n")
    f.write(f"{res.create_tex_table_perf_metrics()}\n")
    f.write(f"{res.create_tex_table_corr_metrics()}\n")

run.finish()
