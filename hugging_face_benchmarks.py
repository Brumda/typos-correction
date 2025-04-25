import argparse

import wandb
from transformers import pipeline

from benchmark import ModelBenchmark
from helpers import get_data_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Which model to use")

# models:
# prithivida/grammar_error_correcter_v1
# pszemraj/bart-base-grammar-synthesis
# oliverguhr/spelling-correction-english-base
# grammarly/coedit-large

args = parser.parse_args()
run = wandb.init(project=f"Benchmarks-{args.model}", name=f"{args.model}")

print(f"Model used: {args.model}")

grammar_corrector = pipeline("text2text-generation", model=args.model)

if args.model == "grammarly/coedit-large":
    pred_func = lambda model, text: model(f"Fix grammatical errors in this sentence: {text}")[0]['generated_text']
else:
    pred_func = lambda model, text: model(text)[0]['generated_text']

corrupt, clean = get_data_from_file('test')
benchmark = ModelBenchmark(verbose=True)
res = benchmark.benchmark_model(grammar_corrector,
                                corrupt,
                                clean,
                                f"{args.model}",
                                pred_func, )

run.log(res.__dict__)
with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(f"{args.model} benchmark results:\n")
    f.write(f"{res}\n")
    f.write(f"{res.create_tex_table_perf_metrics()}\n")
    f.write(f"{res.create_tex_table_corr_metrics()}\n")

run.finish()
