import argparse

import wandb
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer

from benchmark import ModelBenchmark
from helpers import get_data_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Which model to use")
parser.add_argument("--finetuned", action="store_true", help="Use finetuned model")
args = parser.parse_args()
"""
models:
pszemraj/bart-base-grammar-synthesis
oliverguhr/spelling-correction-english-base

prithivida/grammar_error_correcter_v1

qsub -N prithivida-grammar_error_correcter_v1-finetuned -v 'model=prithivida-grammar_error_correcter_v1-finetuned' typos-correction/metacentrum/hugging_face_benchmarks.sh
qsub -N pszemraj-bart-base-grammar-synthesis-finetuned -v 'model=pszemraj-bart-base-grammar-synthesis-finetuned' typos-correction/metacentrum/hugging_face_benchmarks.sh
qsub -N oliverguhr-spelling-correction-english-base-finetuned -v 'model=oliverguhr-spelling-correction-english-base-finetuned' typos-correction/metacentrum/hugging_face_benchmarks.sh
"""

models = {"T5":
              ["prithivida-grammar_error_correcter_v1-finetuned", ],
          "BART":
              ["pszemraj-bart-base-grammar-synthesis-finetuned",
               "oliverguhr-spelling-correction-english-base-finetuned", ],
          }

CHECKPOINTS = "./checkpoints/"
name = args.model.replace("/", "-")
run = wandb.init(project=f"Benchmark-{name}", name=name)
print(f"Model used: {args.model}")

if args.finetuned:
    model_path = CHECKPOINTS + name
    if args.model in models["T5"]:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path,
                                                           local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)

    elif args.model in models["BART"]:
        model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path,
                                                             local_files_only=True)
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
    else:
        raise ValueError("Model not found")

    corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
else:
    corrector = pipeline("text2text-generation", model=args.model)

pred_func = lambda model, text: model(text)[0]['generated_text']

corrupt, clean = get_data_from_file('test')
benchmark = ModelBenchmark(verbose=True)
res = benchmark.benchmark_model(corrector,
                                corrupt,
                                clean,
                                f"{name}",
                                pred_func, )

run.log(res.__dict__)
with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(f"{name} benchmark results:\n")
    f.write(f"{res}\n")
    f.write(f"{res.create_tex_table_perf_metrics()}\n")
    f.write(f"{res.create_tex_table_corr_metrics()}\n")

run.finish()
