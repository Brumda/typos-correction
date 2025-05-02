import argparse

import wandb
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

from benchmark import ModelBenchmark
from helpers import get_data_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, help="What prefix to use")
parser.add_argument("--finetuned", action="store_true", help="Use finetuned model")
args = parser.parse_args()
"""
model:
    grammarly/coedit-large
prefix:
    Fix grammatical errors in this sentence
    Fix grammatical errors
    grammar
job:
    qsub -v 'model=grammarly-coedit-large' , 'prefix=grammar'  typos-correction/metacentrum/grammarly_benchmark.sh
    qsub -v 'model=grammarly-coedit-large' , 'prefix=Fix grammatical errors'  typos-correction/metacentrum/grammarly_benchmark.sh
    qsub -v 'model=grammarly-coedit-large-finetuned' , 'prefix=Fix grammatical errors'  typos-correction/metacentrum/grammarly_benchmark.sh
"""
model_name = "grammarly-coedit-large-finetuned" if args.finetuned else "grammarly/coedit-large"
CHECKPOINTS = "./checkpoints/"
name = model_name.replace("/", "-")
run = wandb.init(project=f"Benchmark-{name}", name=name)
print(f"Model used: {model_name}")

if args.finetuned:
    print("Fine-tuned model")
    model_path = CHECKPOINTS + name
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path,
                                                       local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
    corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
else:
    print("Pre-trained model")
    corrector = pipeline("text2text-generation", model=model_name)

pred_func = lambda model, text: model(f"{args.prefix}: {text}")[0]['generated_text']

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
