import argparse

import wandb
from happytransformer import HappyTextToText

from benchmark import ModelBenchmark
from helpers import get_data_from_file

"""
qsub -v 'model=T5-Vennify-normal' typos-correction/metacentrum/t5_benchmark_job.sh
qsub -v 'model=T5-Vennify-finetuned' typos-correction/metacentrum/t5_benchmark_job.sh
"""

parser = argparse.ArgumentParser()
parser.add_argument("--finetuned", action="store_true", help="Use finetuned model")
args = parser.parse_args()

run = wandb.init(project="Benchmarks-T5", name="T5" + "-finetuned" if args.finetuned else "")

if args.finetuned:
    print("Fine-tuned model")
    model_path = "./checkpoints/vennify-t5-base-grammar-correction-finetuned"
    happy_tt = HappyTextToText(model_type="T5", model_name=model_path)
else:
    print("Pre-trained model")
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

corrupt, clean = get_data_from_file('test')

benchmark = ModelBenchmark(verbose=True)
res = benchmark.benchmark_model(happy_tt,
                                corrupt,
                                clean,
                                "T5 Vennify" + "-finetuned" if args.finetuned else "",
                                lambda model, data: model.generate_text(f"grammar: {data}").text, )
run.log(res.__dict__)
with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write("T5" + "-finetuned" if args.finetuned else "" + " benchmark results:\n")
    f.write(f"{res}\n")
    f.write(f"{res.create_tex_table_perf_metrics()}\n")
    f.write(f"{res.create_tex_table_corr_metrics()}\n")

run.finish()
