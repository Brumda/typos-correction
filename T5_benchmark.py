import wandb
from happytransformer import HappyTextToText

from benchmark import ModelBenchmark
from helpers import get_data_from_file

run = wandb.init(project="Benchmarks-T5", name="T5")
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

corrupt, clean = get_data_from_file('test')
warm_up_runs = 2
num_runs = 5

benchmark = ModelBenchmark(verbose=True)
res = benchmark.benchmark_model(happy_tt,
                                corrupt,
                                clean,
                                "T5",
                                lambda model, data: model.generate_text(f"grammar: {data}").text,
                                warm_up_runs=warm_up_runs,
                                num_runs=num_runs)
run.log(res.__dict__)
with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write("T5 benchmark results:\n")
    f.write(f"{res}\n")
    f.write(f"{res.create_tex_table_perf_metrics()}\n")
    f.write(f"{res.create_tex_table_corr_metrics()}\n")

run.finish()
