import jamspell
import wandb

from benchmark import ModelBenchmark
from helpers import get_data_from_file

run = wandb.init(project="Benchmark", name="jamspell")

corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('en.bin')

corrupt, clean = get_data_from_file('test')
warm_up_runs = 2
num_runs = 5

benchmark = ModelBenchmark(device='cpu', verbose=True)
res = benchmark.benchmark_model(corrector,
                                corrupt,
                                clean,
                                "jamspell",
                                lambda model, data: model.FixFragment(data),
                                warm_up_runs=warm_up_runs,
                                num_runs=num_runs)
run.log(res.__dict__)
with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write("Jamspell benchmark results:\n")
    f.write(f"{res}\n")
    f.write(f"{res.create_tex_table_perf_metrics()}\n")
    f.write(f"{res.create_tex_table_corr_metrics()}\n")

run.finish()
