import wandb
from benchmark import ModelBenchmark
from helpers import get_data_from_file
import jamspell

wandb.init(project="test_Benchmarks", name="jamspell", resume="allow", id="jamspell",
           config={'GPU': 'CPU'})

corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('en.bin')

benchmark = ModelBenchmark(device='cpu')

clean, corrupt = get_data_from_file('small')
warm_up_runs = 2
num_runs = 5
res = benchmark.benchmark_model(corrector,
                                corrupt,
                                clean,
                                "jamspell",
                                lambda model, data: model.FixFragment(data),
                                warm_up_runs=warm_up_runs,
                                num_runs=num_runs)
wandb.log(res.__dict__)
with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write("Jamspell benchmark results:\n")
    f.write(f"{res}")

wandb.finish()
