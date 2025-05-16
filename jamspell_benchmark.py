import jamspell
import wandb

from benchmark import ModelBenchmark
from helpers import get_data_from_file

benchmark_flag = True
corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('en.bin')

corrupt, clean = get_data_from_file('test')
benchmark = ModelBenchmark(device='cpu', verbose=True)
pred_func = lambda model, data: model.FixFragment(data)

if benchmark_flag:
    run = wandb.init(project="Benchmarks-jamspell", name="jamspell")

    res = benchmark.benchmark_model(corrector,
                                    corrupt,
                                    clean,
                                    "jamspell",
                                    pred_func, )
    run.log(res.__dict__)
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("Jamspell benchmark results:\n")
        f.write(f"{res}\n")
        f.write(f"{res.create_tex_table_perf_metrics()}\n")
        f.write(f"{res.create_tex_table_corr_metrics()}\n")

    run.finish()
else:
    benchmark.get_wrong_words(corrector, corrupt, clean, pred_func, start_idx=666, num_sen=30)
