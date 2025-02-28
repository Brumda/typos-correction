from neuspell import BertChecker

from benchmark import ModelBenchmark
from helpers import get_data_from_file

checker = BertChecker(device='cuda')
checker.from_pretrained()

benchmark = ModelBenchmark()

clean, corrupt = get_data_from_file('test')

res = benchmark.benchmark_model(checker,
                                corrupt,
                                clean,
                                "neuspell-bert",
                                lambda model, data: model.correct_string(data),
                                warm_up_runs=3,
                                num_runs=10)

with open("benchmark_results.txt", "w") as f:
    f.write("Pretrained BERT benchmark results:\n")
    f.write(f"{res}")
