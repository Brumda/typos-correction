import torch
from neuspell import BertChecker

import wandb
from benchmark import ModelBenchmark
from helpers import get_data_from_file

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

wandb.init(project="test_benchmark", name="test", resume="allow", id="test",
           config={'GPU': gpu_name, })

checker = BertChecker(device='cuda')
checker.from_pretrained()

benchmark = ModelBenchmark()

clean, corrupt = get_data_from_file('small')

res = benchmark.benchmark_model(checker,
                                corrupt,
                                clean,
                                "neuspell-bert",
                                lambda model, data: model.correct_string(data),
                                warm_up_runs=0,
                                num_runs=10)
wandb.log(res.__dict__)
with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write("Pretrained BERT benchmark results:\n")
    f.write(f"{res}")

wandb.finish()
