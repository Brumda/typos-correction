import gc
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from helpers import tilde_format

# this way I don't need to install torch with non torch models
try:
    import torch
except ImportError:
    pass


# from tqdm import tqdm


@dataclass
class BenchmarkResult:
    model_name: str
    model_size: float
    inference_time: float
    peak_memory_mb: float
    gpu_memory_mb: float
    throughput_sentences: float
    throughput_tokens: float
    ms_per_sentence: float
    accuracy_tokens: float
    accuracy_sentences: float
    corr2corr: float
    corr2incorr: float
    incorr2corr: float
    incorr2incorr: float
    token_correction_rate: float
    token_incorrection_rate: float  # for lack of a better name

    def __str__(self):
        return (f"Benchmark results:\n"
                f"   Model: {self.model_name}\n"
                f"   Size: {self.model_size:.2f} MB\n"
                f"   Inference Time: {self.inference_time:.2f} s\n"
                f"   Peak Memory: {self.peak_memory_mb:.2f} MB\n"
                f"   GPU Memory: {self.gpu_memory_mb:.2f} MB\n"
                f"   Throughput: {self.throughput_sentences:.2f} sentences/sec\n"
                f"   Throughput: {self.ms_per_sentence:.2f} ms/sentence\n"
                f"   Throughput: {self.throughput_tokens:.2f} tokens/sec\n"
                f"   Accuracy sentences: {self.accuracy_sentences:.2%}\n"
                f"   Accuracy tokens: {self.accuracy_tokens:.2%}\n"
                f"   Correct → Correct: {self.corr2corr}\n"
                f"   Correct → Incorrect: {self.corr2incorr}\n"
                f"   Incorrect → Correct: {self.incorr2corr}\n"
                f"   Incorrect → Incorrect: {self.incorr2incorr}\n"
                f"   Token Correction Rate: {self.token_correction_rate:.2%}\n"
                f"   Token Incorrection Rate: {self.token_incorrection_rate:.2%}\n")

    def __repr__(self):
        return self.__str__()

    def create_tex_table_perf_metrics(self):
        return f"""
        \\begin{{table}}[h]
           \centering
           \\begin{{tabular}}{{@{{}}lr@{{}}}}
                \\toprule
                \\multicolumn{{2}}{{c}}{{Model statistics}} \\\\
                \\midrule
                Model size & {self.model_size:.2f} MB \\\\
                Peak Memory & {self.peak_memory_mb:.2f} MB \\\\
                GPU Memory & {self.gpu_memory_mb:.2f} MB \\\\
                \\midrule
                Total Inference Time & {self.inference_time:.2f} s \\\\
                Throughput (sentences/sec) & {self.throughput_sentences:.2f} \\\\
                Throughput (ms/sentence) & {self.ms_per_sentence:.2f} \\\\
                Throughput (tokens/sec) & {self.throughput_tokens:.2f} \\\\
                \\bottomrule
            \\end{{tabular}}
            \\caption{{Performance metrics for the {self.model_name} model.}}
            \\label{{tab:{self.model_name}_metrics}}
        \\end{{table}}\n"""

    def create_tex_table_corr_metrics(self):
        return f"""
        \\begin{{table}}[h]
           \centering
           \\begin{{tabular}}{{@{{}}lr@{{}}}}
                \\toprule
                \\multicolumn{{2}}{{c}}{{Token corrections}} \\\\
                \\midrule
                Accuracy (sentences) & {self.accuracy_sentences:.2%} \\\\
                Accuracy (tokens) & {self.accuracy_tokens:.2%} \\\\
                \\midrule
                Correct $\\rightarrow$ Correct & {tilde_format(self.corr2corr)} \\\\
                Correct $\\rightarrow$ Incorrect & {tilde_format(self.corr2incorr)} \\\\
                Incorrect $\\rightarrow$ Correct & {tilde_format(self.incorr2corr)} \\\\
                Incorrect $\\rightarrow$ Incorrect & {tilde_format(self.incorr2incorr)} \\\\
                \\midrule
                Token Correction Rate & {self.token_correction_rate:.2f} \\\\
                Token Incorrection Rate & {self.token_incorrection_rate:.2f} \\\\
                \\bottomrule
            \\end{{tabular}}
            \\caption{{Correction metrics for the {self.model_name} model.}}
            \\label{{tab:{self.model_name}_metrics}}
        \\end{{table}}\n"""


class ModelBenchmark:
    def __init__(self, device: str = 'cuda', prints: bool = False):
        self.device = device
        self.peak_memory = 0
        self.prints = prints

    @contextmanager
    def _measure_memory(self):
        tracemalloc.start()
        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.peak_memory = peak / 1024 ** 2

    def _get_gpu_memory(self) -> float:
        if self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        return 0

    def _clear_memory(self):
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _get_model_size(self, model) -> float:
        """Pytorch models only"""
        if self.device == 'cuda':
            param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.model.buffers())
            total_size_mb = (param_size + buffer_size) / 1024 ** 2
            return total_size_mb
        return 0

    def benchmark_model(self,
                        model,
                        corrupt_texts: list[str],
                        clean_texts: list[str],
                        model_name: str,
                        predict: Callable[[Any, str], str],
                        warm_up_runs: int = 3,
                        num_runs: int = 5
                        ) -> BenchmarkResult:

        if self.prints: print(f"Starting {warm_up_runs} warm-up iterations for {model_name}...")
        start = time.time()
        for _ in range(warm_up_runs):
            for text in corrupt_texts[:len(corrupt_texts) // 3]:
                _ = predict(model, text)
        if self.prints: print(f"Finished warm-up after {time.time() - start} seconds.")

        inference_times = []
        throughputs_tokens = []
        throughputs_sentences = []
        memory_usages = []
        gpu_memory_usages = []
        accuracies_tokens = []
        accuracies_sentences = []
        token_correction = []
        token_correction_rates = []
        token_incorrection_rates = []
        ms_per_sentences = []

        if self.prints: print(f"Starting benchmark iterations...")
        # for run in tqdm(range(num_runs)):
        for run in range(num_runs):
            self._clear_memory()
            with self._measure_memory():
                acc_sen = 0
                inference_time = 0
                corr2corr, corr2incorr, incorr2corr, incorr2incorr = 0, 0, 0, 0
                # for corrupt, clean in tqdm(zip(corrupt_texts, clean_texts)):
                for corrupt, clean in zip(corrupt_texts, clean_texts):
                    # prediction
                    start_time = time.time()
                    prediction = predict(model, corrupt)
                    inference_time += time.time() - start_time

                    # statistics
                    acc_sen += prediction == clean
                    for corrupt_token, clean_token, predict_token in zip(corrupt.split(), clean.split(),
                                                                         prediction.split()):
                        if corrupt_token == clean_token and predict_token == clean_token:
                            corr2corr += 1
                        elif corrupt_token == clean_token and predict_token != clean_token:
                            corr2incorr += 1
                        elif corrupt_token != clean_token and predict_token == clean_token:
                            incorr2corr += 1
                        elif corrupt_token != clean_token and predict_token != clean_token:
                            incorr2incorr += 1

                total_tokens = corr2corr + corr2incorr + incorr2corr + incorr2incorr
                token_correction.append((corr2corr, corr2incorr, incorr2corr, incorr2incorr))
                accuracies_tokens.append((corr2corr + incorr2corr) / total_tokens)
                token_correction_rates.append(incorr2corr / (incorr2corr + incorr2incorr))
                token_incorrection_rates.append(corr2incorr / (corr2incorr + corr2corr))

                accuracies_sentences.append(acc_sen / len(clean_texts))

                throughputs_tokens.append(total_tokens / inference_time)
                throughputs_sentences.append(len(corrupt_texts) / inference_time)
                ms_per_sentences.append((inference_time / len(clean_texts)) * 1000)

                inference_times.append(inference_time)
                memory_usages.append(self.peak_memory)
                gpu_memory_usages.append(self._get_gpu_memory())

            if self.prints: print(f"Finished {run + 1}/{num_runs} iteration in {inference_time} seconds.")

        avg_accuracy_tok = np.mean(accuracies_tokens)
        avg_accuracy_sen = np.mean(accuracies_sentences)
        avg_inference_time = np.mean(inference_times)
        avg_throughput_tok = np.mean(throughputs_tokens)
        avg_throughput_sen = np.mean(throughputs_sentences)
        avg_memory = np.mean(memory_usages)
        avg_gpu_memory = np.mean(gpu_memory_usages)
        avg_token_correction = np.mean(token_correction, axis=0)
        avg_token_correction_rate = np.mean(token_correction_rates)
        avg_token_incorrection_rate = np.mean(token_incorrection_rates)
        avg_ms_per_sentence = np.mean(ms_per_sentences)

        return BenchmarkResult(model_name=model_name,
                               model_size=self._get_model_size(model),
                               inference_time=avg_inference_time,
                               peak_memory_mb=avg_memory,
                               gpu_memory_mb=avg_gpu_memory,
                               throughput_tokens=avg_throughput_tok,
                               throughput_sentences=avg_throughput_sen,
                               ms_per_sentence=avg_ms_per_sentence,
                               accuracy_tokens=avg_accuracy_tok,
                               accuracy_sentences=avg_accuracy_sen,
                               corr2corr=avg_token_correction[0],
                               corr2incorr=avg_token_correction[1],
                               incorr2corr=avg_token_correction[2],
                               incorr2incorr=avg_token_correction[3],
                               token_correction_rate=avg_token_correction_rate,
                               token_incorrection_rate=avg_token_incorrection_rate)
