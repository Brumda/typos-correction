import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import psutil

from detect_typo_model import TypoDetectionModel

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
    ram_memory_mb: float
    peak_ram_memory_mb: float
    gpu_memory_mb: float
    throughput_sentences: float
    throughput_words: float
    ms_per_sentence: float
    accuracy_words: float
    accuracy_sentences: float
    corr2corr: float
    corr2incorr: float
    incorr2corr: float
    incorr2incorr: float
    recall: float
    precision: float
    f05: float
    word_correction_rate: float
    word_incorrection_rate: float  # for lack of a better name

    typo_detection_model_inference_time: float
    typo_detection_model_ms_per_sentence: float
    skipped: int

    def __str__(self):
        return (f"Benchmark results:\n"
                f"   Model: {self.model_name}\n"
                f"   Size: {self.model_size:.2f} MB\n"
                f"   Inference Time: {self.inference_time:.2f} s\n"
                f"   GPU Memory: {self.gpu_memory_mb:.2f} MB\n"
                f"   RAM Memory: {self.ram_memory_mb:.2f} MB\n"
                f"   Peak RAM Memory: {self.peak_ram_memory_mb:.2f} MB\n"
                f"   Throughput: {self.throughput_sentences:.2f} sentences/sec\n"
                f"   Throughput: {self.ms_per_sentence:.2f} ms/sentence\n"
                f"   Throughput: {self.throughput_words:.2f} words/sec\n"
                f"   Accuracy sentences: {self.accuracy_sentences:.2%}\n"
                f"   Accuracy words: {self.accuracy_words:.2%}\n"
                f"   Correct → Correct: {self.corr2corr}\n"
                f"   Correct → Incorrect: {self.corr2incorr}\n"
                f"   Incorrect → Correct: {self.incorr2corr}\n"
                f"   Incorrect → Incorrect: {self.incorr2incorr}\n"
                f"   Recall: {self.recall:.2f}\n"
                f"   Precision: {self.precision:.2f}\n"
                f"   F0.5: {self.f05:.2f}\n"
                f"   Word Correction Rate: {self.word_correction_rate:.2f}\n"
                f"   Word Incorrection Rate: {self.word_incorrection_rate:.2f}\n"
                f"   Inference Time typo detection: {self.typo_detection_model_inference_time:.2f} s\n"
                f"   Throughput typo detection: {self.typo_detection_model_ms_per_sentence:.2f} ms/sentence\n"
                f"   Skipped sentences: {self.skipped:.2f}\n"
                )

    def __repr__(self):
        return self.__str__()

    def create_tex_table_perf_metrics(self):
        return f"""
        \\begin{{table}}[h]
           \\centering
           \\begin{{tabular}}{{@{{}}lr@{{}}}}
                \\toprule
                \\multicolumn{{2}}{{c}}{{Model statistics}} \\\\
                \\midrule
                Model size & \\num{{{self.model_size:.2f}}} MB \\\\
                RAM Memory & \\num{{{self.ram_memory_mb:.2f}}} MB \\\\
                Peak RAM Memory & \\num{{{self.peak_ram_memory_mb:.2f}}} MB \\\\
                GPU Memory & \\num{{{self.gpu_memory_mb:.2f}}} MB \\\\
                \\midrule
                Total Inference Time & \\num{{{self.inference_time:.2f}}} s \\\\
                Throughput (sentences/sec) & \\num{{{self.throughput_sentences:.2f}}} \\\\
                Throughput (ms/sentence) & \\num{{{self.ms_per_sentence:.2f}}} \\\\
                Throughput (words/sec) & \\num{{{self.throughput_words:.2f}}} \\\\
                \\bottomrule
            \\end{{tabular}}
            \\caption{{Performance metrics for the {self.model_name} model.}}
            \\label{{tab:{self.model_name}_metrics}}
        \\end{{table}}\n"""

    def create_tex_table_corr_metrics(self):
        return f"""
        \\begin{{table}}[h]
           \\centering
           \\begin{{tabular}}{{@{{}}lr@{{}}}}
                \\toprule
                \\multicolumn{{2}}{{c}}{{Word corrections}} \\\\
                \\midrule
                Accuracy (sentences) & \\num{{{self.accuracy_sentences:.2%}}} \\\\
                Accuracy (words) & \\num{{{self.accuracy_words:.2%}}} \\\\
                \\midrule
                Correct $\\rightarrow$ Correct & \\num{{{self.corr2corr}}} \\\\
                Correct $\\rightarrow$ Incorrect & \\num{{{self.corr2incorr}}} \\\\
                Incorrect $\\rightarrow$ Correct & \\num{{{self.incorr2corr}}} \\\\
                Incorrect $\\rightarrow$ Incorrect & \\num{{{self.incorr2incorr}}} \\\\
                \\midrule
                Precision (words) & \\num{{{self.precision:.2f}}} \\\\
                Recall (words) & \\num{{{self.recall:.2f}}} \\\\
                F0.5 (words) & \\num{{{self.f05:.2f}}} \\\\
                Word Correction Rate & \\num{{{self.word_correction_rate:.2f}}} \\\\
                Word Incorrection Rate & \\num{{{self.word_incorrection_rate:.2f}}} \\\\
                \\bottomrule
            \\end{{tabular}}
            \\caption{{Correction metrics for the {self.model_name} model.}}
            \\label{{tab:{self.model_name}_metrics}}
        \\end{{table}}\n"""


class ModelBenchmark:
    def __init__(self, path: str = "detect_typo_models/best_model.pt", device: str = 'cuda', verbose: bool = False):
        self.device = device
        self.peak_ram = 0
        self.verbose = verbose
        self.process = psutil.Process(os.getpid())
        self.predict_typo = TypoDetectionModel()
        self.predict_typo.load_model(path)

    @contextmanager
    def _measure_memory(self):
        ram_start = self._get_ram_usage()
        try:
            yield
        finally:
            ram_end = self._get_ram_usage()
            ram_used = ram_end - ram_start
            self.peak_ram = max(self.peak_ram, ram_used)

    def _get_ram_usage(self):
        """Get current RAM usage in MB"""
        self.process.memory_info()  # Refresh memory info
        return self.process.memory_info().rss / 1024**2

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0

    def _clear_memory(self):
        """Clear memory and caches"""
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _get_model_size(self, model) -> float:
        """Get model size in MB (PyTorch models only)"""
        if self.device == 'cuda':
            param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.model.buffers())
            total_size_mb = (param_size + buffer_size) / 1024**2
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

        if self.verbose: print(f"Starting {warm_up_runs} warm-up iterations for {model_name}...")
        start = time.time()
        for _ in range(warm_up_runs):
            for text in corrupt_texts[:len(corrupt_texts) // 3]:
                _ = predict(model, text)
        if self.verbose: print(f"Finished warm-up after {time.time() - start} seconds.")

        inference_times = []
        throughputs_words = []
        throughputs_sentences = []
        ram_usages = []
        gpu_memory_usages = []
        accuracies_words = []
        accuracies_sentences = []
        word_correction = []
        word_correction_rates = []
        word_incorrection_rates = []
        precisions = []
        recalls = []
        f05s = []
        ms_per_sentences = []
        ms_per_sentences_typo_detect = []
        inference_times_typo_detect = []

        if self.verbose: print(f"Starting benchmark iterations...")
        # for run in tqdm(range(num_runs)):
        for run in range(num_runs):
            self._clear_memory()
            acc_sen = 0
            inference_time = 0
            inference_time_typo_detect = 0
            ram_usage = 0
            corr2corr, corr2incorr, incorr2corr, incorr2incorr = 0, 0, 0, 0
            skipped = 0

            with self._measure_memory():
                # for corrupt, clean in tqdm(zip(corrupt_texts, clean_texts)):
                for corrupt, clean in zip(corrupt_texts, clean_texts):
                    # decide if sentence should be fixed or if it's not a typo
                    start_time_detect = time.time()
                    typo_prob = self.predict_typo.predict(corrupt)
                    inference_time_typo_detect += time.time() - start_time_detect

                    if typo_prob > 0.5:
                        # prediction
                        ram_before = self._get_ram_usage()
                        start_time = time.time()
                        prediction = predict(model, corrupt)
                        inference_time += time.time() - start_time
                        ram_usage += self._get_ram_usage() - ram_before
                    else:
                        skipped += 1
                        continue
                    # statistics
                    acc_sen += (prediction == clean)
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

                ####################################
                # bare word statistics
                ####################################
                if self.verbose: print(
                        f"corr2corr: {corr2corr}, corr2incorr: {corr2incorr}, incorr2corr: {incorr2corr},"
                        f" incorr2incorr: {incorr2incorr}\n"
                        f"skipped: {skipped}")

                total_words = corr2corr + corr2incorr + incorr2corr + incorr2incorr
                word_correction.append((corr2corr, corr2incorr, incorr2corr, incorr2incorr))
                accuracies_words.append((corr2corr + incorr2corr) / total_words)
                word_correction_rates.append(incorr2corr / (incorr2corr + incorr2incorr))
                word_incorrection_rates.append(corr2incorr / (corr2incorr + corr2corr))

                ####################################
                # more convoluted statistics
                ####################################
                precisions.append(incorr2corr / (incorr2corr + corr2incorr))
                recalls.append(incorr2corr / (incorr2corr + incorr2incorr))
                f05s.append((1.25 * precisions[-1] * recalls[-1]) / (0.25 * precisions[-1] + recalls[-1]))
                accuracies_sentences.append(acc_sen / (len(clean_texts) - skipped))

                ####################################
                # time based statistics
                ####################################
                throughputs_words.append(total_words / inference_time)
                throughputs_sentences.append((len(clean_texts) - skipped) / inference_time)
                ms_per_sentences.append((inference_time / (len(clean_texts) - skipped)) * 1000)
                inference_times.append(inference_time)

                # typo detection model
                ms_per_sentences_typo_detect.append((inference_time_typo_detect / len(clean_texts)) * 1000)
                inference_times_typo_detect.append(inference_time_typo_detect)

                ####################################
                # memory based statistics
                ####################################
                ram_usages.append(ram_usage / (len(clean_texts) - skipped))
                gpu_memory_usages.append(self._get_gpu_memory())

            if self.verbose: print(f"Finished {run + 1}/{num_runs} iteration in {inference_time} seconds.")

        avg_word_correction = np.mean(word_correction, axis=0)
        return BenchmarkResult(model_name=model_name,
                               model_size=self._get_model_size(model),
                               inference_time=np.mean(inference_times),
                               gpu_memory_mb=np.mean(gpu_memory_usages),
                               throughput_words=np.mean(throughputs_words),
                               throughput_sentences=np.mean(throughputs_sentences),
                               ms_per_sentence=np.mean(ms_per_sentences),
                               accuracy_words=np.mean(accuracies_words),
                               accuracy_sentences=np.mean(accuracies_sentences),
                               corr2corr=avg_word_correction[0],
                               corr2incorr=avg_word_correction[1],
                               incorr2corr=avg_word_correction[2],
                               incorr2incorr=avg_word_correction[3],
                               word_correction_rate=np.mean(word_correction_rates),
                               word_incorrection_rate=np.mean(word_incorrection_rates),
                               precision=np.mean(precisions),
                               recall=np.mean(recalls),
                               f05=np.mean(f05s),
                               ram_memory_mb=np.mean(ram_usages),
                               peak_ram_memory_mb=self.peak_ram,
                               typo_detection_model_inference_time=np.mean(inference_times_typo_detect),
                               typo_detection_model_ms_per_sentence=np.mean(ms_per_sentences_typo_detect),
                               skipped=skipped,
                               )
