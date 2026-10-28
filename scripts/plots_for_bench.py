#!/usr/bin/env python3

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

TIME_UNITS_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

TITLES = {
    "BM_Construct": "Создание",
    "BM_Copy": "Копирование",
    "BM_RowMajorSum": "Обход по строкам",
    "BM_ColumnMajorSum": "Обход по столбцам",
    "BM_SwapRows": "SwapRows",
    "BM_Transpose": "Transpose",
    "BM_DeterminantDense": "Определитель (плотная матрица)",
    "BM_DeterminantRowSwaps": "Определитель (много перестановок строк)",
}

STORAGES = {"Flat": "FlatStorage", "Jagged": "JaggedStorage"}
COLORS = {"Flat": "tab:blue", "Jagged": "tab:orange"}

NAME_PATTERN = re.compile(r"^(BM_\w+)<(\w+)>/(\d+)$")


def load_results(path):
    with open(path) as file:
        data = json.load(file)

    has_aggregates = any(item["run_type"] == "aggregate" for item in data["benchmarks"])
    results = defaultdict(lambda: defaultdict(dict))

    for item in data["benchmarks"]:
        if has_aggregates and item.get("aggregate_name") != "median":
            continue

        match = NAME_PATTERN.match(item["run_name"])
        if match is None:
            continue

        benchmark, type_name, size = match.group(1), match.group(2), int(match.group(3))
        storage = "Flat" if type_name.startswith("Flat") else "Jagged"
        results[benchmark][storage][size] = item["real_time"] * TIME_UNITS_TO_NS[item["time_unit"]]

    return results


def humanize(time_ns):
    for unit, factor in (("с", 1e9), ("мс", 1e6), ("мкс", 1e3)):
        if time_ns >= factor:
            return f"{time_ns / factor:g} {unit}"
    return f"{time_ns:g} нс"


def plot_times(results, output_path):
    benchmarks = [name for name in TITLES if name in results]
    columns = 3
    rows = (len(benchmarks) + columns - 1) // columns

    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.8 * rows), squeeze=False)

    for axis, benchmark in zip(axes.flat, benchmarks):
        for storage, series in sorted(results[benchmark].items()):
            sizes = sorted(series)
            axis.plot(sizes, [series[size] for size in sizes], marker="o",
                      label=STORAGES[storage], color=COLORS[storage])

        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(TITLES[benchmark])
        axis.set_xlabel("Размер матрицы n")
        axis.set_ylabel("Время, нс")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    for axis in axes.flat[len(benchmarks):]:
        axis.axis("off")

    figure.suptitle("FlatStorage vs JaggedStorage: время работы (меньше - лучше)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


RATIO_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0,
               10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]


def plot_ratio(results, output_path):
    benchmarks = [name for name in TITLES if name in results]
    columns = 3
    rows = (len(benchmarks) + columns - 1) // columns

    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.8 * rows), squeeze=False)

    for axis, benchmark in zip(axes.flat, benchmarks):
        flat = results[benchmark].get("Flat", {})
        jagged = results[benchmark].get("Jagged", {})
        sizes = sorted(set(flat) & set(jagged))
        ratios = [flat[size] / jagged[size] for size in sizes]

        low = min(min(ratios) / 1.4, 0.8)
        high = max(max(ratios) * 1.4, 1.25)
        axis.set_ylim(low, high)
        axis.set_yscale("log")

        axis.axhspan(1.0, high, color="tab:orange", alpha=0.08)
        axis.axhspan(low, 1.0, color="tab:blue", alpha=0.08)
        axis.axhline(1.0, color="black", linewidth=1)

        axis.plot(sizes, ratios, marker="o", color="tab:green")
        for size, ratio in zip(sizes, ratios):
            axis.annotate(f"{ratio:.2f}x", (size, ratio), textcoords="offset points",
                          xytext=(0, 7 if ratio >= 1 else -13), ha="center", fontsize=8)

        ticks = []
        for tick in sorted((t for t in RATIO_TICKS if low <= t <= high), key=lambda t: abs(math.log(t))):
            if all(abs(math.log(tick / other)) >= 0.3 for other in ticks):
                ticks.append(tick)
        ticks.sort()
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{tick:g}x" for tick in ticks])
        axis.minorticks_off()

        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.set_title(TITLES[benchmark])
        axis.set_xlabel("Размер матрицы n")
        axis.set_ylabel("Время Flat / время Jagged")
        axis.grid(True, alpha=0.3)

    for axis in axes.flat[len(benchmarks):]:
        axis.axis("off")

    figure.suptitle("Отношение времени Flat / Jagged: выше 1 (оранжевая зона) - Jagged быстрее, "
                    "ниже 1 (синяя зона) - Flat быстрее")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(sys.argv[1])
    plot_times(results, output_dir / "bench_times.png")
    plot_ratio(results, output_dir / "bench_ratio.png")


if __name__ == "__main__":
    main()
