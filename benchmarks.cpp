#include <benchmark/benchmark.h>

#include <cstddef>

#include "flat_storage.hpp"
#include "jagged_storage.hpp"
#include "matrixes.hpp"

namespace {

using FlatDouble = Matrix::FlatStorage<double>;
using JaggedDouble = Matrix::JaggedStorage<double>;

using FlatMatrix = Matrix::Matrix<double, FlatDouble>;
using JaggedMatrix = Matrix::Matrix<double, JaggedDouble>;

template <typename Container>
void Fill(Container& container, std::size_t size) {
    for (std::size_t row = 0; row < size; row++) {
        for (std::size_t col = 0; col < size; col++) {
            container[row, col] = static_cast<double>((row * 7 + col * 3) % 11) - 5.0;
        }
    }
}

template <typename MatrixType>
void FillDiagonallyDominant(MatrixType& matrix) {
    const std::size_t size = matrix.GetSize();

    Fill(matrix, size);
    for (std::size_t i = 0; i < size; i++) {
        matrix[i, i] = static_cast<double>(size) * 8.0;
    }
}

template <typename MatrixType>
void FillCyclicShift(MatrixType& matrix) {
    const std::size_t size = matrix.GetSize();

    for (std::size_t i = 0; i < size; i++) {
        for (std::size_t j = 0; j < size; j++) {
            matrix[i, j] = 0;
        }
    }

    for (std::size_t i = 0; i < size; i++) {
        matrix[i, (i + 1) % size] = 1;
    }
}

void SetItems(benchmark::State& state, std::size_t size) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size * size));
}

template <typename Storage>
void BM_Construct(benchmark::State& state) {
    const std::size_t size = state.range(0);

    for (auto _ : state) {
        Storage storage(size);
        benchmark::DoNotOptimize(storage);
    }

    SetItems(state, size);
}

template <typename Storage>
void BM_Copy(benchmark::State& state) {
    const std::size_t size = state.range(0);

    Storage source(size);
    Fill(source, size);

    for (auto _ : state) {
        Storage copy(source);
        benchmark::DoNotOptimize(copy);
    }

    SetItems(state, size);
}

template <typename Storage>
void BM_RowMajorSum(benchmark::State& state) {
    const std::size_t size = state.range(0);

    Storage storage(size);
    Fill(storage, size);

    for (auto _ : state) {
        double sum = 0;
        for (std::size_t row = 0; row < size; row++) {
            for (std::size_t col = 0; col < size; col++) {
                sum += storage[row, col];
            }
        }

        benchmark::DoNotOptimize(sum);
    }

    SetItems(state, size);
}

template <typename Storage>
void BM_ColumnMajorSum(benchmark::State& state) {
    const std::size_t size = state.range(0);

    Storage storage(size);
    Fill(storage, size);

    for (auto _ : state) {
        double sum = 0;
        for (std::size_t col = 0; col < size; col++) {
            for (std::size_t row = 0; row < size; row++) {
                sum += storage[row, col];
            }
        }

        benchmark::DoNotOptimize(sum);
    }

    SetItems(state, size);
}

template <typename Storage>
void BM_SwapRows(benchmark::State& state) {
    const std::size_t size = state.range(0);

    Storage storage(size);
    Fill(storage, size);

    for (auto _ : state) {
        storage.SwapRows(0, size - 1);
        benchmark::ClobberMemory();
    }
}

template <typename MatrixType>
void BM_Transpose(benchmark::State& state) {
    const std::size_t size = state.range(0);

    MatrixType matrix(size);
    Fill(matrix, size);

    for (auto _ : state) {
        auto transposed = matrix.Transpose();
        benchmark::DoNotOptimize(transposed);
    }

    SetItems(state, size);
}

template <typename MatrixType>
void BM_DeterminantDense(benchmark::State& state) {
    const std::size_t size = state.range(0);

    MatrixType matrix(size);
    FillDiagonallyDominant(matrix);

    for (auto _ : state) {
        double determinant = matrix.Determinant();
        benchmark::DoNotOptimize(determinant);
    }
}

template <typename MatrixType>
void BM_DeterminantRowSwaps(benchmark::State& state) {
    const std::size_t size = state.range(0);

    MatrixType matrix(size);
    FillCyclicShift(matrix);

    for (auto _ : state) {
        double determinant = matrix.Determinant();
        benchmark::DoNotOptimize(determinant);
    }
}

} // namespace

#define REGISTER_STORAGE_BENCHMARK(name)                                        \
    BENCHMARK_TEMPLATE(name, FlatDouble)->RangeMultiplier(4)->Range(64, 2048);  \
    BENCHMARK_TEMPLATE(name, JaggedDouble)->RangeMultiplier(4)->Range(64, 2048)

REGISTER_STORAGE_BENCHMARK(BM_Construct);
REGISTER_STORAGE_BENCHMARK(BM_Copy);
REGISTER_STORAGE_BENCHMARK(BM_RowMajorSum);
REGISTER_STORAGE_BENCHMARK(BM_ColumnMajorSum);
REGISTER_STORAGE_BENCHMARK(BM_SwapRows);

#define REGISTER_MATRIX_BENCHMARK(name, from, to)                               \
    BENCHMARK_TEMPLATE(name, FlatMatrix)->RangeMultiplier(2)->Range(from, to);  \
    BENCHMARK_TEMPLATE(name, JaggedMatrix)->RangeMultiplier(2)->Range(from, to)

REGISTER_MATRIX_BENCHMARK(BM_Transpose, 64, 1024);
REGISTER_MATRIX_BENCHMARK(BM_DeterminantDense, 64, 512);
REGISTER_MATRIX_BENCHMARK(BM_DeterminantRowSwaps, 128, 2048);

BENCHMARK_MAIN();
