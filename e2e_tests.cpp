#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <list>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flat_storage.hpp"
#include "jagged_storage.hpp"
#include "matrixes.hpp"

namespace {

constexpr std::size_t kLargeSize = 1000;
constexpr std::size_t kLargeDenseFloatingSize = 300;
constexpr std::size_t kLargeDenseIntegralSize = 200;

template <typename T>
using Flat = Matrix::FlatStorage<T>;

template <typename T>
using Jagged = Matrix::JaggedStorage<T>;

template <typename T, typename Storage>
struct MatrixConfig {
    using Value = T;
    using MatrixType = Matrix::Matrix<T, Storage>;
};

using IntegralConfigs = ::testing::Types<
    MatrixConfig<int, Flat<int>>,
    MatrixConfig<int, Jagged<int>>,
    MatrixConfig<long long, Flat<long long>>,
    MatrixConfig<long long, Jagged<long long>>>;

using FloatingConfigs = ::testing::Types<
    MatrixConfig<float, Flat<float>>,
    MatrixConfig<float, Jagged<float>>,
    MatrixConfig<double, Flat<double>>,
    MatrixConfig<double, Jagged<double>>>;

using DoubleConfigs = ::testing::Types<
    MatrixConfig<double, Flat<double>>,
    MatrixConfig<double, Jagged<double>>>;

using AllConfigs = ::testing::Types<
    MatrixConfig<int, Flat<int>>,
    MatrixConfig<int, Jagged<int>>,
    MatrixConfig<long long, Flat<long long>>,
    MatrixConfig<long long, Jagged<long long>>,
    MatrixConfig<float, Flat<float>>,
    MatrixConfig<float, Jagged<float>>,
    MatrixConfig<double, Flat<double>>,
    MatrixConfig<double, Jagged<double>>>;

using FlatIntMatrix = Matrix::Matrix<int, Flat<int>>;
using JaggedIntMatrix = Matrix::Matrix<int, Jagged<int>>;
using FlatDoubleMatrix = Matrix::Matrix<double, Flat<double>>;
using JaggedDoubleMatrix = Matrix::Matrix<double, Jagged<double>>;

template <typename Config>
class MatrixTestBase : public ::testing::Test {
public:
    using Value = typename Config::Value;
    using MatrixType = typename Config::MatrixType;

    static MatrixType Identity(std::size_t size) {
        return MatrixType(size);
    }

    static MatrixType Zero(std::size_t size) {
        const std::vector<Value> no_values;
        return MatrixType(size, no_values.begin(), no_values.end());
    }

    static MatrixType Make(std::size_t size, std::initializer_list<Value> values) {
        return MatrixType(size, values.begin(), values.end());
    }

    template <typename Range>
    static MatrixType FromRange(std::size_t size, const Range& range) {
        return MatrixType(size, std::begin(range), std::end(range));
    }

    static MatrixType Sequential(std::size_t size) {
        std::vector<Value> values(size * size);
        std::iota(values.begin(), values.end(), Value{0});

        return MatrixType(size, values.begin(), values.end());
    }

    static void ExpectEquals(const MatrixType& matrix, std::initializer_list<Value> expected) {
        ASSERT_EQ(matrix.GetSize() * matrix.GetSize(), expected.size());

        auto it = expected.begin();
        for (std::size_t row = 0; row < matrix.GetSize(); row++) {
            for (std::size_t col = 0; col < matrix.GetSize(); col++) {
                const Value actual = matrix[row, col];
                ASSERT_EQ(actual, *it) << "at (" << row << ", " << col << ")";
                ++it;
            }
        }
    }
};

template <typename Config>
class MatrixTest : public MatrixTestBase<Config> {};

template <typename Config>
class FloatingDeterminantTest : public MatrixTestBase<Config> {};

template <typename Config>
class IntegralDeterminantTest : public MatrixTestBase<Config> {};

template <typename Config>
class LargeFloatingDeterminantTest : public MatrixTestBase<Config> {};

template <typename LhsMatrix, typename RhsMatrix>
void ExpectMatricesEqual(const LhsMatrix& lhs, const RhsMatrix& rhs) {
    ASSERT_EQ(lhs.GetSize(), rhs.GetSize());

    for (std::size_t row = 0; row < lhs.GetSize(); row++) {
        for (std::size_t col = 0; col < lhs.GetSize(); col++) {
            if (lhs[row, col] != rhs[row, col]) {
                ADD_FAILURE() << "Matrices differ at (" << row << ", " << col << ")";
                return;
            }
        }
    }
}

template <typename MatrixType, typename Expected>
void ExpectMatrixMatches(const MatrixType& matrix, Expected expected) {
    for (std::size_t row = 0; row < matrix.GetSize(); row++) {
        for (std::size_t col = 0; col < matrix.GetSize(); col++) {
            if (matrix[row, col] != expected(row, col)) {
                ADD_FAILURE() << "Unexpected value at (" << row << ", " << col << ")";
                return;
            }
        }
    }
}

template <typename T>
void ExpectDeterminantNear(T actual, std::type_identity_t<T> expected, std::size_t ulps = 100) {
    const T tolerance =
        std::numeric_limits<T>::epsilon() * static_cast<T>(ulps) * std::max(T{1}, std::abs(expected));

    EXPECT_NEAR(actual, expected, tolerance);
}

template <typename T>
T ReferenceDeterminant(const std::vector<T>& values, std::size_t size) {
    std::vector<std::size_t> permutation(size);
    std::iota(permutation.begin(), permutation.end(), std::size_t{0});

    T result = T{0};
    do {
        T product = T{1};
        std::size_t inversions = 0;

        for (std::size_t row = 0; row < size; row++) {
            product *= values[row * size + permutation[row]];

            for (std::size_t next = row + 1; next < size; next++) {
                if (permutation[row] > permutation[next]) {
                    inversions++;
                }
            }
        }

        result += (inversions % 2 == 0) ? product : -product;
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    return result;
}

template <typename MatrixType>
void FillTridiagonal(MatrixType& matrix) {
    for (std::size_t i = 0; i < matrix.GetSize(); i++) {
        matrix[i, i] = 2;

        if (i + 1 < matrix.GetSize()) {
            matrix[i, i + 1] = -1;
            matrix[i + 1, i] = -1;
        }
    }
}

template <typename MatrixType>
void AddOnesEverywhere(MatrixType& matrix) {
    for (std::size_t i = 0; i < matrix.GetSize(); i++) {
        for (std::size_t j = 0; j < matrix.GetSize(); j++) {
            matrix[i, j] += 1;
        }
    }
}

template <typename MatrixType>
void FillCyclicShift(MatrixType& matrix) {
    for (std::size_t i = 0; i < matrix.GetSize(); i++) {
        matrix[i, (i + 1) % matrix.GetSize()] = 1;
    }
}

template <typename MatrixType>
void FillAntiIdentity(MatrixType& matrix) {
    for (std::size_t i = 0; i < matrix.GetSize(); i++) {
        matrix[i, matrix.GetSize() - 1 - i] = 1;
    }
}

int CyclicShiftDeterminant(std::size_t size) {
    return size % 2 == 1 ? 1 : -1;
}

int AntiIdentityDeterminant(std::size_t size) {
    return (size / 2) % 2 == 0 ? 1 : -1;
}

} // namespace

TYPED_TEST_SUITE(MatrixTest, AllConfigs);
TYPED_TEST_SUITE(FloatingDeterminantTest, FloatingConfigs);
TYPED_TEST_SUITE(IntegralDeterminantTest, IntegralConfigs);
TYPED_TEST_SUITE(LargeFloatingDeterminantTest, DoubleConfigs);

TYPED_TEST(MatrixTest, SizeConstructorCreatesIdentity) {
    for (std::size_t size : {1, 2, 3, 5, 10}) {
        auto matrix = TestFixture::Identity(size);

        ASSERT_EQ(matrix.GetSize(), size);
        ExpectMatrixMatches(matrix, [](std::size_t row, std::size_t col) {
            return row == col ? 1 : 0;
        });
    }
}

TYPED_TEST(MatrixTest, EmptyMatrixSupportsAllOperations) {
    auto matrix = TestFixture::Identity(0);

    EXPECT_EQ(matrix.GetSize(), 0u);
    EXPECT_EQ(matrix.GetTrace(), 0);
    EXPECT_EQ(matrix.Transpose().GetSize(), 0u);
    EXPECT_EQ(matrix.Negate().GetSize(), 0u);

    auto copy = matrix;
    EXPECT_EQ(copy.GetSize(), 0u);

    std::ostringstream stream;
    matrix.Dump(stream);
    EXPECT_TRUE(stream.str().empty());
}

TYPED_TEST(MatrixTest, IteratorConstructorFillsRowByRow) {
    auto matrix = TestFixture::Make(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    TestFixture::ExpectEquals(matrix, {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TYPED_TEST(MatrixTest, IteratorConstructorAcceptsForwardIterators) {
    const std::list<int> values = {1, 2, 3, 4};
    auto matrix = TestFixture::FromRange(2, values);

    TestFixture::ExpectEquals(matrix, {1, 2, 3, 4});
}

TYPED_TEST(MatrixTest, IteratorConstructorWithSingleElement) {
    auto matrix = TestFixture::Make(1, {42});

    TestFixture::ExpectEquals(matrix, {42});
}

TYPED_TEST(MatrixTest, IteratorConstructorWithTooFewValuesLeavesZeros) {
    auto matrix = TestFixture::Make(3, {1, 2, 3, 4});

    TestFixture::ExpectEquals(matrix, {1, 2, 3, 4, 0, 0, 0, 0, 0});
}

TYPED_TEST(MatrixTest, IteratorConstructorWithEmptyRangeGivesZeroMatrix) {
    auto matrix = TestFixture::Zero(4);

    ASSERT_EQ(matrix.GetSize(), 4u);
    ExpectMatrixMatches(matrix, [](std::size_t, std::size_t) {
        return 0;
    });
}

TYPED_TEST(MatrixTest, ElementAccessReadsAndWrites) {
    auto matrix = TestFixture::Zero(3);

    matrix[1, 2] = 7;

    EXPECT_EQ((matrix[1, 2]), 7);
    TestFixture::ExpectEquals(matrix, {0, 0, 0,
                                       0, 0, 7,
                                       0, 0, 0});
}

TYPED_TEST(MatrixTest, ConstElementAccessReads) {
    const auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    EXPECT_EQ((matrix[0, 0]), 1);
    EXPECT_EQ((matrix[0, 1]), 2);
    EXPECT_EQ((matrix[1, 0]), 3);
    EXPECT_EQ((matrix[1, 1]), 4);
}

TYPED_TEST(MatrixTest, ElementAccessReturnsReference) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    matrix[0, 1] += 10;
    matrix[1, 0] *= 3;

    TestFixture::ExpectEquals(matrix, {1, 12, 9, 4});
}

TYPED_TEST(MatrixTest, DistinctCellsDoNotAlias) {
    constexpr std::size_t size = 6;
    auto matrix = TestFixture::Zero(size);

    for (std::size_t row = 0; row < size; row++) {
        for (std::size_t col = 0; col < size; col++) {
            matrix[row, col] = static_cast<int>(row * 10 + col);
        }
    }

    ExpectMatrixMatches(matrix, [](std::size_t row, std::size_t col) {
        return static_cast<int>(row * 10 + col);
    });
}

TYPED_TEST(MatrixTest, CopyHasSameValues) {
    const auto original = TestFixture::Make(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const auto copy = original;

    ExpectMatricesEqual(original, copy);
}

TYPED_TEST(MatrixTest, CopyIsIndependentFromOriginal) {
    auto original = TestFixture::Make(2, {1, 2, 3, 4});
    auto copy = original;

    copy[0, 0] = 100;
    original[1, 1] = -50;

    TestFixture::ExpectEquals(original, {1, 2, 3, -50});
    TestFixture::ExpectEquals(copy, {100, 2, 3, 4});
}

TYPED_TEST(MatrixTest, TraceOfIdentityEqualsSize) {
    for (std::size_t size : {1, 2, 5, 17}) {
        EXPECT_EQ(TestFixture::Identity(size).GetTrace(), static_cast<int>(size));
    }
}

TYPED_TEST(MatrixTest, TraceSumsOnlyDiagonal) {
    auto matrix = TestFixture::Make(3, {1, 100, 100,
                                        100, 5, 100,
                                        100, 100, 9});

    EXPECT_EQ(matrix.GetTrace(), 15);
}

TYPED_TEST(MatrixTest, TraceWithNegativeValues) {
    auto matrix = TestFixture::Make(2, {-3, 7, 7, -4});

    EXPECT_EQ(matrix.GetTrace(), -7);
}

TYPED_TEST(MatrixTest, TraceOfSingleElement) {
    EXPECT_EQ(TestFixture::Make(1, {13}).GetTrace(), 13);
}

TYPED_TEST(MatrixTest, TraceOfZeroMatrixIsZero) {
    EXPECT_EQ(TestFixture::Zero(5).GetTrace(), 0);
}

TYPED_TEST(MatrixTest, TransposeOfTwoByTwo) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    TestFixture::ExpectEquals(matrix.Transpose(), {1, 3, 2, 4});
}

TYPED_TEST(MatrixTest, TransposeOfThreeByThree) {
    auto matrix = TestFixture::Make(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    TestFixture::ExpectEquals(matrix.Transpose(), {1, 4, 7, 2, 5, 8, 3, 6, 9});
}

TYPED_TEST(MatrixTest, TransposeOfSingleElement) {
    TestFixture::ExpectEquals(TestFixture::Make(1, {5}).Transpose(), {5});
}

TYPED_TEST(MatrixTest, TransposeOfSymmetricMatrixIsSame) {
    const auto matrix = TestFixture::Make(3, {1, 2, 3,
                                              2, 4, 5,
                                              3, 5, 6});

    ExpectMatricesEqual(matrix, matrix.Transpose());
}

TYPED_TEST(MatrixTest, TransposeOfIdentityIsIdentity) {
    const auto identity = TestFixture::Identity(6);

    ExpectMatricesEqual(identity, identity.Transpose());
}

TYPED_TEST(MatrixTest, TransposeTwiceReturnsOriginal) {
    const auto matrix = TestFixture::Sequential(7);

    ExpectMatricesEqual(matrix, matrix.Transpose().Transpose());
}

TYPED_TEST(MatrixTest, TransposeDoesNotModifyOriginal) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    const auto transposed = matrix.Transpose();
    static_cast<void>(transposed);

    TestFixture::ExpectEquals(matrix, {1, 2, 3, 4});
}

TYPED_TEST(MatrixTest, NegateFlipsSigns) {
    auto matrix = TestFixture::Make(2, {1, -2, 3, -4});

    TestFixture::ExpectEquals(matrix.Negate(), {-1, 2, -3, 4});
}

TYPED_TEST(MatrixTest, NegateOfZeroMatrixIsZero) {
    const auto zero = TestFixture::Zero(4);

    ExpectMatricesEqual(zero, zero.Negate());
}

TYPED_TEST(MatrixTest, NegateTwiceReturnsOriginal) {
    const auto matrix = TestFixture::Sequential(5);

    ExpectMatricesEqual(matrix, matrix.Negate().Negate());
}

TYPED_TEST(MatrixTest, NegateDoesNotModifyOriginal) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    const auto negated = matrix.Negate();
    static_cast<void>(negated);

    TestFixture::ExpectEquals(matrix, {1, 2, 3, 4});
}

TYPED_TEST(MatrixTest, NegateAndTransposeCommute) {
    const auto matrix = TestFixture::Sequential(5);

    ExpectMatricesEqual(matrix.Transpose().Negate(), matrix.Negate().Transpose());
}

TYPED_TEST(MatrixTest, MultiplyRowScalesOnlyChosenRow) {
    auto matrix = TestFixture::Make(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    TestFixture::ExpectEquals(matrix.MultiplyRow(1, 10), {1, 2, 3,
                                                          40, 50, 60,
                                                          7, 8, 9});
}

TYPED_TEST(MatrixTest, MultiplyFirstAndLastRows) {
    auto matrix = TestFixture::Make(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    TestFixture::ExpectEquals(matrix.MultiplyRow(0, 2), {2, 4, 6, 4, 5, 6, 7, 8, 9});
    TestFixture::ExpectEquals(matrix.MultiplyRow(2, 2), {1, 2, 3, 4, 5, 6, 14, 16, 18});
}

TYPED_TEST(MatrixTest, MultiplyRowByZeroClearsRow) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    TestFixture::ExpectEquals(matrix.MultiplyRow(0, 0), {0, 0, 3, 4});
}

TYPED_TEST(MatrixTest, MultiplyRowByOneKeepsMatrix) {
    const auto matrix = TestFixture::Sequential(4);

    ExpectMatricesEqual(matrix, matrix.MultiplyRow(2, 1));
}

TYPED_TEST(MatrixTest, MultiplyRowByNegativeNumber) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    TestFixture::ExpectEquals(matrix.MultiplyRow(1, -1), {1, 2, -3, -4});
}

TYPED_TEST(MatrixTest, MultiplyRowOfSingleElement) {
    TestFixture::ExpectEquals(TestFixture::Make(1, {6}).MultiplyRow(0, 7), {42});
}

TYPED_TEST(MatrixTest, MultiplyRowDoesNotModifyOriginal) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    const auto scaled = matrix.MultiplyRow(0, 5);
    static_cast<void>(scaled);

    TestFixture::ExpectEquals(matrix, {1, 2, 3, 4});
}

TYPED_TEST(MatrixTest, DumpPrintsRowsWithTrailingSpaces) {
    auto matrix = TestFixture::Make(2, {1, 2, 3, 4});

    std::ostringstream stream;
    matrix.Dump(stream);

    EXPECT_EQ(stream.str(), "1 2 \n3 4 \n");
}

TYPED_TEST(MatrixTest, DumpOfSingleElement) {
    std::ostringstream stream;
    TestFixture::Make(1, {7}).Dump(stream);

    EXPECT_EQ(stream.str(), "7 \n");
}

TYPED_TEST(MatrixTest, DumpOfNegativeValues) {
    std::ostringstream stream;
    TestFixture::Make(2, {-1, 0, 0, -2}).Dump(stream);

    EXPECT_EQ(stream.str(), "-1 0 \n0 -2 \n");
}

TYPED_TEST(MatrixTest, DumpAppendsToStream) {
    std::ostringstream stream;
    stream << "header\n";
    TestFixture::Identity(2).Dump(stream);

    EXPECT_EQ(stream.str(), "header\n1 0 \n0 1 \n");
}

TYPED_TEST(MatrixTest, DumpOfLargeMatrixHasLinePerRow) {
    constexpr std::size_t size = 200;

    std::ostringstream stream;
    TestFixture::Identity(size).Dump(stream);

    const std::string output = stream.str();
    EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), static_cast<std::ptrdiff_t>(size));
}

TYPED_TEST(MatrixTest, LargeIdentity) {
    auto matrix = TestFixture::Identity(kLargeSize);

    ASSERT_EQ(matrix.GetSize(), kLargeSize);
    EXPECT_EQ(matrix.GetTrace(), static_cast<int>(kLargeSize));
    ExpectMatrixMatches(matrix, [](std::size_t row, std::size_t col) {
        return row == col ? 1 : 0;
    });
}

TYPED_TEST(MatrixTest, LargeMatrixKeepsAllValues) {
    const auto matrix = TestFixture::Sequential(kLargeSize);

    ASSERT_EQ(matrix.GetSize(), kLargeSize);
    ExpectMatrixMatches(matrix, [](std::size_t row, std::size_t col) {
        return static_cast<int>(row * kLargeSize + col);
    });
}

TYPED_TEST(MatrixTest, LargeCopyIsIndependentFromOriginal) {
    auto original = TestFixture::Sequential(kLargeSize);
    auto copy = original;

    ExpectMatricesEqual(original, copy);

    copy[0, 0] = -1;
    copy[kLargeSize - 1, kLargeSize - 1] = -2;

    EXPECT_EQ((original[0, 0]), 0);
    EXPECT_EQ((original[kLargeSize - 1, kLargeSize - 1]), static_cast<int>(kLargeSize * kLargeSize - 1));
}

TYPED_TEST(MatrixTest, LargeTranspose) {
    const auto matrix = TestFixture::Sequential(kLargeSize);
    const auto transposed = matrix.Transpose();

    ExpectMatrixMatches(transposed, [](std::size_t row, std::size_t col) {
        return static_cast<int>(col * kLargeSize + row);
    });
}

TYPED_TEST(MatrixTest, LargeNegate) {
    const auto matrix = TestFixture::Sequential(kLargeSize);
    const auto negated = matrix.Negate();

    ExpectMatrixMatches(negated, [](std::size_t row, std::size_t col) {
        return -static_cast<int>(row * kLargeSize + col);
    });
}

TYPED_TEST(MatrixTest, LargeMultiplyRow) {
    constexpr std::size_t scaled_row = kLargeSize / 2;

    const auto matrix = TestFixture::Sequential(kLargeSize);
    const auto scaled = matrix.MultiplyRow(scaled_row, 2);

    ExpectMatrixMatches(scaled, [](std::size_t row, std::size_t col) {
        const int original = static_cast<int>(row * kLargeSize + col);
        return row == scaled_row ? original * 2 : original;
    });
}

TYPED_TEST(MatrixTest, LargeTrace) {
    auto matrix = TestFixture::Zero(kLargeSize);
    for (std::size_t i = 0; i < kLargeSize; i++) {
        matrix[i, i] = static_cast<int>(i);
    }

    EXPECT_EQ(matrix.GetTrace(), static_cast<int>(kLargeSize * (kLargeSize - 1) / 2));
}

TYPED_TEST(FloatingDeterminantTest, IdentityHasDeterminantOne) {
    for (std::size_t size : {1, 2, 3, 7, 20}) {
        ExpectDeterminantNear(TestFixture::Identity(size).Determinant(), 1);
    }
}

TYPED_TEST(FloatingDeterminantTest, EmptyMatrixHasDeterminantOne) {
    ExpectDeterminantNear(TestFixture::Identity(0).Determinant(), 1);
}

TYPED_TEST(FloatingDeterminantTest, SingleElement) {
    ExpectDeterminantNear(TestFixture::Make(1, {-4.5}).Determinant(), -4.5);
}

TYPED_TEST(FloatingDeterminantTest, TwoByTwo) {
    ExpectDeterminantNear(TestFixture::Make(2, {1, 2, 3, 4}).Determinant(), -2);
}

TYPED_TEST(FloatingDeterminantTest, ThreeByThree) {
    auto matrix = TestFixture::Make(3, {6, 1, 1,
                                        4, -2, 5,
                                        2, 8, 7});

    ExpectDeterminantNear(matrix.Determinant(), -306);
}

TYPED_TEST(FloatingDeterminantTest, FourByFour) {
    auto matrix = TestFixture::Make(4, {1, 2, 3, 4,
                                        5, 6, 7, 8,
                                        2, 6, 4, 8,
                                        3, 1, 1, 2});

    ExpectDeterminantNear(matrix.Determinant(), 72);
}

TYPED_TEST(FloatingDeterminantTest, FractionalValues) {
    ExpectDeterminantNear(TestFixture::Make(2, {0.5, 0.25, 2, 3}).Determinant(), 1);
}

TYPED_TEST(FloatingDeterminantTest, ZeroPivotRequiresRowSwap) {
    ExpectDeterminantNear(TestFixture::Make(2, {0, 1, 1, 0}).Determinant(), -1);

    auto matrix = TestFixture::Make(3, {0, 2, 1,
                                        1, 0, 3,
                                        4, 1, 0});
    ExpectDeterminantNear(matrix.Determinant(), 25);
}

TYPED_TEST(FloatingDeterminantTest, ZeroPivotAppearsInTheMiddle) {
    auto matrix = TestFixture::Make(3, {1, 1, 1,
                                        1, 1, 2,
                                        1, 2, 3});

    ExpectDeterminantNear(matrix.Determinant(), -1);
}

TYPED_TEST(FloatingDeterminantTest, SingularMatricesHaveZeroDeterminant) {
    ExpectDeterminantNear(TestFixture::Zero(4).Determinant(), 0);
    ExpectDeterminantNear(TestFixture::Make(2, {1, 2, 2, 4}).Determinant(), 0);
    ExpectDeterminantNear(TestFixture::Make(2, {0, 1, 0, 2}).Determinant(), 0);

    auto zero_row = TestFixture::Make(3, {1, 2, 3,
                                          0, 0, 0,
                                          4, 5, 6});
    ExpectDeterminantNear(zero_row.Determinant(), 0);

    auto dependent_rows = TestFixture::Make(3, {1, 2, 3,
                                                4, 5, 6,
                                                7, 8, 9});
    ExpectDeterminantNear(dependent_rows.Determinant(), 0);
}

TYPED_TEST(FloatingDeterminantTest, DiagonalMatrix) {
    auto matrix = TestFixture::Zero(4);
    for (std::size_t i = 0; i < 4; i++) {
        matrix[i, i] = static_cast<int>(i) + 2;
    }

    ExpectDeterminantNear(matrix.Determinant(), 2 * 3 * 4 * 5);
}

TYPED_TEST(FloatingDeterminantTest, TriangularMatrices) {
    auto upper = TestFixture::Make(3, {2, 5, 7,
                                       0, 3, 9,
                                       0, 0, 4});
    auto lower = TestFixture::Make(3, {2, 0, 0,
                                       5, 3, 0,
                                       7, 9, 4});

    ExpectDeterminantNear(upper.Determinant(), 24);
    ExpectDeterminantNear(lower.Determinant(), 24);
}

TYPED_TEST(FloatingDeterminantTest, DoesNotModifyMatrix) {
    auto matrix = TestFixture::Make(3, {0, 2, 1,
                                        1, 0, 3,
                                        4, 1, 0});

    const auto determinant = matrix.Determinant();
    static_cast<void>(determinant);

    TestFixture::ExpectEquals(matrix, {0, 2, 1,
                                       1, 0, 3,
                                       4, 1, 0});
}

TYPED_TEST(FloatingDeterminantTest, SwappingRowsFlipsSign) {
    auto matrix = TestFixture::Make(3, {6, 1, 1, 4, -2, 5, 2, 8, 7});
    auto swapped = TestFixture::Make(3, {4, -2, 5, 6, 1, 1, 2, 8, 7});

    ExpectDeterminantNear(swapped.Determinant(), -matrix.Determinant());
}

TYPED_TEST(FloatingDeterminantTest, MultiplyRowScalesDeterminant) {
    auto matrix = TestFixture::Make(3, {6, 1, 1, 4, -2, 5, 2, 8, 7});

    ExpectDeterminantNear(matrix.MultiplyRow(1, 3).Determinant(), 3 * matrix.Determinant());
    ExpectDeterminantNear(matrix.MultiplyRow(2, 0).Determinant(), 0);
}

TYPED_TEST(FloatingDeterminantTest, TransposePreservesDeterminant) {
    auto matrix = TestFixture::Make(4, {2, 1, 0, 3,
                                        1, 3, 1, 0,
                                        0, 1, 4, 2,
                                        3, 0, 2, 5});

    ExpectDeterminantNear(matrix.Transpose().Determinant(), matrix.Determinant());
}

TYPED_TEST(FloatingDeterminantTest, NegateScalesDeterminantByParityOfSize) {
    auto odd = TestFixture::Make(3, {6, 1, 1, 4, -2, 5, 2, 8, 7});
    auto even = TestFixture::Make(2, {1, 2, 3, 4});

    ExpectDeterminantNear(odd.Negate().Determinant(), -odd.Determinant());
    ExpectDeterminantNear(even.Negate().Determinant(), even.Determinant());
}

TYPED_TEST(FloatingDeterminantTest, MatchesReferenceOnRandomMatrices) {
    using Value = typename TestFixture::Value;

    std::mt19937 generator(12345);
    std::uniform_int_distribution<int> off_diagonal(-3, 3);
    std::uniform_int_distribution<int> diagonal(20, 25);

    for (std::size_t size = 1; size <= 6; size++) {
        for (int iteration = 0; iteration < 200; iteration++) {
            std::vector<double> values(size * size);
            for (std::size_t row = 0; row < size; row++) {
                for (std::size_t col = 0; col < size; col++) {
                    values[row * size + col] = row == col ? diagonal(generator) : off_diagonal(generator);
                }
            }

            const auto matrix = TestFixture::FromRange(size, values);
            const Value expected = static_cast<Value>(ReferenceDeterminant(values, size));

            ExpectDeterminantNear(matrix.Determinant(), expected, 1000);
        }
    }
}

TYPED_TEST(IntegralDeterminantTest, IdentityHasDeterminantOne) {
    for (std::size_t size : {1, 2, 3, 7, 20}) {
        EXPECT_EQ(TestFixture::Identity(size).Determinant(), 1);
    }
}

TYPED_TEST(IntegralDeterminantTest, SingleElement) {
    EXPECT_EQ(TestFixture::Make(1, {-9}).Determinant(), -9);
    EXPECT_EQ(TestFixture::Make(1, {0}).Determinant(), 0);
}

TYPED_TEST(IntegralDeterminantTest, TwoByTwo) {
    EXPECT_EQ(TestFixture::Make(2, {1, 2, 3, 4}).Determinant(), -2);
}

TYPED_TEST(IntegralDeterminantTest, ThreeByThree) {
    auto matrix = TestFixture::Make(3, {6, 1, 1,
                                        4, -2, 5,
                                        2, 8, 7});

    EXPECT_EQ(matrix.Determinant(), -306);
}

TYPED_TEST(IntegralDeterminantTest, FourByFour) {
    auto matrix = TestFixture::Make(4, {1, 2, 3, 4,
                                        5, 6, 7, 8,
                                        2, 6, 4, 8,
                                        3, 1, 1, 2});

    EXPECT_EQ(matrix.Determinant(), 72);
}

TYPED_TEST(IntegralDeterminantTest, ZeroPivotRequiresRowSwap) {
    EXPECT_EQ(TestFixture::Make(2, {0, 1, 1, 0}).Determinant(), -1);

    auto matrix = TestFixture::Make(3, {0, 2, 1,
                                        1, 0, 3,
                                        4, 1, 0});
    EXPECT_EQ(matrix.Determinant(), 25);
}

TYPED_TEST(IntegralDeterminantTest, ZeroPivotAppearsInTheMiddle) {
    auto matrix = TestFixture::Make(3, {1, 1, 1,
                                        1, 1, 2,
                                        1, 2, 3});

    EXPECT_EQ(matrix.Determinant(), -1);
}

TYPED_TEST(IntegralDeterminantTest, SingularMatricesHaveZeroDeterminant) {
    EXPECT_EQ(TestFixture::Zero(4).Determinant(), 0);
    EXPECT_EQ(TestFixture::Make(2, {1, 2, 2, 4}).Determinant(), 0);
    EXPECT_EQ(TestFixture::Make(2, {0, 1, 0, 2}).Determinant(), 0);

    auto zero_row = TestFixture::Make(3, {1, 2, 3,
                                          0, 0, 0,
                                          4, 5, 6});
    EXPECT_EQ(zero_row.Determinant(), 0);

    auto dependent_rows = TestFixture::Make(3, {1, 2, 3,
                                                4, 5, 6,
                                                7, 8, 9});
    EXPECT_EQ(dependent_rows.Determinant(), 0);

    auto dependent_columns = TestFixture::Make(3, {1, 2, 3,
                                                   2, 4, 5,
                                                   3, 6, 10});
    EXPECT_EQ(dependent_columns.Determinant(), 0);
}

TYPED_TEST(IntegralDeterminantTest, DiagonalMatrix) {
    auto matrix = TestFixture::Zero(4);
    for (std::size_t i = 0; i < 4; i++) {
        matrix[i, i] = static_cast<int>(i) + 2;
    }

    EXPECT_EQ(matrix.Determinant(), 2 * 3 * 4 * 5);
}

TYPED_TEST(IntegralDeterminantTest, TriangularMatrices) {
    auto upper = TestFixture::Make(3, {2, 5, 7,
                                       0, 3, 9,
                                       0, 0, 4});
    auto lower = TestFixture::Make(3, {2, 0, 0,
                                       5, 3, 0,
                                       7, 9, 4});

    EXPECT_EQ(upper.Determinant(), 24);
    EXPECT_EQ(lower.Determinant(), 24);
}

TYPED_TEST(IntegralDeterminantTest, DoesNotModifyMatrix) {
    auto matrix = TestFixture::Make(3, {0, 2, 1,
                                        1, 0, 3,
                                        4, 1, 0});

    const auto determinant = matrix.Determinant();
    static_cast<void>(determinant);

    TestFixture::ExpectEquals(matrix, {0, 2, 1,
                                       1, 0, 3,
                                       4, 1, 0});
}

TYPED_TEST(IntegralDeterminantTest, SwappingRowsFlipsSign) {
    auto matrix = TestFixture::Make(3, {6, 1, 1, 4, -2, 5, 2, 8, 7});
    auto swapped = TestFixture::Make(3, {4, -2, 5, 6, 1, 1, 2, 8, 7});

    EXPECT_EQ(swapped.Determinant(), -matrix.Determinant());
}

TYPED_TEST(IntegralDeterminantTest, MultiplyRowScalesDeterminant) {
    auto matrix = TestFixture::Make(3, {6, 1, 1, 4, -2, 5, 2, 8, 7});

    EXPECT_EQ(matrix.MultiplyRow(1, 3).Determinant(), 3 * matrix.Determinant());
    EXPECT_EQ(matrix.MultiplyRow(2, 0).Determinant(), 0);
}

TYPED_TEST(IntegralDeterminantTest, TransposePreservesDeterminant) {
    auto matrix = TestFixture::Make(4, {2, 1, 0, 3,
                                        1, 3, 1, 0,
                                        0, 1, 4, 2,
                                        3, 0, 2, 5});

    EXPECT_EQ(matrix.Transpose().Determinant(), matrix.Determinant());
}

TYPED_TEST(IntegralDeterminantTest, NegateScalesDeterminantByParityOfSize) {
    auto odd = TestFixture::Make(3, {6, 1, 1, 4, -2, 5, 2, 8, 7});
    auto even = TestFixture::Make(2, {1, 2, 3, 4});

    EXPECT_EQ(odd.Negate().Determinant(), -odd.Determinant());
    EXPECT_EQ(even.Negate().Determinant(), even.Determinant());
}

TYPED_TEST(IntegralDeterminantTest, MatchesReferenceOnRandomMatrices) {
    using Value = typename TestFixture::Value;

    std::mt19937 generator(12345);
    std::uniform_int_distribution<int> distribution(-2, 2);

    for (std::size_t size = 1; size <= 6; size++) {
        for (int iteration = 0; iteration < 300; iteration++) {
            std::vector<long long> values(size * size);
            for (auto& value : values) {
                value = distribution(generator);
            }

            const auto matrix = TestFixture::FromRange(size, values);
            const Value expected = static_cast<Value>(ReferenceDeterminant(values, size));

            ASSERT_EQ(matrix.Determinant(), expected) << "size = " << size << ", iteration = " << iteration;
        }
    }
}

TYPED_TEST(IntegralDeterminantTest, LargeIdentity) {
    EXPECT_EQ(TestFixture::Identity(kLargeDenseIntegralSize).Determinant(), 1);
}

TYPED_TEST(IntegralDeterminantTest, LargeTridiagonal) {
    auto matrix = TestFixture::Zero(kLargeDenseIntegralSize);
    FillTridiagonal(matrix);

    EXPECT_EQ(matrix.Determinant(), static_cast<int>(kLargeDenseIntegralSize + 1));
}

TYPED_TEST(IntegralDeterminantTest, LargeDenseMatrix) {
    auto matrix = TestFixture::Identity(kLargeDenseIntegralSize);
    AddOnesEverywhere(matrix);

    EXPECT_EQ(matrix.Determinant(), static_cast<int>(kLargeDenseIntegralSize + 1));
}

TYPED_TEST(IntegralDeterminantTest, LargeUpperTriangular) {
    using Value = typename TestFixture::Value;

    auto matrix = TestFixture::Zero(kLargeDenseIntegralSize);
    Value expected = 1;

    for (std::size_t i = 0; i < kLargeDenseIntegralSize; i++) {
        const int diagonal = i % 2 == 0 ? 1 : -1;
        matrix[i, i] = diagonal;
        expected *= diagonal;

        for (std::size_t j = i + 1; j < kLargeDenseIntegralSize; j++) {
            matrix[i, j] = static_cast<int>((i + j) % 5) + 1;
        }
    }

    EXPECT_EQ(matrix.Determinant(), expected);
}

TYPED_TEST(IntegralDeterminantTest, LargeSingularMatrix) {
    auto matrix = TestFixture::Zero(kLargeDenseIntegralSize);
    for (std::size_t i = 0; i < kLargeDenseIntegralSize; i++) {
        for (std::size_t j = 0; j < kLargeDenseIntegralSize; j++) {
            matrix[i, j] = 1;
        }
    }

    EXPECT_EQ(matrix.Determinant(), 0);
}

TYPED_TEST(IntegralDeterminantTest, LargePermutationMatrices) {
    for (std::size_t size : {kLargeDenseIntegralSize, kLargeDenseIntegralSize + 1}) {
        auto cyclic = TestFixture::Zero(size);
        FillCyclicShift(cyclic);
        EXPECT_EQ(cyclic.Determinant(), CyclicShiftDeterminant(size)) << "cyclic shift, size = " << size;

        auto anti_identity = TestFixture::Zero(size);
        FillAntiIdentity(anti_identity);
        EXPECT_EQ(anti_identity.Determinant(), AntiIdentityDeterminant(size)) << "anti-identity, size = " << size;
    }
}

TYPED_TEST(LargeFloatingDeterminantTest, LargeIdentity) {
    EXPECT_DOUBLE_EQ(TestFixture::Identity(kLargeSize).Determinant(), 1.0);
}

TYPED_TEST(LargeFloatingDeterminantTest, LargeTridiagonal) {
    auto matrix = TestFixture::Zero(kLargeSize);
    FillTridiagonal(matrix);

    ExpectDeterminantNear(matrix.Determinant(), static_cast<double>(kLargeSize + 1), kLargeSize * kLargeSize);
}

TYPED_TEST(LargeFloatingDeterminantTest, LargeDenseMatrix) {
    auto matrix = TestFixture::Identity(kLargeDenseFloatingSize);
    AddOnesEverywhere(matrix);

    ExpectDeterminantNear(matrix.Determinant(),
                          static_cast<double>(kLargeDenseFloatingSize + 1),
                          kLargeDenseFloatingSize * kLargeDenseFloatingSize);
}

TYPED_TEST(LargeFloatingDeterminantTest, LargeUpperTriangular) {
    auto matrix = TestFixture::Zero(kLargeSize);
    double expected = 1.0;

    for (std::size_t i = 0; i < kLargeSize; i++) {
        const double diagonal = i % 3 == 0 ? 4.0 : (i % 3 == 1 ? 0.5 : 0.25);
        matrix[i, i] = diagonal;
        expected *= diagonal;

        for (std::size_t j = i + 1; j < kLargeSize; j++) {
            matrix[i, j] = static_cast<double>((i + j) % 7) + 1.0;
        }
    }

    EXPECT_DOUBLE_EQ(matrix.Determinant(), expected);
}

TYPED_TEST(LargeFloatingDeterminantTest, LargeSingularMatrix) {
    auto matrix = TestFixture::Zero(kLargeDenseFloatingSize);
    for (std::size_t i = 0; i < kLargeDenseFloatingSize; i++) {
        for (std::size_t j = 0; j < kLargeDenseFloatingSize; j++) {
            matrix[i, j] = 1;
        }
    }

    EXPECT_DOUBLE_EQ(matrix.Determinant(), 0.0);
}

TYPED_TEST(LargeFloatingDeterminantTest, LargePermutationMatrices) {
    for (std::size_t size : {500, 501, 502}) {
        auto cyclic = TestFixture::Zero(size);
        FillCyclicShift(cyclic);
        EXPECT_DOUBLE_EQ(cyclic.Determinant(), CyclicShiftDeterminant(size)) << "cyclic shift, size = " << size;

        auto anti_identity = TestFixture::Zero(size);
        FillAntiIdentity(anti_identity);
        EXPECT_DOUBLE_EQ(anti_identity.Determinant(), AntiIdentityDeterminant(size))
            << "anti-identity, size = " << size;
    }
}

TEST(MatrixMoveTest, MoveConstructorTransfersContents) {
    const std::vector<int> values = {1, 2, 3, 4};
    FlatIntMatrix source(2, values.begin(), values.end());

    FlatIntMatrix target(std::move(source));

    ASSERT_EQ(target.GetSize(), 2u);
    EXPECT_EQ((target[0, 0]), 1);
    EXPECT_EQ((target[0, 1]), 2);
    EXPECT_EQ((target[1, 0]), 3);
    EXPECT_EQ((target[1, 1]), 4);
}

TEST(MatrixMoveTest, MoveConstructorOfLargeMatrixTransfersContents) {
    std::vector<int> values(kLargeSize * kLargeSize);
    std::iota(values.begin(), values.end(), 0);
    FlatIntMatrix source(kLargeSize, values.begin(), values.end());

    FlatIntMatrix target(std::move(source));

    ASSERT_EQ(target.GetSize(), kLargeSize);
    ExpectMatrixMatches(target, [](std::size_t row, std::size_t col) {
        return static_cast<int>(row * kLargeSize + col);
    });
}

TEST(MatrixMoveTest, JaggedMoveConstructorTransfersContents) {
    const std::vector<int> values = {1, 2, 3, 4};
    JaggedIntMatrix source(2, values.begin(), values.end());

    JaggedIntMatrix target(std::move(source));

    ASSERT_EQ(target.GetSize(), 2u);
    EXPECT_EQ((target[1, 1]), 4);
}

TEST(CrossStorageTest, FlatAndJaggedGiveSameResults) {
    constexpr std::size_t size = 40;

    std::mt19937 generator(2024);
    std::uniform_int_distribution<int> distribution(-3, 3);

    std::vector<double> values(size * size);
    for (std::size_t row = 0; row < size; row++) {
        for (std::size_t col = 0; col < size; col++) {
            values[row * size + col] = row == col ? size * 4.0 : distribution(generator);
        }
    }

    const FlatDoubleMatrix flat(size, values.begin(), values.end());
    const JaggedDoubleMatrix jagged(size, values.begin(), values.end());

    ExpectMatricesEqual(flat, jagged);
    ExpectMatricesEqual(flat.Transpose(), jagged.Transpose());
    ExpectMatricesEqual(flat.Negate(), jagged.Negate());
    ExpectMatricesEqual(flat.MultiplyRow(7, 2.5), jagged.MultiplyRow(7, 2.5));
    EXPECT_DOUBLE_EQ(flat.GetTrace(), jagged.GetTrace());
    EXPECT_DOUBLE_EQ(flat.Determinant(), jagged.Determinant());
}

TEST(CrossStorageTest, FlatAndJaggedGiveSameIntegralDeterminants) {
    std::mt19937 generator(7);
    std::uniform_int_distribution<int> distribution(-2, 2);

    for (std::size_t size = 1; size <= 6; size++) {
        for (int iteration = 0; iteration < 100; iteration++) {
            std::vector<int> values(size * size);
            for (auto& value : values) {
                value = distribution(generator);
            }

            const FlatIntMatrix flat(size, values.begin(), values.end());
            const JaggedIntMatrix jagged(size, values.begin(), values.end());

            ASSERT_EQ(flat.Determinant(), jagged.Determinant()) << "size = " << size << ", iteration = " << iteration;
        }
    }
}

#ifndef NDEBUG

using DeathTestMatrices = ::testing::Types<FlatIntMatrix, JaggedIntMatrix>;

template <typename MatrixType>
class MatrixBoundsDeathTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixBoundsDeathTest, DeathTestMatrices);

TYPED_TEST(MatrixBoundsDeathTest, RowOutOfRangeAsserts) {
    TypeParam matrix(3);

    EXPECT_DEATH(static_cast<void>(matrix[3, 0]), "");
}

TYPED_TEST(MatrixBoundsDeathTest, ColumnOutOfRangeAsserts) {
    TypeParam matrix(3);

    EXPECT_DEATH(static_cast<void>(matrix[0, 3]), "");
}

TYPED_TEST(MatrixBoundsDeathTest, ConstAccessOutOfRangeAsserts) {
    const TypeParam matrix(3);

    EXPECT_DEATH(static_cast<void>(matrix[3, 3]), "");
}

TYPED_TEST(MatrixBoundsDeathTest, AccessToEmptyMatrixAsserts) {
    TypeParam matrix(0);

    EXPECT_DEATH(static_cast<void>(matrix[0, 0]), "");
}

TYPED_TEST(MatrixBoundsDeathTest, TooManyValuesInIteratorConstructorAsserts) {
    const std::vector<int> values(5, 1);
    auto construct = [&values] {
        TypeParam matrix(2, values.begin(), values.end());
    };

    EXPECT_DEATH(construct(), "");
}

#endif // NDEBUG
