#pragma once

#include <cassert>
#include <cstddef>
#include <ostream>
#include <iostream>
#include <type_traits>
#include <utility>

#include "flat_storage.hpp"
#include "jagged_storage.hpp"

namespace Matrix {

template <typename T, typename Storage = FlatStorage<T>>
class Matrix {
public:
    Matrix(std::size_t size) : size_(size), storage_(size_) {
        const T default_value = T{1};

        for (std::size_t index = 0; index < GetSize(); index++) {
            (*this)[index, index] = default_value;
        }
    }

    template <typename It>
    Matrix(std::size_t size, It begin, It end) : size_(size), storage_(size_) {
        std::size_t row = 0;
        std::size_t col = 0;
        for (auto it = begin; it != end; it++) {
            (*this)[row, col] = *it;
            col++;
            if (col == GetSize()) { // FIXME
                col = 0;
                row++;
            }
        }
    }

    Matrix(const Matrix& other) : size_(other.size_), storage_(size_) {
        for (std::size_t i = 0; i < GetSize(); i++) {
            for (std::size_t j = 0; j < GetSize(); j++) {
                (*this)[i, j] = other[i, j];
            }
        }
    }

    Matrix(Matrix&& other) noexcept : size_(other.GetSize()), storage_(std::move(other.storage_)) {}

    Matrix& operator=(const Matrix& other) = delete;
    Matrix& operator=(Matrix&& other) = delete;

    T& operator[](std::size_t row, std::size_t col) {
        return GetStorage()[row, col];
    }

    const T& operator[](std::size_t row, std::size_t col) const {
        return GetStorage()[row, col];
    }

    std::size_t GetSize() const {
        return size_;
    }

    Matrix Transpose() const {
        Matrix res(*this);
        for (std::size_t i = 0; i < GetSize(); i++) {
            for (std::size_t j = 0; j < GetSize(); j++) {
                res[j, i] = (*this)[i, j];
            }
        }

        return res;
    }

    Matrix Negate() const {
        Matrix res(*this);

        for (std::size_t i = 0; i < GetSize(); i++) {
            for (std::size_t j = 0; j < GetSize(); j++) {
                res[i, j] = -(*this)[i, j];
            }
        }

        return res;
    }

    Matrix MultiplyRow(std::size_t row, const T& mult) const {
        Matrix res(*this);
        for (std::size_t i = 0; i < GetSize(); i++) {
            res[row, i] = (*this)[row, i] * mult;
        }

        return res;
    }

    T GetTrace() const {
        T res = T{};
        for (std::size_t index = 0; index < GetSize(); index++) {
            res += (*this)[index, index];
        }

        return res;
    }

    void Dump(std::ostream& stream) const {
        for (std::size_t i = 0; i < GetSize(); i++) {
            for (std::size_t j = 0; j < GetSize(); j++) {
                stream << (*this)[i, j] << " ";
            }

            stream << std::endl;
        }
    }

    T Determinant() const requires std::is_floating_point<T>::value {
        Matrix res(*this);
        bool sign = false;
        const T default_value = T{0};

        for (std::size_t i = 0; i < GetSize(); i++) {
            if (res[i, i] == default_value) {
                for (std::size_t k = i + 1; k < GetSize(); k++) {
                    if (res[k, i] != default_value) {
                        res.SwapRows(i, k);
                        sign = !sign;
                        break;
                    }
                }
            }

            if (res[i, i] == default_value) {
                return default_value;
            }

            for (std::size_t k = i + 1; k < GetSize(); k++) {
                if (res[k, i] != default_value) {
                    SubMultRows(res, i, k, res[k, i] / res[i, i]);
                }
            }
        }

        T result = T{1};
        for (std::size_t i = 0; i < GetSize(); i++) {
            result *= res[i, i];
        }

        return result * (sign ? -1 : 1);
    }

    T Determinant() const requires std::is_integral<T>::value {
        if (GetSize() == 0) {
            return 0;
        }

        Matrix minors(*this);
        bool sign = false;
        T prev = 1;
        const T default_value = 0;

        for (std::size_t i = 0; i + 1 < GetSize(); i++) {
            if (minors[i, i] == default_value) {
                std::size_t row = i + 1;
                for (; row < GetSize(); row++) {
                    if (minors[row, i] != default_value) {
                        break;
                    }
                }

                if (row == GetSize()) {
                    return 0;
                }

                minors.SwapRows(i, row);
                sign = !sign;
            }

            for (std::size_t k = i + 1; k < GetSize(); k++) {
                for (std::size_t j = i + 1; j < GetSize(); j++) {
                    minors[k, j] = (minors[k, j] * minors[i, i] - minors[k, i] * minors[i, j]) / prev;
                }
            }

            prev = minors[i, i];
        }

        return minors[GetSize() - 1, GetSize() - 1]  * (sign ? -1 : 1);
    }

private:
    Storage& GetStorage() {
        return storage_;
    }

    const Storage& GetStorage() const {
        return storage_;
    }

    void SwapRows(std::size_t lhs, std::size_t rhs) {
        GetStorage().SwapRows(lhs, rhs);
    }

    void SubMultRows(Matrix& matrix, std::size_t lhs, std::size_t rhs, const T& mult) const {
        for (std::size_t i = 0; i < GetSize(); i++) {
            matrix[rhs, i] -= matrix[lhs, i] * mult;
        }
    }

    std::size_t size_ = 0;
    Storage storage_;
};

} // namespace Matrix
