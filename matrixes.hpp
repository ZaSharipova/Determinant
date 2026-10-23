#pragma once

#include <cassert>
#include <cstddef>
#include <ostream>
#include <iostream>
#include <type_traits>

namespace Matrix {

template <typename T>
class Matrix {
public:
    Matrix(std::size_t size) : size_(size) {
        AllocateArray(GetSize() * GetSize());
        const T default_value = T{1};

        for (std::size_t index = 0; index < GetSize(); index++) {
            (*this)[index, index] = default_value;
        }
    }

    template <typename It>
    Matrix(std::size_t size, It begin, It end) : size_(size) {
        AllocateArray(GetSize() * GetSize());

        std::size_t counter = 0;
        for (auto it = begin; it != end; it++) {
            array_[counter++] = *it;
        }
    }

    Matrix(const Matrix& other) : size_(other.size_) {
        AllocateArray(other.GetSize() * other.GetSize());

        for (std::size_t i = 0; i < GetSize() * GetSize(); i++) {
            array_[i] = other.array_[i];
        }
    }

    Matrix(Matrix&& other) noexcept : size_(other.GetSize()), array_(other.array_) {
        other.array_ = nullptr;
    }

    Matrix& operator=(const Matrix& other) = delete;
    Matrix& operator=(Matrix&& other) = delete;

    ~Matrix() {
         delete[] array_;
    }

    T& operator[](std::size_t row, std::size_t col) {
        return array_[row * GetSize() + col];
    }

    const T& operator[](std::size_t row, std::size_t col) const {
        return array_[row * GetSize() + col];
    }

    std::size_t GetSize() const {
        return size_;
    }

    const T* GetMatrix() const {
        return array_;
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
        for (std::size_t i = 0; i < GetSize() * GetSize(); i++) {
            res.array_[i] = -array_[i];
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
                        SwapRows(res, i, k);
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

                SwapRows(minors, i, row);
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
    Matrix& SwapRows(Matrix& matrix, std::size_t lhs, std::size_t rhs) const {
        for (std::size_t i = 0; i < GetSize(); i++) {
            std::swap(matrix[lhs, i], matrix[rhs, i]);
        }

        return matrix;
    }

    Matrix& SubMultRows(Matrix& matrix, std::size_t lhs, std::size_t rhs, const T& mult) const {
        for (std::size_t i = 0; i < GetSize(); i++) {
            matrix[rhs, i] -= matrix[lhs, i] * mult;
        }

        return matrix;
    }

    void AllocateArray(std::size_t capacity) {
        array_ = new T[capacity]();
    }

    std::size_t size_ = 0;
    T* array_ = nullptr;
};

} // namespace Matrix
