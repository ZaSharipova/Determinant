#pragma once

#include <array>
#include <cstddef>
#include <ostream>
#include <iostream>
#include <type_traits>

namespace Matrix {

template <typename T>
class Matrix {
public:
    Matrix(std::size_t size) : size_(size) {
        array_ = new T[size_ * size_]();

        for (std::size_t index = 0; index < size; index++) {
            array_[index + size * index] = 1;
        }
    }

    template <typename It>
    Matrix(std::size_t size, It begin, It end) : size_(size) {
        array_ = new T[size_ * size_]();

        std::size_t counter = 0;
        for (auto it = begin; it != end; it++) {
            array_[counter++] = *it;
        }
    }

    Matrix(const Matrix& other) : size_(other.size_), array_(new T[other.size_ * other.size_]) {
        for (std::size_t i = 0; i < size_ * size_; i++) {
            array_[i] = other.array_[i];
        }
    }

    Matrix(Matrix&& other) noexcept : array_(other.array_), size_(other.size_) {
        other.array_ = nullptr;
    }

    Matrix& operator=(const Matrix& other) = delete;
    Matrix& operator=(const Matrix&& other) = delete;

    ~Matrix() {
         delete[] array_;
    }

    std::size_t GetSize() const {
        return size_;
    }

    const T* GetMatrix() const {
        return array_;
    }

    Matrix Transpose() const {
        Matrix res(*this);
        for (std::size_t i = 0; i < size_; i++) {
            for (std::size_t j = 0; j < size_; j++) {
                res[j][i] = array_[i][j];
            }
        }

        return res;
    }

    Matrix Negate() const {
        Matrix res(*this);
        for (std::size_t i = 0; i < size_ * size_; i++) {
            res.array_[i] = -array_[i];
        }

        return res;
    }

    Matrix MultiplyRow(std::size_t row, int mult) const {
        Matrix res(*this);
        for (std::size_t i = 0; i < size_; i++) {
            res.array_[i * size_ * row] = array_[i * size_ * row] * mult;
        }

        return res;
    }

    T GetTrace() const { // TODO
        T res = T{};
        for (std::size_t index = 0; index < size_; index++) {
            res += array_[index * size_ + index];
        }

        return res;
    }

    void Dump(std::ostream& stream) {
        for (std::size_t i = 0; i < size_; i++) {
            for (std::size_t j = 0; j < size_; j++) {
                stream << array_[i * size_ + j] << " ";
            }

            stream << "\n";
        }
    }

    double Determinant() const requires std::is_same_v<T, double> {
        Matrix res(*this);
        bool sign = false;
        for (std::size_t i = 0; i < size_; i++) {
            for (std::size_t k = i; k < size_; k++) {
                if (res.array_[k * size_ + i] != 0.0) {
                    SwapRows(res, i, k);
                    sign = (sign ? false : true);
                    break;
                }
            }

            if (res.array_[i * size_ + i] == T{}) {
                return 0.0;
            }

            for (std::size_t k = i + 1; k < size_; k++) {
                if (res.array_[k * size_ + i] != 0.0) {
                    SubMultRows(res, i, k, res.array_[k * size_ + i] / res.array_[i * size_ + i]);
                }
            }
        }

        res.Dump(std::cout);

        T result = 1.0;
        for (std::size_t i = 0; i < size_; i++) {
            result *= res.array_[i * size_ + i];
        }

        return result * (1 ? !sign : sign);
    }

private:

    Matrix& SwapRows(Matrix& matrix, std::size_t lhs, std::size_t rhs) const {
        for (std::size_t i = 0; i < size_; i++) {
            std::swap(matrix.array_[lhs * size_ + i], matrix.array_[rhs * size_ + i]);
        }

        return matrix;
    }

    Matrix& SubMultRows(Matrix& matrix, std::size_t lhs, std::size_t rhs, double mult) const {
        for (std::size_t i = 0; i < size_; i++) {
            matrix.array_[rhs * size_ + i] -= matrix.array_[lhs * size_ + i] * mult;
        }

        return matrix;
    }

    std::size_t size_ = 0;
    T* array_ = nullptr;
};

} // namespace Matrix
