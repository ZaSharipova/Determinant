#pragma once

#include <cassert>
#include <cstddef>
#include <algorithm>

namespace Matrix {

template <typename T>
class JaggedStorage {
public:
    JaggedStorage(std::size_t size) : size_(size) {
        AllocateArray(size);
    }

    JaggedStorage(const JaggedStorage& other) : size_(other.GetSize()) {
        AllocateArray(GetSize());

        for (std::size_t i = 0; i < GetSize(); i++) {
            for (std::size_t j = 0; j < GetSize(); j++) {
                array_[i][j] = other.array_[i][j];
            }
        }
    }

    ~JaggedStorage() {
        for (std::size_t i = 0; i < GetSize(); i++) {
            delete[] array_[i];
        }

        delete[] array_;
    }

    JaggedStorage(JaggedStorage&& other) noexcept : size_(other.GetSize()), array_(other.array_) {
        other.array_ = nullptr;
    }

    JaggedStorage& operator=(const JaggedStorage& other) = delete;
    JaggedStorage& operator=(JaggedStorage&& other) = delete;

    T& operator[](std::size_t row, std::size_t col) {
        assert(row < GetSize());
        assert(col < GetSize());

        return array_[row][col];
    }

    const T& operator[](std::size_t row, std::size_t col) const {
        assert(row < GetSize());
        assert(col < GetSize());

        return array_[row][col];
    }

    std::size_t GetSize() const {
        return size_;
    }

    void SwapRows(std::size_t lhs, std::size_t rhs) {
        std::swap(array_[lhs], array_[rhs]);
    }

private:
    void AllocateArray(std::size_t size) {
        array_ = new T*[size]();

        for (std::size_t i = 0; i < size; i++) {
            array_[i] = new T[size]();
        }
    }

    std::size_t size_ = 0;
    T** array_ = nullptr;
};

} // namespace Matrix
