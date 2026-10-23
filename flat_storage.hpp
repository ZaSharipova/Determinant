#pragma once

#include <cassert>
#include <cstddef>
#include <algorithm>

namespace Matrix {

template <typename T>
class FlatStorage {
public:
    FlatStorage(std::size_t size) : size_(size) {
        AllocateArray(size * size);
    }

    template <typename It>
    FlatStorage(std::size_t size, It begin, It end) : size_(size) {
        AllocateArray(size * size);

        std::size_t index = 0;
        for (auto it = begin; it != end; it++) {
            array_[index++] = *it;
        }
    }

    FlatStorage(const FlatStorage& other) : size_(other.GetSize()) {
        AllocateArray(GetSize() * GetSize());

        for (std::size_t i = 0; i < GetSize() * GetSize(); i++) {
            array_[i] = other.array_[i];
        }
    }

    ~FlatStorage() {
        delete[] array_;
    }

    FlatStorage(FlatStorage&& other) noexcept : size_(other.GetSize()), array_(other.array_) {
        other.array_ = nullptr;
    }

    FlatStorage& operator=(const FlatStorage& other) = delete;
    FlatStorage& operator=(FlatStorage&& other) = delete;

    T& operator[](std::size_t row, std::size_t col) {
        assert(row < GetSize());
        assert(col < GetSize());

        return array_[row * GetSize() + col];
    }

    const T& operator[](std::size_t row, std::size_t col) const {
        assert(row < GetSize());
        assert(col < GetSize());

        return array_[row * GetSize() + col];
    }

    std::size_t GetSize() const {
        return size_;
    }

    void SwapRows(std::size_t lhs, std::size_t rhs) {
        for (std::size_t i = 0; i < GetSize(); i++) {
            std::swap((*this)[lhs, i], (*this)[rhs, i]);
        }
    }

private:
    void AllocateArray(std::size_t capacity) {
        array_ = new T[capacity]();
    }

    std::size_t size_ = 0;
    T* array_ = nullptr;
};

} // namespace Matrix
