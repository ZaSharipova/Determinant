#pragma once

#include <cassert>
#include <algorithm>

namespace Matrix {
template <typename T>
class FlatStorage {
public:
    FlatStorage(std::size_t size) : size_(size) {
        AllocateArray(size * size);
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

private:
    void AllocateArray(std::size_t capacity) {
        array_ = new T[capacity]();
    }

    std::size_t size_ = 0;
    T* array_ = nullptr;
};

} // namespace Matrix
