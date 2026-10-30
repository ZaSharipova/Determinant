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
        assert(std::distance(begin, end) == size_);
        AllocateArray(size * size);
        std::copy(begin, end, array_);
    }

    FlatStorage(const FlatStorage& other) : size_(other.GetSize()) {
        AllocateArray(GetSize() * GetSize());
        std::copy(other.array_, other.array_ + GetSize() * GetSize(), array_);
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

    T* GetArray() const {
        return array_;
    }

    void SwapRows(std::size_t lhs, std::size_t rhs) {
        std::swap_ranges(GetArray() + GetSize() * lhs, GetArray() + GetSize() * (lhs + 1),
                         GetArray() + GetSize() * rhs);
    }

private:
    void AllocateArray(std::size_t capacity) {
        array_ = new T[capacity]();
    }

    void AllocateArray(std::size_t capacity, const T& elem) {
        AllocateArray(capacity);
        std::fill(array_, array_ + GetSize() * GetSize(), elem);
    }

    std::size_t size_ = 0;
    T* array_ = nullptr;
};

} // namespace Matrix
