#pragma once

#include <cassert>
#include <cstddef>
#include <algorithm>

namespace Matrix {

template <typename T>
class JaggedStorage {
public:
    JaggedStorage(std::size_t size) : size_(size) {
        AllocateArray(GetSize());
    }

    JaggedStorage(const JaggedStorage& other) : size_(other.GetSize()) {
        AllocateArray(GetSize());
        std::copy(other.data_, other.data_ + GetSize() * GetSize(), data_);
    }

    template <typename It>
    JaggedStorage(std::size_t size, It begin, It end) : size_(size) {
        assert(std::distance(begin, end) == size_);
        AllocateArray(GetSize());
        std::copy(begin, end, data_);
    }

    ~JaggedStorage() {
        delete[] data_;
        delete[] array_;
    }

    JaggedStorage(JaggedStorage&& other) noexcept : size_(other.GetSize()), array_(other.array_), data_(other.data_) {
        other.array_ = nullptr;
        other.data_ = nullptr;
        other.size_ = 0; // doing this for no segfault in destructor
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
        array_ = new T*[GetSize()];
        data_ = new T[size * size]();

        for (std::size_t i = 0; i < size; i++) {
            array_[i] = data_ + i * size_;
        }
    }

    std::size_t size_ = 0;
    T** array_ = nullptr;

    T* data_ = nullptr;
};

} // namespace Matrix
