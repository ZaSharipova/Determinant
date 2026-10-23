#include "flat_storage.hpp"
#include "jagged_storage.hpp"
#include "matrixes.hpp"

#include <iostream>
#include <ostream>
#include <vector>

int main() {
    std::vector<double> vec{1, 2, 3, 4};
    Matrix::Matrix<double, Matrix::FlatStorage<double>> matr1(2, vec.begin(), vec.end());
    Matrix::Matrix<double, Matrix::JaggedStorage<double>> matr2(2, vec.begin(), vec.end());
    std::cout << matr1.Determinant() << " " << matr2.Determinant();
}
