#include "matrixes.hpp"

#include <iostream>
#include <ostream>
#include <vector>

int main() {
    std::vector<double> vec{1, 2, 3, 4};
    Matrix::Matrix<double> matr(2, vec.begin(), vec.end());

    matr.Dump(std::cout);
    std::cout << matr.GetSize() << "\n";

    std::cout << matr.Determinant();
}
