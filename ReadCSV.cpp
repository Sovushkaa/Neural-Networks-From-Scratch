#include "ReadCSV.h"
#include <cassert>
#include <fstream>

namespace CNetworks {
    std::pair<Matrix, Matrix> ReadCSV(const std::string &filename) {
        int input_size = 784;
        int output_size = 10;
        std::vector<Vector> x_columns;
        std::vector<Vector> y_columns;
        std::ifstream file(filename);
        std::string cur_line, cur_label, cur_pixel;
        std::getline(file, cur_line);
        while (std::getline(file, cur_line)) {
            std::stringstream ss(cur_line);
            std::getline(ss, cur_label, ',');
            int correct_value = std::stoi(cur_label);
            y_columns.push_back(Vector::Zero(output_size));
            y_columns.back()[correct_value] = 1;
            x_columns.push_back(Vector::Zero(input_size));
            int cur_index = 0;
            while (std::getline(ss, cur_pixel, ',')) {
                x_columns.back()[cur_index] = std::stod(cur_pixel);
                ++cur_index;
            }
        }
        assert(x_columns.size() == y_columns.size());
        Matrix x = Matrix::Zero(input_size, x_columns.size());
        Matrix y = Matrix::Zero(output_size, y_columns.size());
        for (Index i = 0; i < x.cols(); ++i) {
            x.col(i) = std::move(x_columns[i]);
            y.col(i) = std::move(y_columns[i]);
        }
        return {x, y};
    }
} // namespace CNetworks

