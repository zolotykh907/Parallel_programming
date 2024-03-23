#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <memory>
#include <chrono>


const int MATRIX_SIZE = 20000;
const int NUM_THREADS = 4;

using MatrixPtr = std::shared_ptr<std::vector<std::vector<int>>>;
using VectorPtr = std::shared_ptr<std::vector<int>>;
MatrixPtr matrix;
VectorPtr vector;
std::vector<int> result(MATRIX_SIZE);

void initialize() {
    matrix = std::make_shared<std::vector<std::vector<int>>>(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
    vector = std::make_shared<std::vector<int>>(MATRIX_SIZE);

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            (*matrix)[i][j] = i + j;
        }
    }

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        (*vector)[i] = i;
    }
}

void multiply(int start, int end) {
    for (int i = start; i < end; ++i) {
        result[i] = std::inner_product((*matrix)[i].begin(), (*matrix)[i].end(), (*vector).begin(), 0);
    }
}

int main() {
    initialize();

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    int chunk_size = MATRIX_SIZE / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * chunk_size;
        int end = (i == NUM_THREADS - 1) ? MATRIX_SIZE : (i + 1) * chunk_size;
        threads.emplace_back(multiply, start, end);
    }

 
    for (auto& thread : threads) {
        thread.join();
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "time: " << duration.count() << " ms." << std::endl;

    return 0;
}
