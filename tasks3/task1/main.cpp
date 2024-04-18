#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>

const int ROWS = 20000;
const int COLS = 20000;

// Функция для заполнения матрицы случайными значениями в каждом потоке
void fillMatrix(std::vector<std::vector<int>>& matrix, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < COLS; ++j) {
            matrix[i][j] = rand() % 10; // Заполнение случайным числом от 0 до 9
        }
    }
}

// Функция для умножения матрицы на вектор в каждом потоке
void matrixVectorMultiplication(std::vector<std::vector<int>>& matrix, std::vector<int>& vector,
                                std::vector<int>& result, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        int sum = 0;
        for (int j = 0; j < COLS; ++j) {
            sum += matrix[i][j] * vector[j];
        }
        result[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }

    int numThreads = std::stoi(argv[1]);

    std::vector<std::vector<int>> matrix(ROWS, std::vector<int>(COLS));
    std::vector<int> vector(COLS);
    std::vector<int> result(ROWS);

    // Запоминаем время начала выполнения
    auto start = std::chrono::steady_clock::now();

    // Создание потоков для параллельного заполнения матрицы
    std::vector<std::thread> fillThreads;
    for (int i = 0; i < ROWS; ++i) {
        fillThreads.emplace_back(fillMatrix, std::ref(matrix), i, i + 1);
    }

    // Ожидание завершения всех потоков заполнения матрицы
    for (auto& thread : fillThreads) {
        thread.join();
    }

    // Заполнение вектора случайными значениями
    for (int i = 0; i < COLS; ++i) {
        vector[i] = rand() % 10; // Заполнение случайным числом от 0 до 9
    }

    // Создание потоков для параллельного умножения матрицы на вектор
    std::vector<std::thread> multiplicationThreads;
    int rowsPerThread = ROWS / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == numThreads - 1) ? ROWS : (i + 1) * rowsPerThread;
        multiplicationThreads.emplace_back(matrixVectorMultiplication, std::ref(matrix), std::ref(vector),
                                            std::ref(result), startRow, endRow);
    }

    // Ожидание завершения всех потоков умножения матрицы на вектор
    for (auto& thread : multiplicationThreads) {
        thread.join();
    }

    // Запоминаем время окончания выполнения
    auto end = std::chrono::steady_clock::now();

    // Вывод времени выполнения
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;

    return 0;
}
