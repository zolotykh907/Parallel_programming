#include <iostream>
#include <memory>
#include <chrono>
#include <omp.h>
#include <cmath>

double epsilon = 0.00001;  // эпсилон

std::unique_ptr<double[]> coefficients_matrix;  // матрица коэффициентов
std::unique_ptr<double[]> solutions_vector;     // вектор ответов на уравнения
int height = 4, width = 4;  // высота и ширина матрицы

char end_flag = 'f';  // флаг, указывающий на достижение указанной точности
double approximation_coefficient = 0.1;  // коэффициент приближения

double loss = 0;
int use_second_parallel_method = 0;

double cpuSecond()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() * 1e-9;
}

void check_accuracy(std::unique_ptr<double[]>& x, std::unique_ptr<double[]>& x_predict, int numThreads, double sum_of_residuals, double sum_of_squared_residuals)
{
    int is_switched = 0;
    double r = std::sqrt(sum_of_residuals) / std::sqrt(sum_of_squared_residuals);
    if (loss < r && loss != 0 && loss > 0.999999999) {
        approximation_coefficient = approximation_coefficient / 10;
        loss = 0;
        std::cout << "first " << loss << " " << approximation_coefficient << std::endl;
        for (int i = 0; i < width; i++) {
            x[i] = 0;
        }
    }
    else {
        if (std::abs(loss - r) < 0.00001 && is_switched == 0) {
            approximation_coefficient = approximation_coefficient * -1;
            is_switched = 1;
        }
        std::cout << " second " << loss << " " << approximation_coefficient << std::endl;
        loss = r;
        if (r < epsilon) end_flag = 't';
    }
}

std::unique_ptr<double[]> matrix_vector_product_omp(std::unique_ptr<double[]>& x, int numThreads, double sum_of_squared_residuals)
{
    std::unique_ptr<double[]> x_predict(new double[width]);
    double sum_of_residuals = 0;
#pragma omp parallel num_threads(numThreads)
    {
        double thread_sum = 0;
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int items_per_thread = height / num_threads;
        int lb = thread_id * items_per_thread;
        int ub = (thread_id == num_threads - 1) ? (height - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            x_predict[i] = 0;
            for (int j = 0; j < width; j++) {
                x_predict[i] += coefficients_matrix[i * width + j] * x[j];
            }
            x_predict[i] = x_predict[i] - solutions_vector[i];
            thread_sum += x_predict[i] * x_predict[i];
            x_predict[i] = x[i] - approximation_coefficient * x_predict[i];
        }
#pragma omp atomic
        sum_of_residuals += thread_sum;
    }
    check_accuracy(x, x_predict, numThreads, sum_of_residuals, sum_of_squared_residuals);
    return x_predict;
}

std::unique_ptr<double[]> matrix_vector_product_omp_second(std::unique_ptr<double[]>& x, int numThreads, double sum_of_squared_residuals)
{
    std::unique_ptr<double[]> x_predict(new double[width]);
    double sum_of_residuals = 0;
#pragma omp parallel num_threads(numThreads)
    {
#pragma omp for schedule(dynamic, int(height / (numThreads * 3))) nowait reduction(+:sum_of_residuals)
        for (int i = 0; i < height; i++) {
            x_predict[i] = 0;
            for (int j = 0; j < width; j++) {
                x_predict[i] += coefficients_matrix[i * width + j] * x[j];
            }
            x_predict[i] = x_predict[i] - solutions_vector[i];
            sum_of_residuals += x_predict[i] * x_predict[i];
            x_predict[i] = x[i] - approximation_coefficient * x_predict[i];
        }
    }
    check_accuracy(x, x_predict, numThreads, sum_of_residuals, sum_of_squared_residuals);
    return x_predict;
}

void run_parallel(int numThreads)
{
    std::unique_ptr<double[]> x(new double[width]);
    coefficients_matrix.reset(new double[height * width]);
    solutions_vector.reset(new double[height]);
    double sum_of_squared_residuals = 0;
    double time = cpuSecond();
#pragma omp parallel num_threads(numThreads)
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int items_per_thread = height / num_threads;
        int lb = thread_id * items_per_thread;
        int ub = (thread_id == num_threads - 1) ? (height - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < width; j++) {
                if (i == j) coefficients_matrix[i * width + j] = 2;
                else coefficients_matrix[i * width + j] = 1;
            }
            solutions_vector[i] = width + 1;
            x[i] = solutions_vector[i] / coefficients_matrix[i * width + i];
            sum_of_squared_residuals += solutions_vector[i] * solutions_vector[i];
        }
    }

    if (use_second_parallel_method == 0) {
        while (end_flag == 'f') {
            x = matrix_vector_product_omp(x, numThreads, sum_of_squared_residuals);
        }
    }
    else {
        while (end_flag == 'f') {
            x = matrix_vector_product_omp_second(x, numThreads, sum_of_squared_residuals);
        }
    }

    time = cpuSecond() - time; // время работы. начиная с инициализации

    std::cout << "Elapsed time (parallel): " << time << " sec." << std::endl;
}

int main(int argc, char **argv)
{
    int numThreads = 2;
    if (argc > 1)
        numThreads = std::atoi(argv[1]);
    if (argc > 2) {
        height = std::atoi(argv[2]);
        width = std::atoi(argv[2]);
    }
    if (argc > 3) {
        use_second_parallel_method = std::atoi(argv[3]);
    }
    run_parallel(numThreads);
    return 0;
}
