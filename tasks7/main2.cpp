#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "cublas_v2.h"


namespace opt = boost::program_options;

template <class ctype>
class Data {
private:
    int len;
public:
    std::vector<ctype> arr;
    Data(int length) : len(length), arr(len) {
#pragma acc enter data copyin(this)
#pragma acc enter data create(arr[0:len])
    }
    ~Data() {
#pragma acc exit data delete(arr)
#pragma acc exit data delete(this)
    }
};

double linear_interpolation(double x, double x1, double y1, double x2, double y2) {
    
    double res = y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
    return res;
}

void create_matrix(std::vector<double>& arr, int N) {
    arr[0] = 10.0;
    arr[N - 1] = 20.0;
    arr[(N - 1) * N + (N - 1)] = 30.0;
    arr[(N - 1) * N] = 20.0;

    for (size_t i = 1; i < N - 1; i++) {
        arr[0 * N + i] = linear_interpolation(i, 0.0, arr[0], N - 1, arr[N - 1]);
        arr[i * N + 0] = linear_interpolation(i, 0.0, arr[0], N - 1, arr[(N - 1) * N]);
        arr[i * N + (N - 1)] = linear_interpolation(i, 0.0, arr[N - 1], N - 1, arr[(N - 1) * N + (N - 1)]);
        arr[(N - 1) * N + i] = linear_interpolation(i, 0.0, arr[(N - 1) * N], N - 1, arr[(N - 1) * N + (N - 1)]);
    }
}

void write_matrix(const double* matrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);

    int fieldWidth = 10;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * N + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

double sum_elem(double a, double b, double c, double d) {
    return (a + b + c + d) / 4;
}

int main(int argc, char const* argv[]) {
    opt::options_description desc("Arguments");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "accuracy")
        ("matr_size", opt::value<int>()->default_value(256), "matrix_size")
        ("num_iter", opt::value<int>()->default_value(1000000), "num_iterations")
        ("help", "help");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int N = vm["matr_size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["num_iter"].as<int>();

    double error = 1.0;
    int iter = 0;

    Data<double> A(N * N);
    Data<double> B(N * N);

    create_matrix(A.arr, N);
    create_matrix(B.arr, N);

    double* curmatrix = A.arr.data();
    double* prevmatrix = B.arr.data();
    
    cublasHandle_t handle;
    cublasCreate(&handle);


double alpha = -1.0;
int idx=0;

auto start = std::chrono::high_resolution_clock::now();

#pragma acc data copyin(idx, alpha, prevmatrix[0:N*N],curmatrix[0:N*N], N)
    {
        while (iter < countIter && iter < 10000000 && error > accuracy) {
#pragma acc parallel loop independent collapse(2) present(curmatrix,prevmatrix)
            for (size_t i = 1; i < N - 1; i++) {
                for (size_t j = 1; j < N - 1; j++) {
                    curmatrix[i * N + j] = sum_elem(prevmatrix[i * N + j + 1], prevmatrix[i * N + j - 1], prevmatrix[(i - 1) * N + j], prevmatrix[(i + 1) * N + j]);
                }
            }

            if ((iter + 1) % 1000 == 0) {
#pragma acc host_data use_device(curmatrix,prevmatrix)
{
                cublasDaxpy(handle, N*N, &alpha, curmatrix, 1, prevmatrix, 1);
                cublasIdamax(handle, N*N, prevmatrix, 1, &idx);
}        
#pragma acc update self(prevmatrix[idx])
                error = fabs(prevmatrix[idx]);
                std::cout << "iteration: " << iter + 1 << ' ' << "error: " << error << std::endl;
#pragma acc host_data use_device(curmatrix,prevmatrix)
                cudaMemcpy(prevmatrix, curmatrix, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
            }

            std::swap(prevmatrix, curmatrix);
            iter++;
        }
        cublasDestroy(handle);
#pragma acc update self(curmatrix[0:N*N])
    }

    

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "time: " << time_s << " error: " << error << " iteration: " << iter << std::endl;

    if (N <= 13) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                std::cout << A.arr[i * N + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    write_matrix(A.arr.data(), N, "matrix2.txt");

    return 0;
}