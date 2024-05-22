#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

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

__global__ void iterate(double* curmatrix, double* prevmatrix, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j == 0 || i == 0 || i >= N - 1 || j >= N - 1) return;

    curmatrix[i * N + j] = 0.25 * (prevmatrix[i * N + j + 1] + prevmatrix[i * N + j - 1] +
                                   prevmatrix[(i - 1) * N + j] + prevmatrix[(i + 1) * N + j]);
}

template <unsigned int blockSize>
__global__ void compute_error(double* curmatrix, double* prevmatrix, double* max_error, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= N || i >= N) return;

    __shared__ typename cub::BlockReduce<double, blockSize>::TempStorage temp_storage;
    double local_max = 0.0;

    if (j > 0 && i > 0 && j < N - 1 && i < N - 1) {
        int index = i * N + j;
        double error = fabs(curmatrix[index] - prevmatrix[index]);
        local_max = error;
    }

    double block_max = cub::BlockReduce<double, blockSize>(temp_storage).Reduce(local_max, cub::Max());
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicMax(max_error, block_max);
    }
    
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

    double* d_curmatrix;
    double* d_prevmatrix;
    double* d_max_error;
    
    cudaMalloc(&d_curmatrix, N * N * sizeof(double));
    cudaMalloc(&d_prevmatrix, N * N * sizeof(double));
    cudaMalloc(&d_max_error, sizeof(double));
    cudaMemcpy(d_curmatrix, curmatrix, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevmatrix, prevmatrix, N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); //сколько блоков

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    bool graphCreated = false;

    auto start = std::chrono::high_resolution_clock::now();

    while (iter < countIter && error > accuracy) {
        if (!graphCreated) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            iterate<<<gridDim, blockDim, 0, stream>>>(d_curmatrix, d_prevmatrix, N);
            cudaMemset(d_max_error, 0, sizeof(double));
            compute_error<32><<<gridDim, blockDim, 0, stream>>>(d_curmatrix, d_prevmatrix, d_max_error, N);

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            cudaStreamDestroy(stream);
            graphCreated = true;
        }

        cudaGraphLaunch(graphExec, 0);
        cudaMemcpy(&error, d_max_error, sizeof(double), cudaMemcpyDeviceToHost);

        if (iter % 1000 == 0) {
            std::cout << "iteration: " << iter + 1 << " error: " << error << std::endl;
        }

        std::swap(d_prevmatrix, d_curmatrix);
        iter++;
    }

    cudaMemcpy(A.arr.data(), d_curmatrix, N * N * sizeof(double), cudaMemcpyDeviceToHost);

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

    cudaFree(d_curmatrix);
    cudaFree(d_prevmatrix);
    cudaFree(d_max_error);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

    return 0;
}
