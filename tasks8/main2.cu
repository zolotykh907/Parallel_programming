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
    ctype* d_arr;

public:
    std::vector<ctype> arr;

    Data(int length) : len(length), arr(len), d_arr(nullptr) {
        cudaError_t err = cudaMalloc((void**)&d_arr, len * sizeof(ctype));
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ~Data() {
        if (d_arr) {
            cudaFree(d_arr);
        }
    }
    void copyToDevice() {
        cudaError_t err = cudaMemcpy(d_arr, arr.data(), len * sizeof(ctype), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory copy to device failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    void copyToHost() {
        cudaError_t err = cudaMemcpy(arr.data(), d_arr, len * sizeof(ctype), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory copy to host failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ctype* getDevicePointer() {
        return d_arr;
    }
};

void write_matrix(const double* curmatrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);
    int fieldWidth = 10;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << curmatrix[i * N + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void init_error(Data<double>& matr, int size){
    for (int i = 0; i < size*size; ++i){
        matr.arr[i] = 0.0;
    }
}

void init(Data<double>& curmatrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            curmatrix.arr[i * size + j] = 0;
        }
    }
    curmatrix.arr[0] = 10.0;
    curmatrix.arr[size - 1] = 20.0;
    curmatrix.arr[(size - 1) * size + (size - 1)] = 30.0;
    curmatrix.arr[(size - 1) * size] = 20.0;
    for (int i = 1; i < size - 1; ++i) {
        curmatrix.arr[i * size + 0] = linearInterpolation(i, 0.0, curmatrix.arr[0], size - 1, curmatrix.arr[(size - 1) * size]);
    }
    for (int i = 1; i < size - 1; ++i) {
        curmatrix.arr[0 * size + i] = linearInterpolation(i, 0.0, curmatrix.arr[0], size - 1, curmatrix.arr[size - 1]);
    }
    for (int i = 1; i < size - 1; ++i) {
        curmatrix.arr[(size - 1) * size + i] = linearInterpolation(i, 0.0, curmatrix.arr[(size - 1) * size], size - 1, curmatrix.arr[(size - 1) * size + (size - 1)]);
    }
    for (int i = 1; i < size - 1; ++i) {
        curmatrix.arr[i * size + (size - 1)] = linearInterpolation(i, 0.0, curmatrix.arr[size - 1], size - 1, curmatrix.arr[(size - 1) * size + (size - 1)]);
    }
}

__global__ void iterate(double* curmatrix, double* prevmatrix, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j == 0 || i == 0 || i >= size - 1 || j >= size - 1) return;

    curmatrix[i * size + j] = 0.25 * (prevmatrix[i * size + j + 1] + prevmatrix[i * size + j - 1] +
                                   prevmatrix[(i - 1) * size + j] + prevmatrix[(i + 1) * size + j]);
}

template <unsigned int blockSize>
__global__ void compute_error(double* curmatrix, double* prevmatrix, double* max_error, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= size || i >= size) return;

    __shared__ typename cub::BlockReduce<double, blockSize>::TempStorage temp_storage;
    double local_max = 0.0;

    if (j > 0 && i > 0 && j < size - 1 && i < size - 1) {
        int index = i * size + j;
        double error = fabs(curmatrix[index] - prevmatrix[index]);
        local_max = error;
    }

    double block_max = cub::BlockReduce<double, blockSize>(temp_storage).Reduce(local_max, cub::Max());

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(max_error), __double_as_longlong(block_max));
    }
}

int main(int argc, char const *argv[]) {
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

    auto start = std::chrono::high_resolution_clock::now();
    int size = vm["matr_size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["num_iter"].as<int>();

    Data<double> curmatrix(size * size);
    Data<double> prevmatrix(size * size);
    
    init(curmatrix, size);
    init(prevmatrix, size);

    double error;
    error = accuracy + 1;
    int iter = 0;
   
    dim3 blockDim(32, 32);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);
    //int totalBlocks = gridDim.x * gridDim.y;

    Data<double> d_max_error(gridDim.x * gridDim.y);

    d_max_error.copyToDevice();
    curmatrix.copyToDevice();
    prevmatrix.copyToDevice();

    double* curmatrix_adr = curmatrix.getDevicePointer();
    double* prevmatrix_adr = prevmatrix.getDevicePointer();
    double* d_max_error_adr = d_max_error.getDevicePointer();

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool graphCreated = false;

    cudaMemset(d_max_error_adr, 0, sizeof(double));

    while (iter < countIter && error > accuracy) {
        if (!graphCreated) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            
            for(int i = 0; i < 999; i++){
                iterate<<<gridDim, blockDim, 0, stream>>>(curmatrix_adr, prevmatrix_adr, size);
                std::swap(prevmatrix_adr, curmatrix_adr);
            }
            
            iterate<<<gridDim, blockDim, 0, stream>>>(curmatrix_adr, prevmatrix_adr, size);
            compute_error<32><<<gridDim, blockDim, 0, stream>>>(curmatrix_adr, prevmatrix_adr, d_max_error_adr, size);

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            
            graphCreated = true;
        } else {
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
            
            double temp_error;
            cudaMemcpy(&temp_error, d_max_error_adr, sizeof(double), cudaMemcpyDeviceToHost);
            error = temp_error;

            

            std::cout << "Iteration: " << iter + 1000 << ", Error: " << error << std::endl;
            
            iter += 1000;
            cudaMemset(d_max_error_adr, 0, sizeof(double));
        }
    }

    curmatrix.copyToHost();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "time: " << time_s << " error: " << error << " iteration: " << iter << std::endl;

    if (size <= 13) {
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                std::cout << curmatrix.arr[i * size + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    write_matrix(curmatrix.arr.data(), size, "matrix2.txt");

    cudaStreamDestroy(stream);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

    return 0;
}
