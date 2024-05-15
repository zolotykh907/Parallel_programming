#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <fstream>
#include <iomanip>
#include </opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/nvtx3/nvToolsExt.h>
#include <omp.h>
namespace opt = boost::program_options;

template <class ctype>
class Data {
private:
    int len;
public:
    ctype* arr;
    Data(int length) {
        len = length;
        arr = new ctype[len];
#pragma acc enter data copyin(this)
#pragma acc enter data create(arr[0:len])
    }


    ~Data() {

#pragma acc exit data delete(arr)
#pragma acc exit data delete(this)
        delete[] arr;
        len = 0;
    }
};

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void initMatrices(Data<double>& curmatrix, int N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i == 0) {
                curmatrix.arr[i * N + j] = linearInterpolation(j, 0.0, curmatrix.arr[0], N - 1, curmatrix.arr[N - 1]);
            }
            else if (i == N - 1) {
                curmatrix.arr[i * N + j] = linearInterpolation(j, 0.0, curmatrix.arr[(N - 1) * N], N - 1, curmatrix.arr[(N - 1) * N + (N - 1)]);
            }
            else {
                if (j == 0) {
                    curmatrix.arr[i * N + j] = linearInterpolation(i, 0.0, curmatrix.arr[0], N - 1, curmatrix.arr[N - 1]);
                }
                else if (j == N - 1) {
                    curmatrix.arr[i * N + j] = linearInterpolation(i, 0.0, curmatrix.arr[N - 1], N - 1, curmatrix.arr[(N - 1) * N + (N - 1)]);
                }
                else {
                    curmatrix.arr[i * N + j] = 0.0;
                }
            }
        }
    }
    curmatrix.arr[0] = 10.0;
    curmatrix.arr[N - 1] = 20.0;
    curmatrix.arr[(N - 1) * N + (N - 1)] = 30.0;
    curmatrix.arr[(N - 1) * N] = 20.0;
}

void swap(Data<double>& curmatrix, Data<double>& prevmatrix, int N) {

#pragma acc enter data copyin(curmatrix.arr[0:N*N])
#pragma acc update device(curmatrix.arr[0:N*N])

    double* curData = curmatrix.arr;
    double* prevData = prevmatrix.arr;

    std::copy(curData, curData + N * N, prevData);

#pragma acc update device(prevmatrix.arr[0:N*N])
#pragma acc update self(prevmatrix.arr[0:N*N])
#pragma acc exit data delete(curmatrix.arr)

}

double errorCalc(Data<double>& prevmatrix, Data<double>& curmatrix, int N) {

    double error = 0.0;

#pragma acc enter data copyin(curmatrix.arr[0:N*N], prevmatrix.arr[0:N*N])
#pragma acc update device(curmatrix.arr[0:N*N], prevmatrix.arr[0:N*N])
#pragma acc parallel loop reduction(max:error)

    for (size_t i = 1; i < N - 1; i++)
    {
        for (size_t j = 1; j < N - 1; j++)
        {

            error = fmax(error, fabs(curmatrix.arr[i * N + j] - prevmatrix.arr[i * N + j]));
        }
    }

#pragma acc exit data delete(curmatrix.arr, prevmatrix.arr)
    return error;

}

void saveMatrixToFile(const Data<double>& matrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    int fieldWidth = 10;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix.arr[i * N + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}



int main(int argc, char const* argv[]) {

    opt::options_description desc("опции");
    desc.add_options()
        ("accuracy", opt::value<double>(), "точность")
        ("matrix_size", opt::value<int>(), "размер матрицы")
        ("iterations", opt::value<int>(), "количество операций")
        ("help", "помощь")
        ;

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int N = vm["matrix_size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int count_iter = vm["iterations"].as<int>();

    Data<double> curmatrix(N * N);
    initMatrices(std::ref(curmatrix), N);
    Data<double> prevmatrix(N * N);
    swap(std::ref(curmatrix), std::ref(prevmatrix), N);

    double error = 1.0;
    int iter = 0;

    double start = omp_get_wtime();
    while (count_iter > iter && error > accuracy)
    {
#pragma acc enter data copyin(curmatrix.arr[0:N*N], prevmatrix.arr[0:N*N])
#pragma acc update device(curmatrix.arr[0:N*N], prevmatrix.arr[0:N*N])
#pragma acc parallel loop 
        for (size_t i = 1; i < N - 1; i++) {

            for (size_t j = 1; j < N - 1; j++) {

                curmatrix.arr[i * N + j] = 0.25 * (prevmatrix.arr[i * N + j + 1] + prevmatrix.arr[i * N + j - 1] + prevmatrix.arr[(i - 1) * N + j] + prevmatrix.arr[(i + 1) * N + j]);
            }
        }
#pragma acc update self(curmatrix.arr[0:N*N])
#pragma acc exit data delete(curmatrix.arr, prevmatrix.arr)

        if ((iter + 1) % 1000 == 0) {
            error = errorCalc(std::ref(prevmatrix), std::ref(curmatrix), N);
            std::cout << "iteration: " << iter + 1 << ' ' << "error: " << std::setprecision(8) << error << std::endl;
        }
        swap(std::ref(curmatrix), std::ref(prevmatrix), N);
        iter++;
    }
    double end = omp_get_wtime();
    std::cout << end - start << std::endl;


    if (N <= 13) {

        for (size_t i = 0; i < N; i++) {

            for (size_t j = 0; j < N; j++) {

                std::cout << curmatrix.arr[i * N + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    saveMatrixToFile(std::ref(curmatrix), N, "matrix.txt");

}
