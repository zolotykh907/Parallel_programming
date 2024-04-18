#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <memory>
#include <iomanip>
double e = 0.001;  // эпсила

std::unique_ptr<double[]> A;  // матрица коэффициентов
std::unique_ptr<double[]> b;  // вектор ответов на уравнения
int m = 4, n = 4;  // высота на ширину

char end = 'f';  // флаг, достигнута указанная точность
double t = 0.1;  // коэффициент приближения

double loss = 0;
int parallel_loop = 0;

double cpuSecond() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void proverka(std::vector<double>& x, std::vector<double>& x_predict, int numThread, double bhg, double bhgv) {
    int sm = 0;
    double r = sqrt(bhg) / sqrt(bhgv);
    if (loss < r && loss != 0 && loss > 0.999999999) {
        t = t / 10;
        loss = 0;
        for (int i = 0; i < n; i++) {
            x[i] = 0;
        }
    } else {
        if (std::abs(loss - r) < 0.00001 && sm == 0) {
            t = t * -1;
            sm = 1;
        }
        loss = r;
        if (r < e)
            end = 't';
    }
}

std::vector<double> matrix_vector_product_omp(std::vector<double>& x, int numThread, double bhgv) {
    std::vector<double> x_predict(n);
    double bhg = 0;
#pragma omp parallel num_threads(numThread)
    {
        double aaaaaa = 0;
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            x_predict[i] = 0;
            for (int j = 0; j < n; j++) {
                x_predict[i] = x_predict[i] + A[i * n + j] * x[j];
            }
            x_predict[i] = x_predict[i] - b[i];
            aaaaaa += x_predict[i] * x_predict[i];
            x_predict[i] = x[i] - t * x_predict[i];
        }
#pragma omp atomic
        bhg += aaaaaa;
    }
    proverka(x, x_predict, numThread, bhg, bhgv);
    return x_predict;
}

void run_parallel(int numThread) {
    std::vector<double> x(n);
    A.reset(new double[m * n]);
    b.reset(new double[m]);
    double bhgv = 0;
    double time = cpuSecond();
#pragma omp parallel num_threads(numThread)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j)
                    A[i * n + j] = 2;
                else
                    A[i * n + j] = 1;
            }
            b[i] = n + 1;
            x[i] = b[i] / A[i * n + i];
            bhgv += b[i] * b[i];
        }
    }

    if (parallel_loop == 0) {
        while (end == 'f') {
            x = matrix_vector_product_omp(x, numThread, bhgv);
        }
    } else {
        while (end == 'f') {
            x = matrix_vector_product_omp(x, numThread, bhgv);
        }
    }

    time = cpuSecond() - time;  // время работы. начиная с инициализации

    std::cout << "Elapsed time (parallel): " << std::fixed << std::setprecision(6) << time << " sec.\n";
}

int main(int argc, char** argv) {
    int numThread = 2;
    if (argc > 1)
        numThread = atoi(argv[1]);
    if (argc > 2) {
        m = atoi(argv[2]);
        n = atoi(argv[2]);
    }
    if (argc > 3) {
        parallel_loop = atoi(argv[3]);
    }
    run_parallel(numThread);
    return 0;
}
