#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <memory>

int count = 40;

double cpuSecond()
{
    using namespace std::chrono;
    system_clock::time_point now = system_clock::now();
    system_clock::duration tp = now.time_since_epoch();
    return duration_cast<duration<double>>(tp).count();
}

void matrix_vector_product(double* a, double* b, double* c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_thread(double* a, double* b, double* c, int m, int n, int threadid, int items_per_thread)
{
    int lb = threadid * items_per_thread;
    int ub = (threadid == count - 1) ? (m - 1) : (lb + items_per_thread - 1);

    for (int i = lb; i <= ub; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_serial(size_t n, size_t m, double& t_serial)
{
    std::unique_ptr<double[]> a(new double[m * n]);
    std::unique_ptr<double[]> b(new double[n]);
    std::unique_ptr<double[]> c(new double[m]);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    t_serial = cpuSecond();
    matrix_vector_product(a.get(), b.get(), c.get(), m, n);
    t_serial = cpuSecond() - t_serial;

    std::cout << "Elapsed time (serial): " << t_serial << " sec." << std::endl;
}

void run_parallel(size_t n, size_t m, double& t_parallel)
{
    std::unique_ptr<double[]> a(new double[m * n]);
    std::unique_ptr<double[]> b(new double[n]);
    std::unique_ptr<double[]> c(new double[m]);

    // Заполнение a и b данными

    double t_start = cpuSecond(); // Замеряем время до запуска потоков

    std::vector<std::thread> threads;
    int items_per_thread = m / count;

    for (int i = 0; i < count; i++)
    {
        threads.push_back(std::thread(matrix_vector_product_thread, a.get(), b.get(), c.get(), m, n, i, items_per_thread));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    double t_end = cpuSecond(); // Замеряем время после завершения всех потоков

    std::cout << "Elapsed time (parallel): " << t_end - t_start << " sec." << std::endl;

    t_parallel = t_end - t_start;
}

int main(int argc, char* argv[])
{
    size_t M = 1000;
    size_t N = 1000;
    if (argc > 1)
        M = atoi(argv[1]);
        N = atoi(argv[1]);
    if (argc > 2)
        count = atoi(argv[2]);

    double t_serial, t_parallel;
    run_serial(M, N, t_serial);
    run_parallel(M, N, t_parallel);

    double speedup = t_serial / t_parallel;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    return 0;
}
