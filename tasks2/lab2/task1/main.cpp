#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <inttypes.h>
#include <memory>

double t1 = 0;
double t2 = 0;

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_serial(int m, int n)
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

    t1 = omp_get_wtime();
    matrix_vector_product(a.get(), b.get(), c.get(), m, n);
    t1 = omp_get_wtime() - t1;
    printf("Elapsed time (serial): %.6f sec.\n", t1);
}

void run_parallel(int m, int n)
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

    t2 = omp_get_wtime();
    matrix_vector_product_omp(a.get(), b.get(), c.get(), m, n);
    t2 = omp_get_wtime() - t2;

    printf("Elapsed time (parallel): %.6f sec.\n", t2);
}

int main(int argc, char **argv)
{
    int m;
    int n;

    for (int i = 0; i < argc; i++)
    {
        char *p;
        m = strtol(argv[1], &p, 10);
        n = strtol(argv[2], &p, 10);
    }

    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
    run_serial(m, n);
    run_parallel(m, n);

    printf("SpeedUp = %.6f\n", (t1 / t2));

    return 0;
}
