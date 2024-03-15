#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <inttypes.h>

double t1 = 0;
 double t2 = 0;

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
	for (int i = 0; i < m; i++) {
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
		//int k = sizeof(c);
		//int n = items_per_thread;
		//int id = threadid;
		//int ub = id*(k/n) + (k%n < id ? 1 : 0);
		for (int i = lb; i <= ub; i++) {
			c[i] = 0.0;
			for (int j = 0; j < n; j++)
				c[i] += a[i * n + j] * b[j];

		}
	}
}

void run_serial(int m, int n)
{
	double *a, *b, *c;
	a = malloc(sizeof(*a) * m * n);
	b = malloc(sizeof(*b) * n);
	c = malloc(sizeof(*c) * m);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			a[i * n + j] = i + j;

	}
	for (int j = 0; j < n; j++)
		b[j] = j;
	t1 = omp_get_wtime();
	matrix_vector_product(a, b, c, m, n);
	t1 = omp_get_wtime() - t1;
	printf("Elapsed time (serial): %.6f sec.\n", t1);
	free(a);
	free(b);
	free(c);
}

void run_parallel(int m, int n)
{
	double *a, *b, *c;
	// Allocate memory for 2-d array a[m, n]
	a = malloc(sizeof(*a) * m * n);
	b = malloc(sizeof(*b) * n);
	c = malloc(sizeof(*c) * m);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			a[i * n + j] = i + j;

	}
	for (int j = 0; j < n; j++)
		b[j] = j;

	t2 = omp_get_wtime();
	matrix_vector_product_omp(a, b, c, m, n);
	t2 = omp_get_wtime() - t2;

	printf("Elapsed time (parallel): %.6f sec.\n", t2);
	free(a);
	free(b);
	free(c);
}

int main(int argc, char **argv)
{
	int m;
	int n;

	for (int i = 0; i < argc; i++)
	{
		char* p;
		m = strtol(argv[1], &p, 10);
		n = strtol(argv[2], &p, 10);
	}

	printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
	printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
	run_serial(m, n);
	run_parallel(m,n);

	printf("SpeedUp = %.6f\n", (t1/t2));

	return 0;
}
