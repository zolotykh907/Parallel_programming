#include <iostream>
#include <cmath>
#include <vector>
using namespace std

#define PI 3.14159265358979323846

int main() {
    const int size = pow(10,7);

#ifdef DOUBLE
    vector<double> array(size);
#else
    vector<float> array(size);
#endif

    for (int i = 0; i < size; ++i) {
        double angle = (2 * PI * i) / size;
        array[i] = sin(angle);
    }

    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }

    cout << "Summa: " << sum << endl;

    return 0;
}
