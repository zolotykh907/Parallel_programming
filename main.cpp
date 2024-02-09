#include <iostream>
#include <cmath>
#include <vector>

#define PI 3.14159265358979323846

int main() {
    const int size = 10000000;

#ifdef DOUBLE
    std::cout << "double" << std::endl;
    std::vector<double> array(size);
#else
    std::cout << "float" << std::endl;
    std::vector<float> array(size);
#endif

    for (int i = 0; i < size; ++i) {
        double angle = (2 * PI * i) / size;
        array[i] = std::sin(angle);
    }

    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }

    std::cout << "Sum: " << sum << std::endl;

    return 0;
}
