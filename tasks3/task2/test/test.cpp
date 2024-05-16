 #include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        std::cout << "Введите название файла для проверки" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]); 
    if (!file.is_open())
    {
        std::cerr << "Такого файла не существует" << std::endl;
        return 1;
    }

    int all = 0;
    int accepted = 0;
    std::string line;
    while (std::getline(file, line)) 
    {
        std::istringstream iss(line);
        std::vector<std::string> tokens;

        std::string token;
        while (iss >> token)
        {
            tokens.push_back(token);
        }

      
        if (tokens[0] == "pow")
        {
            double base = std::stod(tokens[2]);
            double exponent = std::stod(tokens[5]);
            double result = std::pow(base, 2.0);
            std::cout << "pow " << base << " " << exponent << std::endl;
            std::cout << "Значение в файле = " << result << " Посчитанное значение = " << exponent << std::endl;
            if (std::abs(result - exponent) < 1e-4)
            {
                accepted++;
            }
            all++;
        }
        if (tokens[0] == "sinus")
        {
            double arg = std::stod(tokens[2]);
            double result = std::stod(tokens[5]);
            std::cout << "sin " << arg << std::endl;
            std::cout << "Значение в файле = " << std::sin(arg) << " Посчитанное значение = " << result << std::endl;
            if (std::abs(std::sin(arg) - result) < 1e-4)
            {
                accepted++;
            }
            all++;
        }
        if (tokens[0] == "sqrt")
        {
            double arg = std::stod(tokens[2]);
            double result = std::stod(tokens[5]);
            std::cout << "sqrt " << arg << std::endl;
            std::cout << "Значение в файле = " << std::sqrt(arg) << " Посчитанное значение = " << result << std::endl;
            if (std::abs(std::sqrt(arg) - result) < 1e-4)
            {
                accepted++;
            }
            all++;
        }
    }
    if (accepted == all)
    {
          std::cout << "Все тесты пройдены" << std::endl;
    }
    else
    {
        std::cout << "Были ошибки"<< std::endl;
    }
    

    file.close(); 
    return 0;
}