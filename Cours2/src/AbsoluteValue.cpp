#include "AbsoluteValue.hpp"
#include <chrono>
#include <float.h>
#include <iostream>
#include <immintrin.h>

    void
    scalarMinMax(int size, int iter)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector;
    float *result;
    double min_duration = DBL_MAX;
    vector = createRandomVector(size);

    for (auto it = 0; it < iter; it++)
    {
        start = std::chrono::high_resolution_clock::now();
        for (unsigned long i = 0; i < size; i++) {
            result[i] = vector[i] < 0 ? -vector[i]: vector[i];
        }
        end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        if (duration < min_duration) min_duration = duration;
    }
    std::cout << "Séquentiel : " << min_duration << " " << (min_duration / size) << std::endl;
};

void vectorialMinMax(int size, int iter)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector;
    double min_duration = DBL_MAX;
    vector = createRandomVector(size);

    for (auto it = 1; it < iter; it++)
    {

        start = std::chrono::high_resolution_clock::now();

        for (unsigned long i = 0; i < size; i += 8)
        {
            __m256 a = _mm256_loadu_ps(&vector[i]);
            
        }
        end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        if (duration < min_duration)
            min_duration = duration;
    }
    std::cout << "Vectorial : " << min_duration << " " << (min_duration / size) << std::endl;
};

int main(int argc, char *argv[])
{
    unsigned long int iter = atoi(argv[1]);
    unsigned long int size = 1048576;
    scalarMinMax(size, iter);
    vectorialMinMax(size, iter);
    return 0;
};