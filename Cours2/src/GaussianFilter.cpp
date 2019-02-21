#include "GaussianFilter.hpp"
#include <chrono>
#include <float.h>
#include <iostream>
#include <immintrin.h>

void scalarGaussianFilter(int size, float* coefficients, int iter) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector;
    float *result;
    
    double min_duration = DBL_MAX;
    vector = createRandomVector(size);

    for (auto it = 0; it < iter; it++) {
        start = std::chrono::high_resolution_clock::now();
        // Side effects
        result[0] = coefficients[1] * vector[0] + coefficients[2] * vector[1];
        result[size] = coefficients[1] * vector[size] + coefficients[0] * vector[size - 1];

        for (unsigned long int i = 1; i < size - 1; i++) {
            result[i] = coefficients[0] * vector[i - 1] + coefficients[1] * vector[i] + coefficients[2] * vector[i + 1];
        }
        end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        if (duration < min_duration) min_duration = duration;
    }
    std::cout << "Séquentiel : " << min_duration << " " << (min_duration / size) << std::endl;
}

void vectorialGaussianFilter(int size, float* coefficients, int iter) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector;
    float *result;

    float *coef1, *coef2, *coef3;

    double min_duration = DBL_MAX;
    vector = createRandomVector(size);

    for (auto it = 0; it < iter; it++) {
        start = std::chrono::high_resolution_clock::now();
        /* 
        Setting up coefficients vector
        This must be within clock, it's part of how I do the job
         */
        for (int i = 0; i < 8; i++) {
            coef1[i] = coefficients[0];
            coef2[i] = coefficients[1] / 2; // ! Important divided by 2
            coef3[i] = coefficients[2];
        }

        // Side effects
        result[0] = coefficients[1]*vector[0] + coefficients[2]*vector[1];
        result[size] = coefficients[1] * vector[size] + coefficients[0] * vector[size-1];
        
        // Coef mm256, loaded before
        __m256 m256Coef1 = _mm256_load_ps(coef1);
        __m256 m256Coef2 = _mm256_load_ps(coef2);
        __m256 m256Coef3 = _mm256_load_ps(coef3);

        for (unsigned long i = 1; i < size-1; i++) {
            __m256 vec = _mm256_mul_ps(_mm256_load_ps(&vector[i]), m256Coef2);
            __m256 vecBefore = _mm256_mul_ps(m256Coef1, _mm256_load_ps(&vector[i-1]));
            vec = _mm256_add_ps(vecBefore, vec);
            __m256 vecAfter = _mm256_mul_ps(m256Coef3, _mm256_load_ps(&vector[i + 1]));
            vec = _mm256_add_ps(vecAfter, vec);

            _mm256_storeu_ps(&result[i], vec);
        }
        end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        if (duration < min_duration)  min_duration = duration;
    }
    std::cout << "Vectorial : " << min_duration << " " << (min_duration / size) << std::endl;
}

int main(int argc, char* argv[]) {
    unsigned long int iter = atoi(argv[1]);
    unsigned long int size = 1048576;
    float *coefficients;
    coefficients[0] = 1;
    coefficients[1] = 2;
    coefficients[2] = 1;
    scalarGaussianFilter(size, coefficients, iter);
    
    return 0;
}