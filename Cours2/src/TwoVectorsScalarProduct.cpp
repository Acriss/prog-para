#include "TwoVectorsScalarProduct.hpp"
#include <chrono>
#include <float.h>
#include <iostream>
#include <immintrin.h>

void scalarTwoVectorsScalarProduct(int size, int iter) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float *vector1;
    float *vector2;
    float scalarProduct = 0.0;
    double min_duration = DBL_MAX;
    vector1 = createRandomVector(size);
    vector2 = createRandomVector(size);


    for (auto it =0; it < iter; it++) {
        
        start = std::chrono::high_resolution_clock::now();

        for (unsigned long int i = 0; i < size; i++) {
            scalarProduct += (vector1[i] * vector2[i]);
        }
        end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end-start).count();
        if (duration < min_duration) min_duration = duration;
        
    }
    std::cout << "Séquentiel : " << min_duration << " " << (min_duration / size) << std::endl;
};

void vectorialTwoVectorsScalarProduct(int size, int iter)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector1;
    float *vector2;
    float *partialResult;
    float result = 0.0;
    double min_duration = DBL_MAX;
    vector1 = createRandomVector(size);
    vector2 = createRandomVector(size);
    partialResult = (float *) malloc(8 * sizeof(float));

    for (auto it =0; it < iter; it++) {
        
        start = std::chrono::high_resolution_clock::now();

        for (unsigned long i = 0; i < size; i+=8) {
            __m256 a = _mm256_loadu_ps(&vector1[i]);
            __m256 b = _mm256_loadu_ps(&vector2[i]);
            __m256 s = _mm256_mul_ps(a, b);

            __m256 t = _mm256_hadd_ps(s, s);
            __m256 u = _mm256_hadd_ps(t, t);
            _mm256_storeu_ps(partialResult, u);
            result += partialResult[0] + partialResult[4];
        }

        end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end-start).count();
        if (duration < min_duration) min_duration = duration;
    }
    std::cout << "Vectorial : " << min_duration << " " << (min_duration / size) << std::endl;
};

int main(int argc, char* argv[]) {
    unsigned long int iter = atoi(argv[1]);
    unsigned long int size = 1048576;
    scalarTwoVectorsScalarProduct(size, iter);
    vectorialTwoVectorsScalarProduct(size, iter);
    return 0;
};