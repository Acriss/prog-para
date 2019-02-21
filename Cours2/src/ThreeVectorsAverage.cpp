#include "ThreeVectorsAverage.hpp"
#include <chrono>
#include <float.h>
#include <iostream>
#include <immintrin.h>

void scalarThreeVectorsAverage(int size, int iter) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector1;
    float *vector2;
    float *vector3;
    float *result;
    double min_duration = DBL_MAX;
    vector1 = createRandomVector(size);
    vector2 = createRandomVector(size);
    vector3 = createRandomVector(size);
    result = (float *) malloc(size * sizeof(float));



    for (auto it =0; it < iter; it++) {

        start = std::chrono::high_resolution_clock::now();

        for (unsigned long int i = 0; i < size; i++) {
            result[i] = (vector1[i] + vector2[i] + vector3[i])/3;
        }

        end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end-start).count();
        if (duration < min_duration) min_duration = duration;
        
    }
    std::cout << "Séquentiel : " << min_duration << " " << (min_duration / size) << std::endl;
};

void vectorialThreeVectorsAverage(int size, int iter)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector1;
    float *vector2;
    float *vector3;
    float *result;
    float* threes;
    double min_duration = DBL_MAX;
    vector1 = createRandomVector(size);
    vector2 = createRandomVector(size);
    vector3 = createRandomVector(size);
    result = (float *) malloc(size * sizeof(float));
    threes = (float *) malloc(8 * sizeof(float));
    for (unsigned long i = 0; i < 8; i++) {
        threes[i] = 3.0;
    }



    for (auto it =0; it < iter; it++) {
        
        start = std::chrono::high_resolution_clock::now();

        for (unsigned long i = 0; i < size; i+=8) {
            __m256 a = _mm256_loadu_ps(&vector1[i]);
            __m256 b = _mm256_loadu_ps(&vector2[i]);
            __m256 s = _mm256_add_ps(a, b);

            __m256 c = _mm256_loadu_ps(&vector3[i]);
            __m256 v = _mm256_add_ps(s, c);
            __m256 three = _mm256_loadu_ps(threes);
            __m256 r = _mm256_div_ps(v, three);
            _mm256_storeu_ps(&result[i],r);
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
    scalarThreeVectorsAverage(size, iter);
    vectorialThreeVectorsAverage(size, iter);
    return 0;
};