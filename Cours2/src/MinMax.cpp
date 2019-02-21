#include "MinMax.hpp"
#include <chrono>
#include <float.h>
#include <iostream>
#include <immintrin.h>

void scalarMinMax(int size, int iter) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector;
    float min = 0.0;
    float max = 0.0;
    double min_duration = DBL_MAX;
    vector = createRandomVector(size);

    for (auto it =0; it < iter; it++) {
        
        start = std::chrono::high_resolution_clock::now();

        for (unsigned long int i = 0; i < size; i++) {
            float element = vector[i];
            if ( element < min) {
                min = element;
            } else if ( element > max) {
                max = element;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end-start).count();
        if (duration < min_duration) min_duration = duration;
        
    }
    std::cout << "Séquentiel : " << min_duration << " " << (min_duration/size) << std::endl;
};

void vectorialMinMax(int size, int iter)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    float *vector;
    float *minVec;
    float* maxVec;
    float min = vector[0];
    float max = vector[0];
    double min_duration = DBL_MAX;
    vector = createRandomVector(size);
    minVec = (float *) malloc(8 * sizeof(float));
    maxVec = (float *) malloc(8 * sizeof(float));

    for (unsigned long i = 0; i < 8; i++) {
        //This is disgusting btw
        minVec[i] = vector[0];
        maxVec[i] = vector[0];
    }
    for (auto it =1; it < iter; it++) {
        
        start = std::chrono::high_resolution_clock::now();

        for (unsigned long i = 0; i < size; i+=8) {
            __m256 a = _mm256_loadu_ps(&vector[i]);
            __m256 minimum = _mm256_loadu_ps(minVec);
            minimum = _mm256_min_ps(a, minimum);
            __m256 maximum = _mm256_loadu_ps(maxVec);
            maximum = _mm256_max_ps(a, maximum);
            _mm256_storeu_ps(minVec, minimum);
            _mm256_storeu_ps(maxVec, maximum);
        }
        for (int i = 0; i < 8; i++) {
            if ( minVec[i] < min) {
                min = minVec[i];
            } else if ( maxVec[i] > max) {
                max = maxVec[i];
            }
        }

        end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end-start).count();
        if (duration < min_duration) min_duration = duration;
    }
    std::cout << "Vectorial : " << min_duration << " " << (min_duration/size) << std::endl;
};

int main(int argc, char* argv[]) {
    unsigned long int iter = atoi(argv[1]);
    unsigned long int size = 1048576;
    scalarMinMax(size, iter);
    vectorialMinMax(size, iter);
    return 0;
};