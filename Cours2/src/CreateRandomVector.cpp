#include <immintrin.h>
#include <stdlib.h>
#include <random>

/*
Instead of creating and returning an array, we give a size and a pointer
and we allocate memory and populate the memory cases.
This is because returning a list would cause an unneccessary copy to be created for nothing,
which is an additionnal cost.

*/
float* createRandomVector(int size) {
    srand (time(NULL));
    // Attribute memory
    float *vector = NULL;
    vector = (float *) malloc(size * sizeof(float));

    for (unsigned long i = 0; i < size; i++) {

        vector[i] = (float)(rand() % 360 - 180.0);
    }
    return vector;
};
