#include <immintrin.h>
#include <stdlib.h>

/*
Instead of creating and returning an array, we give a size and a pointer
and we allocate memory and populate the memory cases.
This is because returning a list would cause an unneccessary copy to be created for nothing,
which is an additionnal cost.

*/
void createRandomVector(int size, float* vector, int chosenRandMax) {
    // Attribute memory
    vector = (float *) malloc(size * sizeof(float));

    for (unsigned long i = 0; i < size; i++) {
        //s between 0 and chosenRandMax
        vector[i] = (float)rand()/(float)(RAND_MAX/(float)chosenRandMax);
    }

};
