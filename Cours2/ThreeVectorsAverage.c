#include "ThreeVectorsAverage.h"

float *vector1;
float *vector2;
float *vector3;

float scalarThreeVectorsAverage(int size, int randomMaxBondary) {
    time_t start, end;
    createRandomVector(size, vector1, randomMaxBondary);
    createRandomVector(size, vector2, randomMaxBondary);
    createRandomVector(size, vector3, randomMaxBondary);
    time(&start);
// function goes here
    time(&end);
    return difftime(end, start);
};

float vectorialThreeVectorsAverage(int size, int randomMaxBondary)
{
    time_t start, end;
    createRandomVector(size, vector1, randomMaxBondary);
    createRandomVector(size, vector2, randomMaxBondary);
    createRandomVector(size, vector3, randomMaxBondary);
    time(&start);
    // function goes here
    time(&end);
    return difftime(end, start);
};

int main() {
    int size = 1048576;
    int randomMaxBondary = size;
    float scalarTimer = 0;
    float vectorialTimer = 0;
    scalarTimer = scalarThreeVectorsAverage(size, randomMaxBondary);
    vectorialTimer = vectorialThreeVectorsAverage(size, randomMaxBondary);

    printf("%.2f units of time for scalar average", scalarTimer);
    printf("%.2f units of time for vectorial average", vectorialTimer);
    return 0;
};