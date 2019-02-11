#include "TwoVectorsScalarProduct.h"

float *vector1;
float *vector2;

float scalarTwoVectorsScalarProduct(int size, int randomMaxBondary)
{
    time_t start, end;
    createRandomVector(size, vector1, randomMaxBondary);
    createRandomVector(size, vector2, randomMaxBondary);
    time(&start);
    // function goes here
    time(&end);
    return difftime(end, start);
};

float vectorialTwoVectorsScalarProduct(int size, int randomMaxBondary)
{
    time_t start, end;
    createRandomVector(size, vector1, randomMaxBondary);
    createRandomVector(size, vector2, randomMaxBondary);
    time(&start);
    // function goes here
    time(&end);
    return difftime(end, start);
};

int main()
{
    int size = 1048576;
    int randomMaxBondary = size;
    float scalarTimer = 0;
    float vectorialTimer = 0;
    scalarTimer = scalarTwoVectorsScalarProduct(size, randomMaxBondary);
    vectorialTimer = vectorialTwoVectorsScalarProduct(size, randomMaxBondary);

    printf("%.2f units of time for scalar average", scalarTimer);
    printf("%.2f units of time for vectorial average", vectorialTimer);
    return 0;
};