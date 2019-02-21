#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>

#define SIZE 1024*1024

void sequentiel(float* A, float* B, float* S, unsigned long int size) {
    for (unsigned long int i = 0; i < size; i++) {
        S[i] = A[i] + B[i];
    }
}

void parallele(float* A,float* B, float*S, unsigned long int size) {
    for (unsigned long i = 0; i < size; i+=8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        __m256 s = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&S[i],s);
    }
}

int main(int argc, char* argv[]) {
    unsigned long int iter = atoi(argv[1]);

    /* initialize random seed: */

    srand (time(NULL));

    const unsigned long int size = SIZE;

    // Création des données de travail
    float * A,* B,* S1,* S2;
    A = (float *) malloc(size * sizeof(float));
    B = (float *) malloc(size * sizeof(float));
    S1 = (float *) malloc(size * sizeof(float));
    S2 = (float *) malloc(size * sizeof(float));


    for (unsigned long int i = 0; i < size; i++) {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
    }

    /*** Validation ***/
    sequentiel(A,B,S1,size);
    parallele(A,B,S2,size);
    bool valide = false;
    for (unsigned long int i = 0; i < size; i++) {
        if(S1[i] == S2[i]) {
            valide = true;
        }
        else {
            valide = false;
            break;
        }
    }
    std::cout << "Le résultat est " << std::boolalpha << valide << std::endl;

    

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;
    for (auto it =0; it < iter; it++) {
        t0 = std::chrono::high_resolution_clock::now();
        sequentiel(A, B, S1, size);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }

    std::cout << "Séquentiel : " << min_duration << " " << (min_duration/size) << std::endl;

    min_duration = DBL_MAX;
    for (auto it =0; it < iter; it++) {
        t0 = std::chrono::high_resolution_clock::now();
        parallele(A, B, S2, size);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }

    std::cout << "Parallèle : " << min_duration << " " << (min_duration/size) << std::endl;

    // Libération de la mémoire : indispensable

    free(A);
    free(B);
    free(S1);
    free(S2);    

    return 0;
}
