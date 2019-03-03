#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

#define SIZE 1024*1024

const int boundaries[] = { 500000,600000 };

float sequentiel(float* A, float* B, unsigned long int size) {
    float s=0;
    for (unsigned long int i = 0; i < size; i++) {
        if (i > boundaries[0] && i < boundaries[1]) {
            s += A[i] * B[i];
        }
       
    }
    return s;
}

float parallele(float* A,float* B, unsigned long int size, int n_threads) {
    float s = 0;
    omp_set_num_threads(n_threads);
    #pragma omp parallel
    {
        float local_result=0;
        #pragma omp for schedule(auto)
        for (unsigned long int i = 0; i < size; i++) {
            if (i > boundaries[0] && i < boundaries[1]) {
                local_result += A[i] * B[i];
            }
        }
        #pragma omp critical
            s += local_result;
    }
    
    return s;
}

void print_array(float* arr, int n_elems) {
    for (auto j = 0; j < n_elems; j++) {
        std::cout << arr[j] << " ";
    }
    std::cout << std::endl;
    return;
}

int main(int argc, char* argv[]) {
    unsigned long int iter = atoi(argv[1]);

    /* initialize random seed: */

    srand (1551086793);

    const unsigned long int size = SIZE;

    // Création des données de travail
    float * A,* B;
    A = (float *) malloc(size * sizeof(float));
    B = (float *) malloc(size * sizeof(float));
    

    for (unsigned long int i = 0; i < size; i++) {
        A[i] = (float)(rand() % 360 - 180.0);
        B[i] = (float)(rand() % 360 - 180.0);
    }
    
    float sum1, sum2;
    // print_array(A, 10);
    // print_array(B,10);

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;
    for (auto it =0; it < iter; it++) {
        t0 = std::chrono::high_resolution_clock::now();
        sum1 = sequentiel(A, B, size);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }

    std::cout << "Séquentiel : " << min_duration << " " << (min_duration/size) << std::endl;

    for (auto n_t=0; n_t < 4; n_t++) {
        min_duration = DBL_MAX;
        for (auto it =0; it < iter; it++) {
            t0 = std::chrono::high_resolution_clock::now();
            sum2 = parallele(A, B, size, pow(2, n_t));
            t1 = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(t1-t0).count();
            if (duration < min_duration) min_duration = duration;
        }
        std::cout << "Parallèle " << pow(2, n_t) << " : " << min_duration << " " << (min_duration/size) << std::endl;
    }
    
     /*** Validation ***/
    bool valide = (sum1 == sum2);
    std::cout << sum1 << " " << sum2 << std::endl;
    
    std::cout << "Le résultat est " << std::boolalpha << valide << std::endl;


    // std::cout << omp_get_max_threads() << std::endl;

    // Libération de la mémoire : indispensable

    free(A);
    free(B);   

    return 0;
}
