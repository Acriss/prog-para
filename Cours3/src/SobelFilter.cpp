#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

#define SIZE 4096*4096
#define STEP 4096

void sequentiel(float* A, float* B, unsigned long int size) {
    for (unsigned long int i = STEP; i < size-STEP; i++) {
        if (i % 4096 == 0 || i % 4095 == 0) {
            B[i] = 0;
            continue;
        }
        B[i] = (A[i-STEP-1]+A[i-STEP]+A[i-STEP+1]) - (A[i+STEP-1]+A[i+STEP]+A[i+STEP+1]);
    }
}

void sobel(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width, unsigned int nthreads)
{

    for (auto i = 1; i < height - 1; i++)
    {
        for (auto j = 1; j < width - 1; j++)
        {
            if ((i == 0) || (i == height - 1) || (j == 0) || (j == width - 1))
            {
                Resultat[i][j] = 0;
            }
            else
            {
                Resultat[i][j] = std::abs(Source[i - 1][j - 1] + Source[i - 1][j] + Source[i - 1][j + 1] - (Source[i + 1][j - 1] + Source[i + 1][j] + Source[i + 1][j + 1]));
                Resultat[i][j] += std::abs(Source[i - 1][j - 1] + Source[i][j - 1] + Source[i + 1][j - 1] - (Source[i - 1][j + 1] + Source[i][j + 1] + Source[i + 1][j + 1]));
            }
        }
    }
}

void parallele(float* A,float* B, unsigned long int size, int n_threads) {
    omp_set_num_threads(n_threads);
    #pragma omp parallel for
        for (unsigned long int i = STEP; i < size-STEP; i++) {
            if (i % 4096 == 0 || i % 4095 == 0) {
                B[i] = 0;
                continue;
            }
            B[i] = (A[i-STEP-1]+A[i-STEP]+A[i-STEP+1]) - (A[i+STEP-1]+A[i+STEP]+A[i+STEP+1]);
        }
    
}

void sobelParallele(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width, unsigned int nthreads)
{
#pragma omp parallel for
    for (auto i = 1; i < height - 1; i++)
    {
        for (auto j = 1; j < width - 1; j++)
        {
            if ((i == 0) || (i == height - 1) || (j == 0) || (j == width - 1))
            {
                Resultat[i][j] = 0;
            }
            else
            {
                Resultat[i][j] = std::abs(Source[i - 1][j - 1] + Source[i - 1][j] + Source[i - 1][j + 1] - (Source[i + 1][j - 1] + Source[i + 1][j] + Source[i + 1][j + 1]));
                Resultat[i][j] += std::abs(Source[i - 1][j - 1] + Source[i][j - 1] + Source[i + 1][j - 1] - (Source[i - 1][j + 1] + Source[i][j + 1] + Source[i + 1][j + 1]));
            }
        }
    }
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

    srand (time(NULL));

    const unsigned long int size = SIZE;

    // Création des données de travail
    float * A,* S1,* S2;
    A = (float *) malloc(size * sizeof(float));
    S1 = (float *) malloc(size * sizeof(float));
    S2 = (float *) malloc(size * sizeof(float));


    for (unsigned long int i = 0; i < size; i++) {
        A[i] = (float)(rand() % 360 - 180.0);
    }
    
    /*** Validation ***/
    sequentiel(A,S1, size);
    parallele(A,S2,size, 4);
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
        sequentiel(A, S1, size);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }

    std::cout << "Séquentiel : " << min_duration << " " << (min_duration/size) << std::endl;

    for (auto n_t=0; n_t < 5; n_t++) {
        min_duration = DBL_MAX;
        for (auto it =0; it < iter; it++) {
            t0 = std::chrono::high_resolution_clock::now();
            parallele(A, S2, size, pow(2, n_t));
            t1 = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(t1-t0).count();
            if (duration < min_duration) min_duration = duration;
        }
        std::cout << "Parallèle " << pow(2, n_t) << " : " << min_duration << " " << (min_duration/size) << std::endl;
    }


    // std::cout << omp_get_max_threads() << std::endl;

    // Libération de la mémoire : indispensable

    free(A);
    free(S1);   
    free(S2);
    return 0;
}
