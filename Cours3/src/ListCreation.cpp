#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

#define SIZE 1024*1024

int sequentiel(int* M, int* S, unsigned long int size) {
    unsigned long int ne = 0;
    for (unsigned long int i = 0; i < size; i++) {
        if ( M[i] % 2 == 0 ) {
            S[ne] = M[i];
            ne += 1;
        }
    }
    return ne;
}

int parallele(int* M, int*S, unsigned long int size, int n_threads) {
    omp_set_num_threads(n_threads);
    unsigned long int ne = 0;
    int *T;
    T = (int *) malloc(size * sizeof(int));
    for (auto i = 0; i < size; i++) {
        T[i] = 200;
    }
    

    #pragma omp parallel shared(ne)
    {
        #pragma omp for reduction(+:ne) schedule(static)
        for (unsigned long int i = 0; i < size; i++) {
            if ( M[i] % 2 == 0 ) {
                T[i] = M[i];
                ne += 1;
            }        
        }
    }


    int ind = 0;
    for (auto i = 0; i < size; i++) {
        if (T[i] != 200) {
            S[ind] = T[i];
            ind += 1;
        }
    }
    free(T);

    return ne;

}

int parallele(int* M, int*S, unsigned long int size, int n_threads) {
    omp_set_num_threads(n_threads);
    unsigned long int ne = 0;

    #pragma omp parallel
    {
        int local_result=0;
        #pragma omp for schedule(static)
        for (unsigned long int i = 0; i < size; i++) {
            if (M[i] % 2 == 0) {
                local_result += 1;
            }
        }
        #pragma omp critical
            ne += local_result;
    }

    return ne;

}

void print_array(int* arr, int n_elems) {
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
    int n1, n2;
    int * M,* S1,* S2;
    M = (int *) malloc(size * sizeof(int));
    
    S1 = (int *) malloc(size * sizeof(int));
    S2 = (int *) malloc(size * sizeof(int));


    for (unsigned long int i = 0; i < size; i++) {
        M[i] = (int)(rand() % 360 - 180.0);
    }  

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;
    for (auto it =0; it < iter; it++) {
        t0 = std::chrono::high_resolution_clock::now();
        n1 = sequentiel(M, S1, size);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }

    std::cout << "Séquentiel : " << min_duration << " " << (min_duration/size) << std::endl;

    for (auto n_t=0; n_t < 5; n_t++) {
        min_duration = DBL_MAX;
        for (auto it =0; it < iter; it++) {
            t0 = std::chrono::high_resolution_clock::now();
            n2 = parallele(M, S2, size, pow(2, n_t));
            t1 = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(t1-t0).count();
            if (duration < min_duration) min_duration = duration;
        }
        std::cout << "Parallèle " << pow(2, n_t) << " : " << min_duration << " " << (min_duration/size) << std::endl;
    }
    
    print_array(S1, 20);
    print_array(S2, 20);
    /*** Validation ***/
    bool valide = false;
    for (unsigned long int i = 0; i < n1; i++) {
        if(S1[i] == S2[i]) {
            valide = true;
        }
        else {
            valide = false;
            break;
        }
    }
            

    if (n1 != n2) {
        valide =false;
        std::cout << "Le compteur est " << n1 << " et " << n2 << std::endl;
    }
    std::cout << "Le résultat est " << std::boolalpha << valide << std::endl;

  

    // std::cout << omp_get_max_threads() << std::endl;

    // Libération de la mémoire : indispensable

    free(M);
    free(S1);
    free(S2);    

    return 0;
}
